use std::{env, sync::Arc, time::Duration};

use futures::{stream::FuturesUnordered, StreamExt};
use onnx_bert::{Entity, Pipeline};
use opentelemetry::{
    sdk::{propagation::TraceContextPropagator, trace::Sampler, Resource},
    KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use tokio::{
    select,
    sync::{mpsc, oneshot},
    task::{spawn_blocking, JoinError, JoinHandle},
    time::sleep,
};
use tokio_rayon::{
    rayon::{ThreadPool, ThreadPoolBuilder},
    AsyncThreadPool,
};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{debug, error, info, instrument, metadata::LevelFilter, Instrument, Span};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use trast_proto::{
    trast_server::{Trast, TrastServer},
    NerInput, NerOutput,
};

use crate::trace::TraceLayer;

mod trace;

const PIPELINE_TTL: Duration = Duration::from_secs(60);

struct TrastService {
    actor_tx: mpsc::Sender<Message>,
}

#[tonic::async_trait]
impl Trast for TrastService {
    async fn ner(&self, request: Request<NerInput>) -> Result<Response<NerOutput>, Status> {
        let NerInput { sentence } = request.into_inner();

        let (tx, rx) = oneshot::channel();
        self.actor_tx
            .send(Message {
                sentence,
                tx,
                span: Span::current(),
            })
            .await
            .unwrap();

        let entities = rx.await.unwrap()?.into_iter().map(
            |onnx_bert::Entity {
                 label,
                 score,
                 word,
                 start,
                 end,
             }| trast_proto::Entity {
                label,
                score,
                word,
                start: start.try_into().unwrap(),
                end: end.try_into().unwrap(),
            },
        );

        Ok(Response::new(NerOutput {
            entities: entities.into_iter().map(Into::into).collect(),
        }))
    }
}

#[derive(Debug)]
struct Message {
    sentence: String,
    tx: oneshot::Sender<Result<Vec<Entity>>>,
    span: Span,
}

type Result<T, E = Error> = core::result::Result<T, E>;

impl From<Error> for Status {
    fn from(value: Error) -> Self {
        Self::internal(value.to_string())
    }
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("{0}")]
    Join(#[from] JoinError),
    #[error("{0}")]
    Bert(#[from] onnx_bert::Error),
}

type Handles = FuturesUnordered<JoinHandle<()>>;

#[instrument]
async fn get_pipeline() -> Result<Pipeline> {
    let span = Span::current();
    let pipeline = spawn_blocking(move || {
        span.in_scope(|| Pipeline::from_pretrained("amcoff/bert-based-swedish-cased-ner"))
    })
    .await??;
    Ok(pipeline)
}

#[instrument(skip_all, fields(cold))]
async fn spawn_ner_task(
    sentence: String,
    cb: oneshot::Sender<Result<Vec<Entity>>>,
    pipeline: &mut Option<Arc<Pipeline>>,
    threadpool: &Arc<ThreadPool>,
) -> Option<JoinHandle<()>> {
    tracing::Span::current().record("cold", pipeline.is_none());

    if pipeline.is_none() {
        debug!("initializing pipeline");

        match get_pipeline().await {
            Ok(p) => *pipeline = Some(Arc::new(p)),
            Err(e) => {
                let _ = cb.send(Err(e));
                return None;
            }
        }

        debug!("initialized pipeline");
    }

    let pipeline = Arc::clone(pipeline.as_ref().unwrap());
    let threadpool = threadpool.clone();

    debug!("recognizing entities");

    let handle = tokio::spawn(
        async move {
            let span = Span::current();
            match threadpool
                .spawn_fifo_async(move || span.in_scope(|| pipeline.predict(sentence)))
                .await
            {
                Ok(entities) => {
                    let _ = cb.send(Ok(entities));
                }
                Err(e) => {
                    error!(?e);
                    let _ = cb.send(Err(e.into()));
                }
            };
        }
        .in_current_span(),
    );

    Some(handle)
}

async fn wait(handles: &mut Handles) {
    while handles.next().await.is_some() {}
    sleep(PIPELINE_TTL).await;
}

fn act(threadpool: ThreadPool) -> mpsc::Sender<Message> {
    let (tx, mut rx) = mpsc::channel::<Message>(16);
    let threadpool = Arc::new(threadpool);
    let mut pipeline = None;
    let mut handles = FuturesUnordered::new();

    tokio::spawn(async move {
        loop {
            select! {
                Some(Message { sentence, tx, span }) = rx.recv() => {
                    if let Some(handle) = spawn_ner_task(sentence, tx, &mut pipeline, &threadpool).instrument(span).await {
                        handles.push(handle);
                    }
                }
                _ = wait(&mut handles) => if pipeline.take().is_some() {
                    info!("dropped pipeline");
                }
            }
        }
    });

    tx
}

fn init_telemetry(otlp_endpoint: impl Into<String>) -> anyhow::Result<()> {
    let exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint(otlp_endpoint);

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(exporter)
        .with_trace_config(
            opentelemetry::sdk::trace::config()
                .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
                .with_resource(Resource::new(vec![
                    KeyValue::new(
                        opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                        env!("CARGO_PKG_NAME"),
                    ),
                    KeyValue::new(
                        opentelemetry_semantic_conventions::resource::SERVICE_VERSION,
                        env!("CARGO_PKG_VERSION"),
                    ),
                ])),
        )
        .install_batch(opentelemetry::runtime::Tokio)?;

    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .with(fmt::layer())
        .with(otel_layer)
        .init();

    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

    Ok(())
}

#[tokio::main]
async fn main() {
    let _ = dotenv::dotenv();
    let otlp_endpoint =
        env::var("OTLP_ENDPOINT").unwrap_or_else(|_| "http://localhost:4317".to_owned());
    let num_threads = env::var("NUM_WORKER_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    init_telemetry(otlp_endpoint).unwrap();

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<TrastServer<TrastService>>()
        .await;

    let threadpool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let trast = TrastService {
        actor_tx: act(threadpool),
    };

    let addr = "0.0.0.0:8000".parse().unwrap();

    info!("listening on {addr}");

    let trace_layer = tower::ServiceBuilder::new()
        .layer(TraceLayer::default())
        .into_inner();

    Server::builder()
        .layer(trace_layer)
        .add_service(health_service)
        .add_service(TrastServer::new(trast))
        .serve(addr)
        .await
        .unwrap();
}
