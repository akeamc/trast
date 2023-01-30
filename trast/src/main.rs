use std::{sync::Arc, time::Duration};

use futures::{stream::FuturesUnordered, StreamExt};
use onnx_bert::{Entity, Pipeline};
use tokio::{
    select,
    sync::{mpsc, oneshot},
    task::{spawn_blocking, JoinError, JoinHandle},
    time::sleep,
};
use tonic::{transport::Server, Request, Response, Status};
use tracing::{debug, info, instrument};
use trast_proto::{
    trast_server::{Trast, TrastServer},
    NerInput, NerOutput,
};

struct TrastService {
    actor_tx: mpsc::Sender<Message>,
}

#[tonic::async_trait]
impl Trast for TrastService {
    async fn ner(&self, request: Request<NerInput>) -> Result<Response<NerOutput>, Status> {
        let (tx, rx) = oneshot::channel();
        let NerInput { sentence } = request.into_inner();
        self.actor_tx.send((sentence, tx)).await.unwrap();
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

type Message = (String, oneshot::Sender<Result<Vec<Entity>>>);

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
    let pipeline =
        spawn_blocking(|| Pipeline::from_pretrained("amcoff/bert-based-swedish-cased-ner"))
            .await??;

    Ok(pipeline)
}

async fn handle_msg(
    (sentence, tx): Message,
    handles: &mut Handles,
    pipeline: &mut Option<Arc<Pipeline>>,
) {
    if pipeline.is_none() {
        debug!("initializing pipeline");

        match get_pipeline().await {
            Ok(p) => *pipeline = Some(Arc::new(p)),
            Err(e) => {
                let _ = tx.send(Err(e));
                return;
            }
        }

        debug!("initialized pipeline");
    }

    let pipeline = Arc::clone(pipeline.as_ref().unwrap());

    let handle = tokio::spawn(async move {
        let res = tokio_rayon::spawn(move || pipeline.predict(sentence)).await;
        let _ = tx.send(res.map_err(Into::into));
    });

    handles.push(handle);
}

async fn wait(handles: &mut Handles) {
    while handles.next().await.is_some() {}
    const PIPELINE_TTL: Duration = Duration::from_secs(10);
    sleep(PIPELINE_TTL).await;
}

fn act() -> mpsc::Sender<Message> {
    let (tx, mut rx) = mpsc::channel::<Message>(16);

    tokio::spawn(async move {
        let mut pipeline = None;
        let mut handles = FuturesUnordered::new();

        loop {
            select! {
                Some(msg) = rx.recv() => {
                    debug!("received message");
                    handle_msg(msg, &mut handles, &mut pipeline).await;
                }
                _ = wait(&mut handles) => if pipeline.take().is_some() {
                    debug!("dropped pipeline");
                }
            }
        }
    });

    tx
}

#[tokio::main]
async fn main() {
    let _ = dotenv::dotenv();
    tracing_subscriber::fmt::init();

    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<TrastServer<TrastService>>()
        .await;

    let addr = "[::1]:8000".parse().unwrap();

    let trast = TrastService { actor_tx: act() };

    info!("listening on {addr}");

    Server::builder()
        .add_service(health_service)
        .add_service(TrastServer::new(trast))
        .serve(addr)
        .await
        .unwrap();
}
