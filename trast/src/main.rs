use std::{sync::Arc, time::Duration};

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::post, Json, Router};
use futures::{future::try_join_all, stream::FuturesUnordered, StreamExt, TryStreamExt};
use onnx_bert::{Entity, Pipeline};
use tokio::{
    select,
    sync::{mpsc, oneshot},
    task::{spawn_blocking, JoinError, JoinHandle},
    time::sleep,
};

type Message = (Vec<String>, oneshot::Sender<Result<Vec<Vec<Entity>>>>);

type Result<T, E = Error> = core::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("{0}")]
    Join(#[from] JoinError),
    #[error("{0}")]
    Bert(#[from] onnx_bert::Error),
}

impl IntoResponse for Error {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, "internal server error").into_response()
    }
}

type Handles = FuturesUnordered<JoinHandle<()>>;

async fn handle_msg(
    (sentences, tx): Message,
    handles: &mut Handles,
    pipeline: &mut Option<Arc<Pipeline>>,
) {
    if pipeline.is_none() {
        match spawn_blocking(|| {
            Pipeline::from_pretrained("amcoff/bert-based-swedish-cased-ner").unwrap()
        })
        .await
        {
            Ok(p) => *pipeline = Some(Arc::new(p)),
            Err(e) => {
                let _ = tx.send(Err(e.into()));
                return;
            }
        }
    }

    let pipeline = Arc::clone(pipeline.as_ref().unwrap());

    let handle = tokio::spawn(async move {
        let res = try_join_all(sentences.into_iter().map(move |s| {
            let pipeline = Arc::clone(&pipeline);
            tokio_rayon::spawn(move || pipeline.predict(s))
        }))
        .await;

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
    let (tx, mut rx) = mpsc::channel::<Message>(10);

    tokio::spawn(async move {
        let mut pipeline = None;

        let mut handles = FuturesUnordered::new();

        loop {
            select! {
                Some(msg) = rx.recv() => { handle_msg(msg, &mut handles, &mut pipeline).await; }
                _ = wait(&mut handles) => if pipeline.take().is_some() {
                    eprintln!("dropped pipeline");
                }
            }
        }
    });

    tx
}

#[derive(Debug, Clone)]
struct AppState {
    actor_tx: mpsc::Sender<Message>,
}

async fn predict(
    State(s): State<AppState>,
    Json(sentences): Json<Vec<String>>,
) -> Result<impl IntoResponse> {
    let (tx, rx) = oneshot::channel();
    s.actor_tx.send((sentences, tx)).await.unwrap();
    let res = rx.await.unwrap()?;

    Ok(Json(res))
}

#[tokio::main]
async fn main() {
    let actor_tx = act();
    let app = Router::new()
        .route("/", post(predict))
        .with_state(AppState { actor_tx });

    eprintln!("binding");

    axum::Server::bind(&"0.0.0.0:8000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
