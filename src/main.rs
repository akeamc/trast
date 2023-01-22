use std::time::Duration;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use rust_bert::{
    pipelines::{
        common::ModelType,
        ner::{Entity, NERModel},
        token_classification::{LabelAggregationOption, TokenClassificationConfig},
    },
    resources::RemoteResource,
    RustBertError,
};
use tokio::{
    select,
    sync::{
        mpsc,
        oneshot::{self, error::RecvError},
    },
    task::JoinError,
    time::sleep,
};

type Message = (Vec<String>, oneshot::Sender<Result<Vec<Vec<Entity>>>>);

#[derive(Clone)]
struct AppState {
    tx: mpsc::Sender<Message>,
}

impl AppState {
    async fn predict(&self, input: Vec<String>) -> Result<Vec<Vec<Entity>>> {
        let (tx, rx) = oneshot::channel();
        self.tx.send((input, tx)).await.unwrap();
        rx.await?
    }
}

async fn predict(
    State(state): State<AppState>,
    Json(input): Json<Vec<String>>,
) -> Result<impl IntoResponse> {
    Ok(Json(state.predict(input).await?))
}

async fn health() -> impl IntoResponse {
    ([("cache-control", "no-cache")], env!("CARGO_PKG_VERSION"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let (tx, rx) = mpsc::channel(1);

    tokio::spawn(act(rx));

    let state = AppState { tx };

    let app = Router::new()
        .route("/health", get(health))
        .route("/predict", post(predict))
        .with_state(state);

    // run it with hyper on localhost:8000
    axum::Server::bind(&"0.0.0.0:8000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();

    Ok(())
}

async fn handle_msg(input: Vec<String>, model: &mut Option<NERModel>) -> Result<Vec<Vec<Entity>>> {
    if model.is_none() {
        let config = TokenClassificationConfig::new(
            ModelType::Bert,
            RemoteResource::from_pretrained((
                "amcoff/bert-based-swedish-cased-ner",
                "https://huggingface.co/amcoff/bert-based-swedish-cased-ner/resolve/main/rust_model.ot",
            )),
            RemoteResource::from_pretrained((
                "amcoff/bert-based-swedish-cased-ner",
                "https://huggingface.co/KB/bert-base-swedish-cased-ner/resolve/main/config.json",
            )),
            RemoteResource::from_pretrained((
                "amcoff/bert-based-swedish-cased-ner",
                "https://huggingface.co/KB/bert-base-swedish-cased-ner/resolve/main/vocab.txt",
            )),
            None,
            false,
            None,
            None,
            LabelAggregationOption::First,
        );

        *model = Some(tokio::task::spawn_blocking(|| NERModel::new(config)).await??);
    }

    Ok(model.as_ref().unwrap().predict(&input))
}

async fn act(mut rx: mpsc::Receiver<Message>) {
    let mut model = None;
    let ttl = Duration::from_secs(10);

    loop {
        select! {
            Some((input, tx)) = rx.recv() => {
                let _ = tx.send(handle_msg(input, &mut model).await);
            }
            _ = sleep(ttl) => model = None // save memory
        }
    }
}

#[derive(Debug)]
enum Error {
    Bert(RustBertError),
    Bruh,
}

impl From<RustBertError> for Error {
    fn from(e: RustBertError) -> Self {
        Self::Bert(e)
    }
}

impl From<JoinError> for Error {
    fn from(_: JoinError) -> Self {
        Self::Bruh
    }
}

impl From<RecvError> for Error {
    fn from(_: RecvError) -> Self {
        Self::Bruh
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> axum::response::Response {
        let msg = match self {
            Error::Bert(_) => "failed to build model",
            Error::Bruh => "internal server error",
        };

        (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response()
    }
}

type Result<T, E = Error> = core::result::Result<T, E>;
