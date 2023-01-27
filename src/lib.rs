use std::{collections::HashMap, fs::File, io::BufReader, path::Path};

use onnxruntime::{
    environment::Environment, ndarray, session::Session, tensor, GraphOptimizationLevel, OrtError,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::{EncodeInput, Tokenizer};

pub use onnxruntime;

#[cfg(feature = "download")]
mod download;

#[derive(Debug, Serialize, Deserialize)]
pub struct Entity {
    label: String,
    score: f32,
    word: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Deserialize)]
struct Config {
    id2label: HashMap<i64, String>,
}

pub struct Pipeline<'a> {
    tokenizer: Tokenizer,
    config: Config,
    session: Session<'a>,
}

impl<'a> Pipeline<'a> {
    pub fn from_files(
        env: &'a Environment,
        config: impl AsRef<Path>,
        tokenizer: impl AsRef<Path>,
        model: impl AsRef<Path> + 'a,
    ) -> Result<Self> {
        let config: Config = serde_json::from_reader(BufReader::new(File::open(config)?))?;
        let tokenizer = Tokenizer::from_file(tokenizer)?;
        let session = env
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_model_from_file(model)?;

        Ok(Self {
            tokenizer,
            config,
            session,
        })
    }

    #[cfg(feature = "download")]
    pub fn from_pretrained(env: &'a Environment, model: impl AsRef<str>) -> Result<Self> {
        let model = model.as_ref();
        let download_file = |file: &str| {
            download::download(format!(
                "https://huggingface.co/{model}/resolve/main/{file}"
            ))
        };

        Self::from_files(
            env,
            download_file("config.json")?,
            download_file("tokenizer.json")?,
            download_file("model.onnx")?,
        )
    }

    pub fn predict(&mut self, sentence: impl AsRef<str>) -> Result<Vec<Entity>> {
        let input = self
            .tokenizer
            .encode(EncodeInput::Single(sentence.as_ref().into()), true)?;

        let ids: Vec<i64> = input.get_ids().iter().map(|x| (*x).into()).collect();
        let ids = ndarray::Array::from_vec(ids)
            .into_shape((1, input.len()))
            .unwrap();

        let attention_mask: Vec<i64> = input
            .get_attention_mask()
            .iter()
            .map(|x| (*x).into())
            .collect();
        let attention_mask = ndarray::Array::from_vec(attention_mask)
            .into_shape((1, input.len()))
            .unwrap();

        let type_ids: Vec<i64> = input.get_type_ids().iter().map(|x| (*x).into()).collect();
        let type_ids = ndarray::Array::from_vec(type_ids)
            .into_shape((1, input.len()))
            .unwrap();

        let outputs: Vec<tensor::OrtOwnedTensor<f32, _>> =
            self.session.run(vec![ids, attention_mask, type_ids])?;

        let entities = outputs[0]
            .rows()
            .into_iter()
            .enumerate()
            .filter_map(|(i, scores)| {
                let mut sum = 0.;
                let mut max = f32::MIN;
                let mut label = 0;

                for (i, z) in scores.iter().enumerate() {
                    let z = z.exp();
                    sum += z;
                    if z > max {
                        max = z;
                        label = i as _;
                    }
                }

                if label == 0 {
                    None
                } else {
                    let (start, end) = input.get_offsets()[i];

                    Some(Entity {
                        label: self.config.id2label[&label].clone(),
                        score: max / sum,
                        word: input.get_tokens()[i].clone(),
                        start,
                        end,
                    })
                }
            })
            .collect::<Vec<Entity>>();

        Ok(entities)
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[cfg(feature = "download")]
    #[cfg_attr(feature = "download", error("{0}"))]
    Download(#[from] cached_path::Error),
    #[error("{0}")]
    Serde(#[from] serde_json::Error),
    #[error("{0}")]
    Onnx(#[from] OrtError),
    #[error("tokenizer error")]
    Tokenizer,
}

impl From<Box<dyn std::error::Error + Send + Sync>> for Error {
    fn from(_: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::Tokenizer
    }
}

pub type Result<T, E = Error> = core::result::Result<T, E>;
