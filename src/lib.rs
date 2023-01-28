use std::{collections::HashMap, fmt::Debug, fs::File, io::BufReader, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::{EncodeInput, Tokenizer};
use tract_onnx::{
    prelude::{tvec, Framework, Graph, InferenceModelExt, SimplePlan, Tensor, TypedFact, TypedOp},
    tract_hir::tract_ndarray::{Array2, ShapeError},
};

#[cfg(feature = "remote")]
mod remote;

#[derive(Debug, Serialize, Deserialize)]
pub struct Entity {
    pub label: String,
    pub score: f32,
    pub word: String,
    pub start: usize,
    pub end: usize,
}

pub struct Pipeline {
    tokenizer: Tokenizer,
    config: Config,
    model: Model,
}

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Debug, Deserialize)]
struct Config {
    id2label: HashMap<i64, String>,
}

#[derive(Debug)]
struct RawEntity {
    label: i64,
    score: f32,
    start: usize,
    end: usize,
}

impl Pipeline {
    pub fn from_files(
        config: impl AsRef<Path>,
        tokenizer: impl AsRef<Path>,
        model: impl AsRef<Path>,
    ) -> Result<Self> {
        let config: Config = serde_json::from_reader(BufReader::new(File::open(config)?))?;
        let tokenizer = Tokenizer::from_file(tokenizer)?;
        let model = tract_onnx::onnx()
            .model_for_path(model)?
            .into_optimized()?
            .into_runnable()?;

        Ok(Self {
            tokenizer,
            config,
            model,
        })
    }

    #[cfg(feature = "remote")]
    pub fn from_pretrained(model: impl AsRef<str>) -> Result<Self> {
        let model = model.as_ref();
        let download_file = |file: &str| {
            remote::download(format!(
                "https://huggingface.co/{model}/resolve/main/{file}"
            ))
        };

        Self::from_files(
            download_file("config.json")?,
            download_file("tokenizer.json")?,
            download_file("model.onnx")?,
        )
    }

    pub fn predict(&self, sentence: impl AsRef<str>) -> Result<Vec<Entity>> {
        let sentence = sentence.as_ref();
        let input = self
            .tokenizer
            .encode(EncodeInput::Single(sentence.into()), true)?;

        let input_ids: Tensor = Array2::from_shape_vec(
            (1, input.len()),
            input.get_ids().iter().map(|&x| x as i64).collect(),
        )?
        .into();
        let attention_mask: Tensor = Array2::from_shape_vec(
            (1, input.len()),
            input
                .get_attention_mask()
                .iter()
                .map(|&x| x as i64)
                .collect(),
        )?
        .into();
        let token_type_ids: Tensor = Array2::from_shape_vec(
            (1, input.len()),
            input.get_type_ids().iter().map(|&x| x as i64).collect(),
        )?
        .into();

        let outputs = self.model.run(tvec![
            input_ids.into(),
            attention_mask.into(),
            token_type_ids.into()
        ])?;

        let mut entities: Vec<RawEntity> = vec![];

        let logits = outputs[0].to_array_view::<f32>()?;

        for (i, scores) in logits.rows().into_iter().enumerate() {
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

            let score = max / sum;
            let (start, end) = input.get_offsets()[i];

            match entities.last_mut() {
                Some(prev) if prev.label == label => {
                    prev.score = prev.score.max(score);
                    prev.start = prev.start.min(start);
                    prev.end = prev.end.max(end);
                }
                _ => entities.push(RawEntity {
                    label,
                    score,
                    start,
                    end,
                }),
            }
        }

        let entities = entities
            .into_iter()
            .filter(|e| e.label != 0 && e.end > e.start)
            .map(
                |RawEntity {
                     label,
                     score,
                     start,
                     end,
                 }| Entity {
                    label: self.config.id2label[&label].clone(),
                    score,
                    word: sentence[start..end].to_owned(),
                    start,
                    end,
                },
            )
            .collect::<Vec<Entity>>();

        Ok(entities)
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[cfg(feature = "remote")]
    #[cfg_attr(feature = "remote", error("{0}"))]
    Download(#[from] cached_path::Error),
    #[error("{0}")]
    Serde(#[from] serde_json::Error),
    #[error("{0}")]
    Onnx(#[from] tract_onnx::tract_core::anyhow::Error),
    #[error("tokenizer error")]
    Tokenizer,
    #[error("shape error: {0}")]
    Shape(#[from] ShapeError),
}

impl From<Box<dyn std::error::Error + Send + Sync>> for Error {
    fn from(_: Box<dyn std::error::Error + Send + Sync>) -> Self {
        Self::Tokenizer
    }
}

pub type Result<T, E = Error> = core::result::Result<T, E>;
