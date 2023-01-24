use std::time::Instant;

use onnxruntime::{
    environment::Environment,
    ndarray,
    tensor,
    GraphOptimizationLevel,
};
use tokenizers::{
    models::wordpiece::WordPieceBuilder, normalizers::BertNormalizer,
    pre_tokenizers::bert::BertPreTokenizer, processors::bert::BertProcessing, AddedToken,
    EncodeInput, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
};

fn main() -> anyhow::Result<()> {
    let start = Instant::now();

    let wp = WordPieceBuilder::new()
        .files("vocab.txt".to_owned())
        .build()
        .unwrap();

    // let tokenizer = Tokenizer::from_pretrained("KB/bert-base-swedish-cased-ner", None).unwrap();
    // let tokenizer = Tokenizer::from_file(file)

    let mut tokenizer = Tokenizer::new(wp);
    tokenizer.with_pre_tokenizer(BertPreTokenizer);
    tokenizer.with_truncation(Some(TruncationParams::default()));
    tokenizer.with_post_processor(BertProcessing::new(
        ("[SEP]".to_owned(), 102),
        ("[CLS]".to_owned(), 101),
    ));
    tokenizer.with_normalizer(BertNormalizer::new(true, true, None, false));
    tokenizer.add_special_tokens(&[
        AddedToken::from("[PAD]", true),
        AddedToken::from("[CLS]", true),
        AddedToken::from("[SEP]", true),
        AddedToken::from("[MASK]", true),
    ]);
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(512),
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".to_owned(),
    }));

    let environment = Environment::builder().with_name("test").build()?;
    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_model_from_file("onnx/model.onnx")?;

    dbg!(&session.inputs);
    let setup = Instant::now();

    let read = Instant::now();

    let input = tokenizer
        .encode(
            EncodeInput::Single("Idag släpper KB tre nya språkmodeller.".into()),
            true,
        )
        .unwrap();
    let id_dim = input.get_ids().len();
    let ids: Vec<i64> = input.get_ids().iter().map(|x| (*x).into()).collect();
    let ids = ndarray::Array::from_vec(ids)
        .into_shape((1, id_dim))
        .unwrap();

    let mask_dim = input.get_attention_mask().len();
    let mask: Vec<i64> = input
        .get_attention_mask()
        .iter()
        .map(|x| (*x).into())
        .collect();
    let mask = ndarray::Array::from_vec(mask)
        .into_shape((1, mask_dim))
        .unwrap();

    let type_id_dim = input.get_type_ids().len();
    let type_ids: Vec<i64> = input.get_type_ids().iter().map(|x| (*x).into()).collect();
    let type_ids = ndarray::Array::from_vec(type_ids)
        .into_shape((1, type_id_dim))
        .unwrap();

    let encode = Instant::now();

    dbg!(&session.outputs);

    let outputs: Vec<tensor::OrtOwnedTensor<f32, _>> =
        session.run(vec![ids, mask, type_ids]).unwrap();
    let output = &outputs[0]; // last hidden state

    dbg!(outputs.len());

    let start_logits = output.first().unwrap();
    dbg!(start_logits);

    // let mut masks: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>> =
    //     Vec::new();

    // let mut tokens: Vec<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>>> =
    //     Vec::new();
    // let mut mask = ndarray::Array::zeros((256, 60));
    // let mut token = ndarray::Array::zeros((256, 60));
    // for (i, input) in input_ids.iter().rev().enumerate() {
    //     for (j, attention) in input.get_attention_mask().iter().enumerate() {
    //         mask[[255 - i % 256, j]] = *attention as i64;
    //     }
    //     for (j, attention) in input.get_ids().iter().enumerate() {
    //         token[[255 - i % 256, j]] = *attention as i64;
    //     }
    //     if (i + 1) % 256 == 0 || i == input_ids.len() - 1 {
    //         masks.push(mask);
    //         mask = ndarray::Array::zeros((256, 60));
    //         tokens.push(token);
    //         token = ndarray::Array::zeros((256, 60));
    //     }
    // }

    // for _ in 0..masks.len() {
    //     let clone = session.clone();
    //     let mut clone = clone.lock().unwrap();
    //     let result: Vec<tensor::OrtOwnedTensor<f32, ndarray::Dim<ndarray::IxDynImpl>>> =
    //         clone.run(vec![tokens.pop().unwrap(), masks.pop().unwrap()])?;
    //     result.iter().for_each(|array| {
    //         for row in array.outer_iter() {
    //             let row = row.as_slice().unwrap();
    //             // writer.serialize(row).unwrap();
    //             dbg!(row);
    //         }
    //     });
    // }

    let write_onnx = Instant::now();
    eprintln!("Setup: {}ms", (setup - start).as_millis());

    eprintln!("Read: {}ms", (read - setup).as_millis());

    eprintln!("Encode: {}ms", (encode - read).as_millis());
    eprintln!("Write Onnx: {}ms", (write_onnx - encode).as_millis());
    //   println!(
    //       "outputs: {:#?}",
    //       outputs
    //           .pop()
    //           .unwrap()
    //           .map_axis(ndarray::Axis(1), |x| x[0] > x[1])
    //           .map(|x| match x {
    //               True => "Open",
    //               False => "Not Open",
    //           })
    //   );
    //   println!("outputs: {:#?}\n", &outputs);
    // find and display the max value with its index

    Ok(())
}
