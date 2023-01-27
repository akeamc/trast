use std::time::Instant;

use onnxruntime::environment::Environment;
use trast::Pipeline;

fn main() -> Result<(), trast::Error> {
    let env = Environment::builder().build()?;
    let mut pipeline = Pipeline::from_pretrained(&env, "amcoff/bert-based-swedish-cased-ner")?;

    let start = Instant::now();
    let output = pipeline.predict("Idag släpper KB tre nya språkmodeller.")?;
    let duration = start.elapsed();

    println!("{output:?}");
    eprintln!("inference took {duration:?}");

    Ok(())
}
