use std::time::Instant;

use trast::Pipeline;

fn main() -> Result<(), trast::Error> {
    let pipeline = Pipeline::from_pretrained("amcoff/bert-based-swedish-cased-ner")?;

    let start = Instant::now();
    let output = pipeline.predict("Idag släpper Kungliga biblioteket tre nya språkmodeller.")?;
    let duration = start.elapsed();

    println!("{output:?}");
    eprintln!("inference took {duration:?}");

    Ok(())
}
