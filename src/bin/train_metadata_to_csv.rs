#[macro_use]
extern crate log;
extern crate csv;
extern crate humantime;
extern crate image;
extern crate math;
extern crate rusty_machine;
extern crate serde_derive;
extern crate structopt;

use humantime::format_duration;
use log::Level;
use std::fs;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "read_train_metadata", about = "read train metadata")]
struct Options {
    #[structopt(short = "i", long = "base_dir", long_help = "input dir", required = true, parse(from_os_str))]
    base_dir: path::PathBuf,

    #[structopt(short = "o", long = "output", long_help = "output", required = true, parse(from_os_str))]
    output: path::PathBuf,

    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "Info")]
    log_level: String,
}

fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    let mut training_dir = options.base_dir.clone();
    training_dir.push("train");

    let mut train_metadata_path = training_dir.clone();
    train_metadata_path.push("metadata.json");

    let train_metadata_file = fs::File::open(train_metadata_path.as_path()).expect("missing file");
    let br = io::BufReader::new(train_metadata_file);

    let metadata: rusty_herbarium::TrainMetadata = serde_json::from_reader(br)?;
    //info!("metadata.info.year: {}", metadata.info.year);

    let output_options = fs::OpenOptions::new()
        .write(true)
        .append(false)
        .create(true)
        .open(&options.output.as_path())
        .unwrap();

    let bw = io::BufWriter::new(output_options);

    let mut writer = csv::Writer::from_writer(bw);
    writer.write_record(&["annotation_id", "category_id", "family", "genus", "region_id", "name", "image_id", "width", "height", "file_name"])?;
    for annotation in metadata.annotations.iter() {
        let mut row = Vec::new();
        row.push(annotation.id.to_string());

        let category = metadata
            .categories
            .iter()
            .filter(|e| e.id == annotation.category_id)
            .take(1)
            .nth(0)
            .unwrap();
        row.push(category.id.to_string());
        row.push(category.family.to_string());
        row.push(category.genus.to_string());

        let region = metadata.regions.iter().filter(|e| e.id == annotation.region_id).take(1).nth(0).unwrap();
        row.push(region.id.to_string());
        row.push(region.name.to_string());

        let image = metadata.images.iter().filter(|e| e.id == annotation.image_id).take(1).nth(0).unwrap();
        // let mut image_path = training_dir.clone();
        // image_path.push(image.file_name.clone());

        row.push(image.id.to_string());
        // row.push(image_path.to_string_lossy().to_string());
        row.push(image.width.to_string());
        row.push(image.height.to_string());
        row.push(image.file_name.to_string());

        writer.write_record(&row)?;

        writer.flush()?;
    }

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
