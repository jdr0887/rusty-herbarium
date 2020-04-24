#[macro_use]
extern crate log;
extern crate csv;
extern crate humantime;
extern crate image;
extern crate math;
extern crate rusty_machine;
extern crate serde_derive;
extern crate structopt;

use bincode;
use flate2::read::GzDecoder;
use humantime::format_duration;
use log::Level;
use rustlearn::array;
use rustlearn::prelude::*;
use std::fs;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "logistic_regression_rustlearn", about = "logistic regression using rustlearn")]
struct Options {
    #[structopt(short = "w", long = "width", long_help = "width", default_value = "315")]
    width: u32,

    #[structopt(short = "h", long = "height", long_help = "width", default_value = "390")]
    height: u32,

    #[structopt(short = "s", long = "serialization_dir", long_help = "serialization directory", required = true, parse(from_os_str))]
    serialization_dir: path::PathBuf,

    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "debug")]
    log_level: String,
}

fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    // deserializing the training data
    let mut training_data_path = options.serialization_dir.clone();
    training_data_path.push(format!("herbarium-training-data-{}x{}.ser.gz", options.width, options.height));

    let training_data_reader = io::BufReader::new(fs::File::open(training_data_path).unwrap());
    let mut training_data_decoder = GzDecoder::new(training_data_reader);
    let training_data: array::sparse::SparseRowArray = bincode::deserialize_from(&mut training_data_decoder).unwrap();
    debug!("training_data.rows(): {}, training_data.cols(): {}", training_data.rows(), training_data.cols());

    // deserializing the training labels
    let mut training_labels_path = options.serialization_dir.clone();
    training_labels_path.push(format!("herbarium-training-labels-{}x{}.ser.gz", options.width, options.height));

    let training_labels_reader = io::BufReader::new(fs::File::open(training_labels_path).unwrap());
    let mut training_labels_decoder = GzDecoder::new(training_labels_reader);
    let training_labels: Vec<f32> = bincode::deserialize_from(&mut training_labels_decoder).unwrap();
    debug!("training_labels.rows(): {}", training_data.rows());

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
