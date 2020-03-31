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
use rustlearn::array;
use rustlearn::linear_models::sgdclassifier;
use rustlearn::metrics;
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
    #[structopt(short = "b", long = "base_dir", long_help = "base directory for ", required = true, parse(from_os_str))]
    base_dir: path::PathBuf,

    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "debug")]
    log_level: String,
}

fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    let mut train_dir = options.base_dir.clone();
    train_dir.push("train");

    let mut train_metadata_path = train_dir.clone();
    train_metadata_path.push("metadata.json");

    let train_metadata_file = fs::File::open(train_metadata_path.as_path()).expect("missing file");
    let br = io::BufReader::new(train_metadata_file);

    let metadata: rusty_herbarium::TrainMetadata = serde_json::from_reader(br)?;

    // let size = (520u32, 660u32);
    //let size = (315u32, 400u32); // killed at 35k
    //let size = (284u32, 360u32); // killed at 70k
    //let size = (236u32, 300u32);
    let size = (210u32, 264u32);

    let (train_data, train_labels) = rusty_herbarium::normalized_train_data(&train_dir, &metadata, size.0, size.1)?;

    let col_size = ((size.0 * size.1) * 3) as usize;

    let mut model = sgdclassifier::Hyperparameters::new(col_size)
        .learning_rate(0.1)
        .l1_penalty(0.0)
        .l2_penalty(0.0)
        .one_vs_rest();

    model.fit(&train_data, &train_labels).unwrap();

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
