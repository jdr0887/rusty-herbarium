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
use rustlearn::ensemble::random_forest;
use rustlearn::metrics;
use rustlearn::prelude::*;
use rustlearn::trees::decision_tree;
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
    let training_data: Vec<Vec<f32>> = bincode::deserialize_from(&mut training_data_decoder).unwrap();

    let features_count = training_data.first().unwrap().len();
    debug!("features_count: {}", features_count);

    let all_same_size = itertools::all(&training_data, |e| e.len() == features_count);
    debug!("all_same_size: {}", all_same_size);

    let mut sparse_array_training_data = array::sparse::SparseRowArray::zeros(training_data.len(), features_count);
    for (i, row) in training_data.into_iter().enumerate() {
        for (j, col) in row.into_iter().enumerate() {
            if col > 0.04 {
                sparse_array_training_data.set(i, j, col);
            }
        }
    }
    debug!("sparse_array_training_data.rows(): {}", sparse_array_training_data.rows());

    // deserializing the training labels
    let mut training_labels_path = options.serialization_dir.clone();
    training_labels_path.push(format!("herbarium-training-labels-{}x{}.ser.gz", options.width, options.height));

    let training_labels_reader = io::BufReader::new(fs::File::open(training_labels_path).unwrap());
    let mut training_labels_decoder = GzDecoder::new(training_labels_reader);
    let training_labels: Vec<f32> = bincode::deserialize_from(&mut training_labels_decoder).unwrap();
    let dense_array_training_labels = array::dense::Array::from(training_labels);
    debug!("dense_array_training_labels.rows(): {}", dense_array_training_labels.rows());

    // deserializing the validation data
    let mut validation_data_path = options.serialization_dir.clone();
    validation_data_path.push(format!("herbarium-validation-data-{}x{}.ser.gz", options.width, options.height));

    let validation_data_reader = io::BufReader::new(fs::File::open(validation_data_path).unwrap());
    let mut validation_data_decoder = GzDecoder::new(validation_data_reader);
    let validation_data: Vec<Vec<f32>> = bincode::deserialize_from(&mut validation_data_decoder).unwrap();

    let mut sparse_array_validation_data = array::sparse::SparseRowArray::zeros(validation_data.len(), validation_data.first().unwrap().len());
    for (i, row) in validation_data.into_iter().enumerate() {
        for (j, col) in row.into_iter().enumerate() {
            if col > 0.04 {
                sparse_array_validation_data.set(i, j, col);
            }
        }
    }
    debug!("sparse_array_validation_data.rows(): {}", sparse_array_validation_data.rows());

    // deserializing the validation labels
    let mut validation_labels_path = options.serialization_dir.clone();
    validation_labels_path.push(format!("herbarium-validation-labels-{}x{}.ser.gz", options.width, options.height));

    let validation_labels_reader = io::BufReader::new(fs::File::open(validation_labels_path).unwrap());
    let mut validation_labels_decoder = GzDecoder::new(validation_labels_reader);
    let validation_labels: Vec<f32> = bincode::deserialize_from(&mut validation_labels_decoder).unwrap();
    let dense_array_validation_labels = array::dense::Array::from(validation_labels);
    debug!("dense_array_validation_labels.rows(): {}", dense_array_validation_labels.rows());

    let mut tree_params = decision_tree::Hyperparameters::new(features_count);
    tree_params.min_samples_split(10);
    tree_params.max_depth(10);
    tree_params.max_features(10);

    let mut model = random_forest::Hyperparameters::new(tree_params, 10).one_vs_rest();

    let start_fitting = Instant::now();
    model.fit(&sparse_array_training_data, &dense_array_training_labels).unwrap();
    debug!("model fitting duration: {}", format_duration(start_fitting.elapsed()).to_string());

    let prediction_output = model.predict(&sparse_array_validation_data).unwrap();

    debug!("validation_labels: {:?}", dense_array_validation_labels);
    debug!("prediction_output: {:?}", prediction_output);

    info!("accuracy: {}", metrics::accuracy_score(&dense_array_validation_labels, &prediction_output));

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
