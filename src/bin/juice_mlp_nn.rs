#[macro_use]
extern crate log;
extern crate coaster;
extern crate coaster_nn;
extern crate csv;
extern crate humantime;
extern crate image;
extern crate math;
extern crate rusty_machine;
extern crate serde_derive;
extern crate structopt;

use coaster::prelude::*;
use flate2::read::GzDecoder;
use humantime::format_duration;
use juice::layer;
use juice::layers;
use juice::solver;
use juice::util;
use log::Level;
use std::fs;
use std::io;
use std::path;
use std::rc;
use std::str::FromStr;
use std::sync;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "mlp_nn_juice", about = "multi layer perceptron neural network using juice")]
struct Options {
    #[structopt(short = "s", long = "serialization_dir", long_help = "serialization data directory", required = true, parse(from_os_str))]
    serialization_dir: path::PathBuf,

    #[structopt(short = "w", long = "width", long_help = "width", default_value = "315")]
    width: usize,

    #[structopt(short = "h", long = "height", long_help = "height", default_value = "390")]
    height: usize,

    #[structopt(short = "b", long = "batch_size", long_help = "batch size", default_value = "2")]
    batch_size: usize,

    #[structopt(short = "r", long = "learning_rate", long_help = "learning rate", default_value = "0.01")]
    learning_rate: f32,

    #[structopt(short = "o", long = "momentum", long_help = "momentum", default_value = "0")]
    momentum: f32,

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
    debug!("training_data.len(): {}", training_data.len());

    // deserializing the training labels
    let mut training_labels_path = options.serialization_dir.clone();
    training_labels_path.push(format!("herbarium-training-labels-{}x{}.ser.gz", options.width, options.height));

    let training_labels_reader = io::BufReader::new(fs::File::open(training_labels_path).unwrap());
    let mut training_labels_decoder = GzDecoder::new(training_labels_reader);
    let training_labels: Vec<f32> = bincode::deserialize_from(&mut training_labels_decoder).unwrap();
    debug!("training_labels.len(): {}", training_labels.len());

    let mut associated_data = Vec::new();
    for (i, row_data) in training_data.iter().enumerate() {
        associated_data.push((*training_labels.get(i).unwrap(), row_data));
    }
    let mut tmp_labels = training_labels.clone();
    tmp_labels.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    tmp_labels.dedup();
    let unique_labels_count = tmp_labels.len();

    let features_count = training_data.first().unwrap().len();
    debug!("features_count: {}, unique_label_count: {}", features_count, unique_labels_count);

    let all_same_size = itertools::all(&training_data, |e| e.len() == features_count);
    debug!("all_same_size: {}", all_same_size);

    let mut net_cfg = layers::SequentialConfig::default();
    net_cfg.add_input("data", &[options.batch_size, options.width, options.height]);
    net_cfg.force_backward = true;

    let reshape_layer_type = layer::LayerType::Reshape(layers::ReshapeConfig::of_shape(&[options.batch_size, features_count]));
    net_cfg.add_layer(layer::LayerConfig::new("reshape", reshape_layer_type));

    let linear1_layer_type = layer::LayerType::Linear(layers::LinearConfig { output_size: features_count * 2 });
    net_cfg.add_layer(layer::LayerConfig::new("linear1", linear1_layer_type));

    // net_cfg.add_layer(layer::LayerConfig::new("relu", layer::LayerType::ReLU));
    net_cfg.add_layer(layer::LayerConfig::new("sigmoid", layer::LayerType::Sigmoid));

    let linear2_layer_type = layer::LayerType::Linear(layers::LinearConfig { output_size: features_count });
    net_cfg.add_layer(layer::LayerConfig::new("linear2", linear2_layer_type));

    let linear3_layer_type = layer::LayerType::Linear(layers::LinearConfig { output_size: features_count / 2 });
    net_cfg.add_layer(layer::LayerConfig::new("linear3", linear3_layer_type));

    // net_cfg.add_layer(layer::LayerConfig::new("sigmoid", layer::LayerType::Sigmoid));

    let linear4_layer_type = layer::LayerType::Linear(layers::LinearConfig { output_size: unique_labels_count });
    net_cfg.add_layer(layer::LayerConfig::new("linear4", linear4_layer_type));

    net_cfg.add_layer(layer::LayerConfig::new("log_softmax", layer::LayerType::LogSoftmax));

    let mut classifier_cfg = layers::SequentialConfig::default();
    classifier_cfg.add_input("network_out", &[options.batch_size, unique_labels_count]);
    classifier_cfg.add_input("label", &[options.batch_size, 1]);

    let nll_layer_type = layer::LayerType::NegativeLogLikelihood(layers::NegativeLogLikelihoodConfig { num_classes: unique_labels_count });
    classifier_cfg.add_layer(layer::LayerConfig::new("nll", nll_layer_type));

    let backend = rc::Rc::new(Backend::<Cuda>::default().unwrap());
    // let native_backend = ::std::rc::Rc::new(Backend::<Native>::default().unwrap());

    // set up solver
    let mut solver_cfg = solver::SolverConfig {
        minibatch_size: options.batch_size,
        base_lr: options.learning_rate,
        momentum: options.momentum,
        ..solver::SolverConfig::default()
    };
    solver_cfg.network = layer::LayerConfig::new("network", net_cfg);
    solver_cfg.objective = layer::LayerConfig::new("classifier", classifier_cfg);

    let mut solver = solver::Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);

    let inp = SharedTensor::<f32>::new(&[options.batch_size, options.width, options.height]);
    let label = SharedTensor::<f32>::new(&[options.batch_size, 1]);

    let inp_lock = sync::Arc::new(sync::RwLock::new(inp));
    let label_lock = sync::Arc::new(sync::RwLock::new(label));

    // set up confusion matrix
    let mut confusion = solver::ConfusionMatrix::new(unique_labels_count);
    confusion.set_capacity(Some(1000));

    for data in associated_data.chunks(options.batch_size) {
        let mut targets = Vec::new();
        for (idx, d) in data.iter().enumerate() {
            let mut inp = inp_lock.write().unwrap();
            let mut label = label_lock.write().unwrap();

            util::write_batch_sample(&mut inp, &d.1, idx);
            util::write_batch_sample(&mut label, &[d.0], idx);

            targets.push(d.0 as usize);
        }
        // train the network!
        let infered_out = solver.train_minibatch(inp_lock.clone(), label_lock.clone());

        let mut infered = infered_out.write().unwrap();
        let predictions = confusion.get_predictions(&mut infered);

        confusion.add_samples(&predictions, &targets);

        println!("Accuracy {}", confusion.accuracy());
    }
    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
