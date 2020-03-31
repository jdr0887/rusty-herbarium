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
use image::GenericImageView;
use itertools::Itertools;
use log::Level;
use rusty_machine::analysis::score::accuracy;
use rusty_machine::data::transforms::{Standardizer, Transformer};
use rusty_machine::learning;
use rusty_machine::learning::optim::grad_desc::GradientDesc;
use rusty_machine::linalg;
use rusty_machine::prelude::SupModel;
use rusty_machine::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections;
use std::fs;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "read_train_metadata", about = "read train metadata")]
struct Options {
    #[structopt(short = "i", long = "input", long_help = "input dir", required = true, parse(from_os_str))]
    base_dir: path::PathBuf,

    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "info")]
    log_level: String,
}

fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    info!("{:?}", options);

    let mut train_dir = options.base_dir.clone();
    train_dir.push("train");

    let mut train_metadata_path = train_dir.clone();
    train_metadata_path.push("metadata.json");

    let train_metadata_file = fs::File::open(train_metadata_path.as_path()).expect("missing file");
    let br = io::BufReader::new(train_metadata_file);

    let train_metadata: rusty_herbarium::TrainMetadata = serde_json::from_reader(br)?;
    //info!("metadata.info.year: {}", metadata.info.year);

    let images_by_region_and_category_map: collections::HashMap<(i32, i32), Vec<i32>> = train_metadata
        .annotations
        .iter()
        .map(|x| ((x.region_id, x.category_id), x.image_id))
        .into_group_map();
    let mut region_and_category_ids: Vec<_> = images_by_region_and_category_map.keys().collect();
    region_and_category_ids.sort();

    let mut train_feature_size = 0usize;
    let mut train_data_capacity = 0usize;

    for k in region_and_category_ids.iter() {
        let image_ids = images_by_region_and_category_map.get(k).unwrap();
        let filtered_images_ids: Vec<_> = image_ids.iter().take(2).collect();
        for image_id in filtered_images_ids.into_iter() {
            let image = train_metadata.images.iter().find(|e| e.id == *image_id).unwrap();
            let mut image_path = train_dir.clone();
            image_path.push(image.file_name.clone());

            let mut normalized_name = String::new();
            normalized_name.push_str("normalized-");
            normalized_name.push_str(&image_path.as_path().file_name().unwrap().to_string_lossy());

            let mut normalized_path = image_path.parent().unwrap().to_path_buf();
            normalized_path.push(normalized_name.as_str());

            debug!("normalized_path: {}", normalized_path.to_string_lossy());

            let img = image::open(normalized_path).unwrap();

            let size = ((img.width() * img.height()) * 3) as usize;
            if train_feature_size == 0_usize {
                train_feature_size = size;
            }

            train_data_capacity += size;
        }
    }
    debug!("train_data_capacity: {}, train_feature_size: {}", train_data_capacity, train_feature_size);
    let mut train_data: Vec<f64> = Vec::with_capacity(train_data_capacity);
    let mut targets: Vec<f64> = Vec::new();

    for k in region_and_category_ids.iter() {
        let image_ids = images_by_region_and_category_map.get(k).unwrap();
        let filtered_images_ids: Vec<_> = image_ids.iter().take(2).collect();
        for image_id in filtered_images_ids.into_iter() {
            targets.push(k.1.into()); //category_id
            let image = train_metadata.images.iter().find(|e| e.id == *image_id).unwrap();
            let mut image_path = train_dir.clone();
            image_path.push(image.file_name.clone());

            let mut normalized_name = String::new();
            normalized_name.push_str("normalized-");
            normalized_name.push_str(&image_path.as_path().file_name().unwrap().to_string_lossy());

            let mut normalized_path = image_path.parent().unwrap().to_path_buf();
            normalized_path.push(normalized_name.as_str());

            let img = image::open(normalized_path).unwrap();
            let mut train_features: Vec<f64> = Vec::new();
            //features.push(k.0.into()); //region_id

            for x in 0..img.width() {
                for y in 0..img.height() {
                    let pixel = img.get_pixel(x, y);
                    train_features.push(pixel[0].into());
                    train_features.push(pixel[1].into());
                    train_features.push(pixel[2].into());
                }
            }
            train_data.append(&mut train_features);
        }
    }
    info!("targets.len(): {}, train_data.len(): {}", targets.len(), train_data.len());

    let train_data_matrix = linalg::Matrix::new(train_data.len() / train_feature_size, train_feature_size, train_data);
    //info!("train_data_matrix: {:?}", train_data_matrix);
    // let mut transformer = Standardizer::default();
    // let train_data_transformed = transformer.transform(train_data_matrix).unwrap();
    let train_targets = linalg::Vector::new(targets);

    let mut test_dir = options.base_dir.clone();
    test_dir.push("test");

    let mut test_metadata_path = test_dir.clone();
    test_metadata_path.push("metadata.json");

    let test_metadata_file = fs::File::open(test_metadata_path.as_path()).expect("missing file");
    let br = io::BufReader::new(test_metadata_file);

    let test_metadata: rusty_herbarium::TestMetadata = serde_json::from_reader(br)?;

    let mut test_feature_size = 0usize;
    let mut test_data: Vec<f64> = Vec::new();

    for image in test_metadata.images.iter().take(100) {
        let mut image_path = test_dir.clone();
        image_path.push(image.file_name.clone());

        let mut normalized_name = String::new();
        normalized_name.push_str("normalized-");
        normalized_name.push_str(&image_path.as_path().file_name().unwrap().to_string_lossy());

        let mut normalized_path = image_path.parent().unwrap().to_path_buf();
        normalized_path.push(normalized_name.as_str());

        let img = image::open(normalized_path).unwrap();

        let mut test_features: Vec<f64> = Vec::new();
        for x in 0..img.width() {
            for y in 0..img.height() {
                let pixel = img.get_pixel(x, y);
                test_features.push(pixel[0].into());
                test_features.push(pixel[1].into());
                test_features.push(pixel[2].into());
            }
        }
        if test_feature_size == 0usize {
            test_feature_size = test_features.len();
        }
        test_data.append(&mut test_features);
    }
    info!("test_data.len(): {}", test_data.len());

    let test_data_matrix = linalg::Matrix::new(test_data.len() / test_feature_size, test_feature_size, test_data);
    // let mut transformer = Standardizer::default();
    // let test_data_transformed = transformer.transform(test_data_matrix).unwrap();

    let mut model = learning::logistic_reg::LogisticRegressor::new(GradientDesc::new(0.1, 1000));
    model.train(&train_data_matrix, &train_targets).unwrap();
    let predicted_outputs = model.predict(&test_data_matrix).unwrap();

    debug!("training_targets: {:?}", train_targets);
    debug!("predicted_outputs: {:?}", predicted_outputs);

    // let rounded_outputs = outputs.apply(&round);
    // info!("rounded_outputs: {:?}", rounded_outputs);

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
