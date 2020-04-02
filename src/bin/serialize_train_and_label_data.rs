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
use flate2::write::GzEncoder;
use flate2::Compression;
use humantime::format_duration;
use itertools::Itertools;
use log::Level;
use rayon::prelude::*;
use rustlearn::array;
use rustlearn::prelude::*;
use std::collections;
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

    #[structopt(short = "b", long = "base_dir", long_help = "base directory for ", required = true, parse(from_os_str))]
    base_dir: path::PathBuf,

    #[structopt(short = "o", long = "output_dir", long_help = "output directory", required = true, parse(from_os_str))]
    output_dir: path::PathBuf,

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

    debug!("reading: {}", train_metadata_path.to_string_lossy());
    let train_metadata_file = fs::File::open(train_metadata_path.as_path()).expect("missing file");
    let br = io::BufReader::new(train_metadata_file);

    let training_metadata: rusty_herbarium::TrainMetadata = serde_json::from_reader(br)?;

    let image_ids_by_category_map: collections::HashMap<i32, Vec<i32>> =
        training_metadata.annotations.iter().map(|x| (x.category_id, x.image_id)).into_group_map();

    let mut category_ids: Vec<_> = image_ids_by_category_map.keys().cloned().take(100).collect();
    category_ids.sort();

    let mut training_image_path_by_category_map: collections::BTreeMap<i32, Vec<path::PathBuf>> = collections::BTreeMap::new();
    let mut validation_image_path_by_category_map: collections::BTreeMap<i32, Vec<path::PathBuf>> = collections::BTreeMap::new();

    let mut training_rows = 0;
    let mut validation_rows = 0;

    for i in category_ids.iter() {
        let image_ids = image_ids_by_category_map.get(i).unwrap();
        // only grabbing 2 images per category (species)...for now
        let filtered_images_ids: Vec<_> = image_ids.iter().take(2).collect();
        for image_id in filtered_images_ids.iter() {
            let image = training_metadata.images.iter().find(|e| &e.id == *image_id).unwrap();
            let mut image_path = train_dir.clone();
            image_path.push(image.file_name.clone());
            training_image_path_by_category_map.entry(*i).or_insert(Vec::new()).push(image_path);
            training_rows += 1;
        }

        if image_ids.len() > 12 {
            let filtered_images_ids_for_validation: Vec<_> = image_ids
                .iter()
                .filter(|image_id| !filtered_images_ids.contains(image_id))
                .take(2)
                .collect();
            for image_id in filtered_images_ids_for_validation.into_iter() {
                let image = training_metadata.images.iter().find(|e| e.id == *image_id).unwrap();
                let mut image_path = train_dir.clone();
                image_path.push(image.file_name.clone());
                validation_image_path_by_category_map.entry(*i).or_insert(Vec::new()).push(image_path);
                validation_rows += 1;
            }
        }
    }

    debug!("training_rows: {}, validation_rows: {}", training_rows, validation_rows);

    let (training_data, training_labels) = get_data_and_labels(options.width, options.height, training_rows, training_image_path_by_category_map)?;

    let mut training_labels_output = options.output_dir.clone();
    training_labels_output.push(format!("herbarium-training-labels-{}x{}.ser.gz", options.width, options.height));
    info!("writing: {}", training_labels_output.to_string_lossy());

    let training_labels_writer = io::BufWriter::new(fs::File::create(training_labels_output.as_path()).unwrap());
    let mut training_labels_encoder = GzEncoder::new(training_labels_writer, Compression::default());
    bincode::serialize_into(&mut training_labels_encoder, &training_labels).unwrap();

    let mut training_data_output = options.output_dir.clone();
    training_data_output.push(format!("herbarium-training-data-{}x{}.ser.gz", options.width, options.height));
    info!("writing: {}", training_data_output.to_string_lossy());

    let training_data_writer = io::BufWriter::new(fs::File::create(training_data_output.as_path()).unwrap());
    let mut training_data_encoder = GzEncoder::new(training_data_writer, Compression::default());
    bincode::serialize_into(&mut training_data_encoder, &training_data).unwrap();

    let (validation_data, validation_labels) =
        get_data_and_labels(options.width, options.height, validation_rows, validation_image_path_by_category_map)?;

    let mut validation_labels_output = options.output_dir.clone();
    validation_labels_output.push(format!("herbarium-validation-labels-{}x{}.ser.gz", options.width, options.height));
    info!("writing: {}", validation_labels_output.to_string_lossy());

    let validation_labels_writer = io::BufWriter::new(fs::File::create(validation_labels_output.as_path()).unwrap());
    let mut validation_labels_encoder = GzEncoder::new(validation_labels_writer, Compression::default());
    bincode::serialize_into(&mut validation_labels_encoder, &validation_labels).unwrap();

    let mut validation_data_output = options.output_dir.clone();
    validation_data_output.push(format!("herbarium-validation-data-{}x{}.ser.gz", options.width, options.height));
    info!("writing: {}", validation_data_output.to_string_lossy());

    let validation_data_writer = io::BufWriter::new(fs::File::create(validation_data_output.as_path()).unwrap());
    let mut validation_data_encoder = GzEncoder::new(validation_data_writer, Compression::default());
    bincode::serialize_into(&mut validation_data_encoder, &validation_data).unwrap();

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}

fn get_data_and_labels(
    width: u32,
    height: u32,
    rows: usize,
    image_path_by_category_map: collections::BTreeMap<i32, Vec<path::PathBuf>>,
) -> io::Result<(array::sparse::SparseRowArray, array::dense::Array)> {
    let col_size = ((width * height) * 3) as usize;

    let image_path_by_category_map_keys: Vec<_> = image_path_by_category_map.keys().cloned().collect();

    // let mut training_labels: Vec<f32> = Vec::with_capacity(20);
    // let mut training_data = array::sparse::SparseRowArray::zeros(20, col_size);

    let mut labels: Vec<f32> = Vec::with_capacity(rows);
    let mut data = array::sparse::SparseRowArray::zeros(rows, col_size);

    for (i, category_id) in image_path_by_category_map_keys.iter().enumerate() {
        labels.push(*category_id as f32);
        debug!("category_id: {}", category_id);

        for image_path in image_path_by_category_map.get(category_id).unwrap() {
            // debug!("image_path: {}", image_path.to_string_lossy());

            let img = image::open(image_path.as_path()).unwrap();
            // risk cropping more from the bottom as roots don't offer identifying species features
            // stems, leafs, and flowers are where it's at
            let mut img = rusty_herbarium::crop_image(img, 30, 30, 80, 140);
            // original images are roughly 680x1000
            // resulting cropping will return roughly 620x780

            image::imageops::invert(&mut img);

            let mut img = image::imageops::brighten(&mut img, 10);
            let img = image::imageops::contrast(&mut img, 20.0);

            let img = image::imageops::resize(&img, width, height, image::imageops::FilterType::Gaussian);

            let mut idx = 0;
            for x in 0..img.width() {
                for y in 0..img.height() {
                    let pixel = img.get_pixel(x, y);
                    let red = pixel[0];
                    let green = pixel[1];
                    let blue = pixel[2];

                    if red > 20 && green > 20 && blue > 20 {
                        data.set(i, idx, red as f32 / 255.0);
                        data.set(i, idx + 1, green as f32 / 255.0);
                        data.set(i, idx + 2, blue as f32 / 255.0);
                    }
                    idx += 3;
                    // debug!("i: {}, idx: {}", i, idx);
                }
            }
        }
        // if i == 19 {
        //     break;
        // }
    }

    Ok((data, array::dense::Array::from(labels)))
}
