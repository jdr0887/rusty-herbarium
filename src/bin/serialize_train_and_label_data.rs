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

    #[structopt(short = "c", long = "category_limit", long_help = "category limit", default_value = "0")]
    category_limit: usize,

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

    let mut category_ids: Vec<i32> = training_metadata.annotations.iter().map(|x| x.category_id).collect();
    category_ids.sort();
    category_ids.dedup();

    if options.category_limit > 0 {
        category_ids = category_ids.iter().cloned().take(options.category_limit).collect();
    }

    let image_ids_by_category_map: collections::HashMap<i32, Vec<i32>> = training_metadata.annotations.iter().map(|x| (x.category_id, x.image_id)).into_group_map();

    let mut image_ids_by_category_map_keys: Vec<_> = image_ids_by_category_map.keys().cloned().collect();
    image_ids_by_category_map_keys.sort();

    let mut training_image_path_by_category_map: collections::BTreeMap<i32, Vec<path::PathBuf>> = collections::BTreeMap::new();
    let mut validation_image_path_by_category_map: collections::BTreeMap<i32, Vec<path::PathBuf>> = collections::BTreeMap::new();

    info!("category_ids: {:?}", category_ids);

    for category_id in category_ids.iter() {
        let image_ids = image_ids_by_category_map.get(category_id).unwrap();
        // only grabbing N images per category (species)
        let filtered_images_ids: Vec<_> = image_ids.iter().take(10).collect();
        for image_id in filtered_images_ids.iter() {
            let image = training_metadata.images.iter().find(|e| &e.id == *image_id).unwrap();
            let mut image_path = train_dir.clone();
            image_path.push(image.file_name.clone());
            training_image_path_by_category_map.entry(*category_id).or_insert(Vec::new()).push(image_path);
        }

        if image_ids.len() > 25 {
            let filtered_images_ids_for_validation: Vec<_> = image_ids.iter().filter(|image_id| !filtered_images_ids.contains(image_id)).take(1).collect();
            for image_id in filtered_images_ids_for_validation.into_iter() {
                let image = training_metadata.images.iter().find(|e| e.id == *image_id).unwrap();
                let mut image_path = train_dir.clone();
                image_path.push(image.file_name.clone());
                validation_image_path_by_category_map.entry(*category_id).or_insert(Vec::new()).push(image_path);
            }
        }
    }

    let (training_data, training_labels) = get_data_and_labels(options.width, options.height, training_image_path_by_category_map)?;

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

    let (validation_data, validation_labels) = get_data_and_labels(options.width, options.height, validation_image_path_by_category_map)?;

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

fn get_data_and_labels(width: u32, height: u32, image_path_by_category_map: collections::BTreeMap<i32, Vec<path::PathBuf>>) -> io::Result<(Vec<Vec<f32>>, Vec<f32>)> {
    // let col_size = ((width * height) * 3) as usize;
    let col_size = (width * height) as usize;

    let mut image_path_by_category_map_keys: Vec<_> = image_path_by_category_map.keys().cloned().collect();
    image_path_by_category_map_keys.sort();

    let mut training_labels: Vec<f32> = Vec::new();
    let mut training_data: Vec<Vec<f32>> = Vec::new();

    for category_id in image_path_by_category_map_keys.iter() {
        info!("category_id: {}", category_id);

        for image_path in image_path_by_category_map.get(category_id).unwrap() {
            for _ in 0..4 {
                training_labels.push(*category_id as f32);
            }
            // debug!("image_path: {}", image_path.to_string_lossy());

            let img = image::open(image_path.as_path()).unwrap();

            let img = rusty_herbarium::preprocessing_step_1(img);
            let img = rusty_herbarium::preprocessing_step_2(img);
            // let img = preprocessing_step_3(img);
            // let img = preprocessing_step_4(img);

            let mut img = image::imageops::resize(&img, width, height, image::imageops::FilterType::CatmullRom);

            let mut img = image::imageops::grayscale(&mut img);

            let mut image_data = Vec::with_capacity(col_size);
            for x in 0..img.width() {
                for y in 0..img.height() {
                    let pixel = img.get_pixel(x, y);
                    image_data.push(pixel[0] as f32 / 255.0);
                }
            }

            training_data.push(image_data);

            let rotated90_img = image::imageops::rotate90(&mut img);

            let mut image_data = Vec::with_capacity(col_size);
            for x in 0..rotated90_img.width() {
                for y in 0..rotated90_img.height() {
                    let pixel = rotated90_img.get_pixel(x, y);
                    image_data.push(pixel[0] as f32 / 255.0);
                }
            }
            training_data.push(image_data);

            let rotated180_img = image::imageops::rotate180(&mut img);

            let mut image_data = Vec::with_capacity(col_size);
            for x in 0..rotated180_img.width() {
                for y in 0..rotated180_img.height() {
                    let pixel = rotated180_img.get_pixel(x, y);
                    image_data.push(pixel[0] as f32 / 255.0);
                }
            }
            training_data.push(image_data);

            let rotated270_img = image::imageops::rotate270(&mut img);

            let mut image_data = Vec::with_capacity(col_size);
            for x in 0..rotated270_img.width() {
                for y in 0..rotated270_img.height() {
                    let pixel = rotated270_img.get_pixel(x, y);
                    image_data.push(pixel[0] as f32 / 255.0);
                }
            }
            training_data.push(image_data);
        }
    }

    Ok((training_data, training_labels))
}

fn get_data_and_labels_orig(
    width: u32,
    height: u32,
    rows: usize,
    image_path_by_category_map: collections::BTreeMap<i32, Vec<path::PathBuf>>,
) -> io::Result<(Vec<Vec<f32>>, Vec<f32>)> {
    // let col_size = ((width * height) * 3) as usize;
    let col_size = (width * height) as usize;

    let mut image_path_by_category_map_keys: Vec<_> = image_path_by_category_map.keys().cloned().collect();
    image_path_by_category_map_keys.sort();

    let mut training_labels: Vec<f32> = Vec::with_capacity(rows);
    let mut training_data: Vec<Vec<f32>> = Vec::with_capacity(rows);

    // let mut training_data = array::sparse::SparseRowArray::zeros(rows, col_size);

    // let mut row_idx = 0;
    for category_id in image_path_by_category_map_keys.iter() {
        info!("category_id: {}", category_id);

        for image_path in image_path_by_category_map.get(category_id).unwrap() {
            for _ in 0..4 {
                training_labels.push(*category_id as f32);
            }
            // debug!("image_path: {}", image_path.to_string_lossy());

            // risk cropping more from the bottom as roots don't offer identifying species features
            // stems, leafs, and flowers are the identifying features
            // original images are roughly 680x1000 resulting cropping will return roughly 620x780

            let img = image::open(image_path.as_path()).unwrap();

            let img = rusty_herbarium::preprocessing_step_1(img);
            let img = rusty_herbarium::preprocessing_step_2(img);
            // let img = preprocessing_step_3(img);
            // let img = preprocessing_step_4(img);

            let mut img = image::imageops::resize(&img, width, height, image::imageops::FilterType::CatmullRom);

            // let img = rusty_herbarium::crop_image(img, 140, 140, 180, 220);
            // // let img = rusty_herbarium::crop_image(img, 30, 30, 80, 140);
            // let mut img = image::imageops::resize(&img, width, height, image::imageops::FilterType::CatmullRom);
            let mut img = image::imageops::grayscale(&mut img);
            // image::imageops::invert(&mut img);
            // let mut img = image::imageops::brighten(&mut img, -10);
            // let mut img = image::imageops::contrast(&mut img, 40.0);

            // let mut col_idx = 0;
            let mut image_data = Vec::with_capacity(col_size);
            for x in 0..img.width() {
                for y in 0..img.height() {
                    let pixel = img.get_pixel(x, y);
                    image_data.push(pixel[0] as f32 / 255.0);
                    // image_data.push(pixel[1] as f32 / 255.0);
                    // image_data.push(pixel[2] as f32 / 255.0);
                    // col_idx += 1;
                }
            }

            debug!("image_data.len(): {}", image_data.len());
            training_data.push(image_data);
            // row_idx += 1;

            // let mut img = image::imageops::rotate90(&mut img);
            //
            // let mut col_idx = 0;
            // for x in 0..img.width() {
            //     for y in 0..img.height() {
            //         let pixel = img.get_pixel(x, y);
            //         data.set(row_idx, col_idx, pixel[0] as f32 / 255.0);
            //         col_idx += 1;
            //     }
            // }
            // row_idx += 1;
            //
            // let mut img = image::imageops::rotate180(&mut img);
            //
            // let mut col_idx = 0;
            // for x in 0..img.width() {
            //     for y in 0..img.height() {
            //         let pixel = img.get_pixel(x, y);
            //         data.set(row_idx, col_idx, pixel[0] as f32 / 255.0);
            //         col_idx += 1;
            //     }
            // }
            // row_idx += 1;
            //
            // let img = image::imageops::rotate270(&mut img);
            //
            // let mut col_idx = 0;
            // for x in 0..img.width() {
            //     for y in 0..img.height() {
            //         let pixel = img.get_pixel(x, y);
            //         data.set(row_idx, col_idx, pixel[0] as f32 / 255.0);
            //         col_idx += 1;
            //     }
            // }
            // row_idx += 1;
        }
        // if i == 19 {
        //     break;
        // }
    }

    Ok((training_data, training_labels))
}
