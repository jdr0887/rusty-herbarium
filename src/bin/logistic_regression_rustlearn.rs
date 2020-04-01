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
use itertools::Itertools;
use log::Level;
use rustlearn::array;
use rustlearn::linear_models::sgdclassifier;
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

    debug!("reading: {}", train_metadata_path.to_string_lossy());
    let train_metadata_file = fs::File::open(train_metadata_path.as_path()).expect("missing file");
    let br = io::BufReader::new(train_metadata_file);

    let training_metadata: rusty_herbarium::TrainMetadata = serde_json::from_reader(br)?;

    let image_ids_by_category_map: collections::HashMap<i32, Vec<i32>> =
        training_metadata.annotations.iter().map(|x| (x.category_id, x.image_id)).into_group_map();

    let mut category_ids: Vec<_> = image_ids_by_category_map.keys().cloned().collect();
    category_ids.sort();

    let mut image_path_by_category_map: collections::BTreeMap<i32, Vec<path::PathBuf>> = collections::BTreeMap::new();

    for i in category_ids.iter() {
        let image_ids = image_ids_by_category_map.get(i).unwrap();
        // only grabbing 2 images per category...for now
        let filtered_images_ids: Vec<_> = image_ids.iter().take(2).collect();
        for image_id in filtered_images_ids.into_iter() {
            let image = training_metadata.images.iter().find(|e| e.id == *image_id).unwrap();
            let mut image_path = train_dir.clone();
            image_path.push(image.file_name.clone());
            image_path_by_category_map.entry(*i).or_insert(Vec::new()).push(image_path);
        }
    }

    // let width = 315u32;
    // let height = 390u32;
    let width = 242u32;
    let height = 300u32;

    let col_size = ((width * height) * 3) as usize;

    let mut model = sgdclassifier::Hyperparameters::new(col_size)
        .learning_rate(0.1)
        .l1_penalty(0.0)
        .l2_penalty(0.0)
        .one_vs_rest();

    let image_path_by_category_map_keys: Vec<_> = image_path_by_category_map.keys().cloned().collect();

    for chunk in image_path_by_category_map_keys.chunks(100) {
        let mut labels: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut train_data = array::sparse::SparseRowArray::zeros(chunk.len(), col_size);

        for (i, category_id) in chunk.iter().enumerate() {
            labels.push(*category_id as f32);
            for image_paths in image_path_by_category_map.get(category_id) {
                for image_path in image_paths.iter() {
                    debug!("category_id: {}, image_path: {}", category_id, image_path.to_string_lossy());

                    let img = image::open(image_path.as_path()).unwrap();
                    // risk cropping more from the bottom as roots don't offer identifying species features
                    // stems, leafs, and flowers are where it's at
                    let mut cropped_image = rusty_herbarium::crop_image(img, 30, 30, 80, 140);
                    // original images are roughly 680x1000
                    // resulting cropping will return roughly 620x780

                    image::imageops::invert(&mut cropped_image);
                    image::imageops::brighten(&mut cropped_image, 30);
                    image::imageops::contrast(&mut cropped_image, 60.0);

                    let resized_image = image::imageops::resize(&cropped_image, width, height, image::imageops::FilterType::Gaussian);
                    let mut idx = 0;
                    for x in 0..resized_image.width() {
                        for y in 0..resized_image.height() {
                            let pixel = resized_image.get_pixel(x, y);
                            let red = pixel[0];
                            let green = pixel[1];
                            let blue = pixel[2];

                            if red > 20 && green > 20 && blue > 20 {
                                train_data.set(i, idx, red as f32 / 255.0);
                                train_data.set(i, idx + 1, green as f32 / 255.0);
                                train_data.set(i, idx + 2, blue as f32 / 255.0);
                            }
                            idx += 3;
                            // debug!("i: {}, idx: {}", i, idx);
                        }
                    }
                }
            }
        }

        let start_fitting = Instant::now();
        model.fit(&train_data, &array::dense::Array::from(labels)).unwrap();
        debug!("model fitting duration: {}", format_duration(start_fitting.elapsed()).to_string());
    }

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
