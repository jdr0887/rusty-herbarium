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

    let col_size = ((options.width * options.height) * 3) as usize;

    let mut test_dir = options.base_dir.clone();
    test_dir.push("test");

    let mut test_metadata_path = test_dir.clone();
    test_metadata_path.push("metadata.json");

    debug!("reading: {}", test_metadata_path.to_string_lossy());
    let test_metadata_file = fs::File::open(test_metadata_path.as_path()).expect("missing file");
    let br = io::BufReader::new(test_metadata_file);

    let testing_metadata: rusty_herbarium::TestMetadata = serde_json::from_reader(br)?;

    let mut testing_data = array::sparse::SparseRowArray::zeros(testing_metadata.images.len(), col_size);

    for (i, image) in testing_metadata.images.iter().enumerate() {
        let mut image_path = test_dir.clone();
        image_path.push(image.file_name.clone());
        let img = image::open(image_path.as_path()).unwrap();
        // risk cropping more from the bottom as roots don't offer identifying species features
        // stems, leafs, and flowers are where it's at
        let mut img = rusty_herbarium::crop_image(img, 30, 30, 80, 140);
        // original images are roughly 680x1000
        // resulting cropping will return roughly 620x780

        image::imageops::invert(&mut img);

        let mut img = image::imageops::brighten(&mut img, 10);
        let img = image::imageops::contrast(&mut img, 20.0);

        let img = image::imageops::resize(&img, options.width, options.height, image::imageops::FilterType::Gaussian);

        let mut idx = 0;
        for x in 0..img.width() {
            for y in 0..img.height() {
                let pixel = img.get_pixel(x, y);
                let red = pixel[0];
                let green = pixel[1];
                let blue = pixel[2];

                if red > 20 && green > 20 && blue > 20 {
                    testing_data.set(i, idx, red as f32 / 255.0);
                    testing_data.set(i, idx + 1, green as f32 / 255.0);
                    testing_data.set(i, idx + 2, blue as f32 / 255.0);
                }
                idx += 3;
                // debug!("i: {}, idx: {}", i, idx);
            }
        }
    }

    let mut testing_data_output = options.output_dir.clone();
    testing_data_output.push(format!("herbarium-testing-data-{}x{}.ser.gz", options.width, options.height));
    info!("writing: {}", testing_data_output.to_string_lossy());

    let testing_data_writer = io::BufWriter::new(fs::File::create(testing_data_output.as_path()).unwrap());
    let mut testing_data_encoder = GzEncoder::new(testing_data_writer, Compression::default());
    bincode::serialize_into(&mut testing_data_encoder, &testing_data).unwrap();

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
