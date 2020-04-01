#[macro_use]
extern crate log;
extern crate humantime;
extern crate image;
extern crate structopt;

use humantime::format_duration;
use log::Level;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "normalize_image", about = "normalize image")]
struct Options {
    #[structopt(short = "i", long = "input", long_help = "input file", required = true, parse(from_os_str))]
    input: path::PathBuf,

    #[structopt(short = "o", long = "output", long_help = "output file", required = true, parse(from_os_str))]
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

    let img = image::open(options.input).unwrap();
    // 680x1000
    // 620x780
    let mut normalized_image = rusty_herbarium::crop_image(img, 30, 30, 80, 140);
    image::imageops::invert(&mut normalized_image);
    image::imageops::brighten(&mut normalized_image, 30);
    image::imageops::contrast(&mut normalized_image, 60.0);
    // normalized_image.save(options.output).ok();

    // let resized_image = image::imageops::resize(&normalized_image, 440, 660, image::imageops::FilterType::Gaussian);
    // let resized_image = image::imageops::resize(&normalized_image, 85, 112, image::imageops::FilterType::Gaussian);
    // let resized_image = image::imageops::resize(&normalized_image, 315, 400, image::imageops::FilterType::Gaussian);
    // let resized_image = image::imageops::resize(&normalized_image, 520, 660, image::imageops::FilterType::Gaussian);
    let resized_image = image::imageops::resize(&normalized_image, 310, 390, image::imageops::FilterType::Gaussian);
    resized_image.save(options.output).ok();

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
