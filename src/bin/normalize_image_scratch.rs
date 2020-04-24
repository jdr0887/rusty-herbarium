#[macro_use]
extern crate log;
extern crate humantime;
extern crate image;
extern crate statistical;
extern crate structopt;

use crate::image::GenericImageView;
use humantime::format_duration;
use itertools::Itertools;
use log::Level;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "normalize_image", about = "normalize image")]
struct Options {
    #[structopt(short = "w", long = "width", long_help = "width", default_value = "315")]
    width: u32,

    #[structopt(short = "h", long = "height", long_help = "width", default_value = "390")]
    height: u32,

    #[structopt(short = "i", long = "input", long_help = "input file", required = true, parse(from_os_str))]
    input: path::PathBuf,

    #[structopt(short = "o", long = "output", long_help = "output file", required = true, parse(from_os_str))]
    output: path::PathBuf,

    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "info")]
    log_level: String,
}
fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    let img = image::open(options.input).unwrap();

    let img = rusty_herbarium::preprocessing_step_1(img);
    let img = rusty_herbarium::preprocessing_step_2(img);
    // let img = preprocessing_step_3(img);
    // let img = preprocessing_step_4(img);

    let img = image::imageops::resize(&img, options.width, options.height, image::imageops::FilterType::CatmullRom);

    img.save(options.output).ok();

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
