#[macro_use]
extern crate log;
extern crate humantime;
extern crate image;
extern crate structopt;
extern crate threadpool;
extern crate walkdir;

use humantime::format_duration;
use log::Level;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;
use walkdir::WalkDir;

#[derive(StructOpt, Debug)]
#[structopt(name = "normalize_all_images", about = "normalize all images")]
struct Options {
    #[structopt(short = "i", long = "input", long_help = "input dir", required = true, parse(from_os_str))]
    base_dir: path::PathBuf,

    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "Info")]
    log_level: String,
}
fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    let file_entries: Vec<_> = WalkDir::new(options.base_dir.clone()).into_iter().filter_map(|e| e.ok()).collect();

    let mut filtered_files = Vec::new();
    for file_entry in file_entries {
        let entry_path = file_entry.into_path();
        if !entry_path.is_dir() && entry_path.to_string_lossy().ends_with("jpg") && !entry_path.to_string_lossy().contains("normalized") {
            filtered_files.push(entry_path.clone());
        }
    }
    info!("filtered_files.len(): {}", filtered_files.len());

    let pool = threadpool::ThreadPool::new(num_cpus::get() / 2);

    for entry in filtered_files.into_iter() {
        // info!("parent_dir: {:?}", parent_entry_path);
        if !entry.is_dir() && entry.to_string_lossy().ends_with("jpg") && !entry.to_string_lossy().contains("normalized") {
            pool.execute(move || {
                let img = image::open(entry.clone()).unwrap();
                let mut cropped_image = rusty_herbarium::crop_image(img, 25, 25, 75, 100);
                image::imageops::invert(&mut cropped_image);

                //let resized_image = image::imageops::resize(&cropped_image, 440, 660, image::imageops::FilterType::Gaussian);
                // roughly 7k images loaded

                //let resized_image = image::imageops::resize(&cropped_image, 350, 450, image::imageops::FilterType::Gaussian);
                // roughly 14k images loaded

                let resized_image = image::imageops::resize(&cropped_image, 85, 112, image::imageops::FilterType::Gaussian);
                let parent_entry_path = entry.parent().unwrap();
                let mut output = parent_entry_path.to_path_buf();
                output.push(format!("normalized-{}", entry.file_name().unwrap().to_string_lossy()));
                // info!("writing: {:?}", output.as_path().to_str());
                resized_image.save(output.as_path()).ok();
            });
        }
    }

    pool.join();

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
