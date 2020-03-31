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
use std::collections;
use std::fs;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "train_metadata_metrics", about = "train metadata metrics")]
struct Options {
    #[structopt(short = "i", long = "metadata_path", long_help = "metadata path", required = true, parse(from_os_str))]
    metadata_path: path::PathBuf,

    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "Info")]
    log_level: String,
}

fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    let train_metadata_file = fs::File::open(options.metadata_path.as_path()).expect("missing file");
    let br = io::BufReader::new(train_metadata_file);

    let metadata: rusty_herbarium::TrainMetadata = serde_json::from_reader(br)?;
    //info!("metadata.info.year: {}", metadata.info.year);

    let categories_by_region_map: collections::HashMap<i32, Vec<i32>> =
        metadata.annotations.iter().map(|x| (x.region_id, x.category_id)).into_group_map();
    let mut categories_by_region_map_keys: Vec<_> = categories_by_region_map.keys().collect();
    categories_by_region_map_keys.sort();

    info!("region count: {}", categories_by_region_map_keys.len());
    for k in categories_by_region_map_keys.iter() {
        info!("region: {}, categories count: {}", k, categories_by_region_map.get(k).unwrap().len());
    }

    let mut images_by_family_map: collections::HashMap<String, Vec<i32>> = collections::HashMap::new();
    for annotation in metadata.annotations.iter() {
        let category = metadata.categories.iter().find(|e| e.id == annotation.category_id).unwrap();
        images_by_family_map
            .entry(category.family.clone())
            .or_insert(Vec::new())
            .push(annotation.image_id);
    }

    let mut images_by_family_map_keys: Vec<_> = images_by_family_map.keys().collect();
    images_by_family_map_keys.sort();

    info!("families count: {}", images_by_family_map_keys.len());
    for k in images_by_family_map_keys.into_iter() {
        info!("family: {}, images count: {}", k, images_by_family_map.get(k).unwrap().len());
    }

    let images_by_category_map: collections::HashMap<i32, Vec<i32>> =
        metadata.annotations.iter().map(|x| (x.category_id, x.image_id)).into_group_map();
    let mut images_by_category_map_keys: Vec<_> = images_by_category_map.keys().collect();
    images_by_category_map_keys.sort();
    info!("categories count: {}", images_by_category_map_keys.len());

    for k in images_by_category_map_keys.iter() {
        info!("category: {}, images count: {}", k, images_by_category_map.get(k).unwrap().len());
    }

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
