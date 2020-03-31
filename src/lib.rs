extern crate bincode;
extern crate lazy_static;
extern crate libm;
#[macro_use]
extern crate log;
extern crate rayon;
extern crate regex;
extern crate rustlearn;
extern crate rusty_machine;
extern crate serde;
extern crate serde_derive;

use crate::rustlearn::prelude::IndexableMatrix;
use flate2::write::GzEncoder;
use flate2::Compression;
use image::GenericImageView;
use itertools::Itertools;
use rustlearn::array;
use serde::{Deserialize, Serialize};
use std::collections;
use std::fs;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;
use walkdir::WalkDir;

#[derive(Serialize, Deserialize, Debug)]
pub struct Annotation {
    pub id: i32,
    pub image_id: i32,
    pub category_id: i32,
    pub region_id: i32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Category {
    pub id: i32,
    //pub name: String,
    pub family: String,
    pub genus: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Image {
    pub id: i32,
    pub width: i32,
    pub height: i32,
    pub file_name: String,
    pub license: i32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Info {
    pub year: i32,
    pub version: String,
    pub url: String,
    pub description: String,
    pub contributor: String,
    pub date_created: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct License {
    pub id: i32,
    pub name: String,
    pub url: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Region {
    pub id: i32,
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TrainMetadata {
    pub annotations: Vec<Annotation>,
    pub categories: Vec<Category>,
    pub images: Vec<Image>,
    pub info: Info,
    pub licenses: Vec<License>,
    pub regions: Vec<Region>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TestMetadata {
    pub images: Vec<Image>,
    pub info: Info,
    pub licenses: Vec<License>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PixelColor {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

pub fn crop_image(
    mut img: image::DynamicImage,
    from_left: u32,
    from_right: u32,
    from_top: u32,
    from_bottom: u32,
) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let img_height = img.height();
    let img_width = img.width();

    let mut left_border_cropped_image = image::imageops::crop(&mut img, from_left, 0, img_width, img_height).to_image();
    let left_border_cropped_image_width = left_border_cropped_image.width();
    let left_border_cropped_image_height = left_border_cropped_image.height();

    let mut right_border_cropped_image = image::imageops::crop(
        &mut left_border_cropped_image,
        0,
        0,
        left_border_cropped_image_width - from_right,
        left_border_cropped_image_height,
    )
    .to_image();

    let right_border_cropped_image_width = right_border_cropped_image.width();
    let right_border_cropped_image_height = right_border_cropped_image.height();

    let mut top_border_cropped_image = image::imageops::crop(
        &mut right_border_cropped_image,
        0,
        from_top,
        right_border_cropped_image_width,
        right_border_cropped_image_height,
    )
    .to_image();

    let top_border_cropped_image_width = top_border_cropped_image.width();
    let top_border_cropped_image_height = top_border_cropped_image.height();

    let bottom_border_cropped_image = image::imageops::crop(
        &mut top_border_cropped_image,
        0,
        0,
        top_border_cropped_image_width,
        top_border_cropped_image_height - from_bottom,
    )
    .to_image();
    bottom_border_cropped_image
}

// pub fn normalized_train_data(
//     train_dir: &path::PathBuf,
//     train_metadata: &TrainMetadata,
//     resized_width: u32,
//     resized_height: u32,
// ) -> io::Result<(array::sparse::SparseRowArray, array::dense::Array)> {
pub fn normalized_train_data(
    train_dir: &path::PathBuf,
    train_metadata: &TrainMetadata,
    resized_width: u32,
    resized_height: u32,
) -> io::Result<(array::dense::Array, array::dense::Array)> {
    let mut file_entries = Vec::new();

    let images_by_region_and_category_map: collections::HashMap<(i32, i32), Vec<i32>> = train_metadata
        .annotations
        .iter()
        .map(|x| ((x.region_id, x.category_id), x.image_id))
        .into_group_map();
    let mut region_and_category_ids: Vec<_> = images_by_region_and_category_map.keys().collect();
    region_and_category_ids.sort();

    for k in region_and_category_ids.iter() {
        let image_ids = images_by_region_and_category_map.get(k).unwrap();
        let filtered_images_ids: Vec<_> = image_ids.iter().take(2).collect();
        for image_id in filtered_images_ids.into_iter() {
            let image = train_metadata.images.iter().find(|e| e.id == *image_id).unwrap();
            let mut image_path = train_dir.clone();
            image_path.push(image.file_name.clone());
            file_entries.push((k.1, image_path));
        }
    }

    debug!("file_entries.len(): {}", file_entries.len());

    let col_size = ((resized_width * resized_height) * 3) as usize;
    // width * height * 3 colors (rgb)
    let mut labels: Vec<f32> = Vec::with_capacity(file_entries.len());
    // let mut data = array::sparse::SparseRowArray::zeros(file_entries.len(), (resized_width * resized_height * 3) as usize);
    // let mut data: Vec<Vec<PixelColor>> = Vec::with_capacity(file_entries.len());
    let mut data: Vec<Vec<f32>> = Vec::with_capacity(file_entries.len());

    for (i, entry) in file_entries.into_iter().enumerate() {
        debug!("i: {}, entry: {}", i, entry.1.to_string_lossy());
        let img = image::open(entry.1).unwrap();
        labels.push(entry.0 as f32);
        // rist cropping more from the bottom as roots don't offer identifying species features
        // stems, leafs, and flowers are where it's at
        let mut cropped_image = crop_image(img, 30, 30, 80, 140);
        // original images are roughly 680x1000
        // resulting cropping will return roughly 620x780

        image::imageops::invert(&mut cropped_image);
        image::imageops::brighten(&mut cropped_image, 30);
        image::imageops::contrast(&mut cropped_image, 60.0);

        let resized_image = image::imageops::resize(&cropped_image, resized_width, resized_height, image::imageops::FilterType::Gaussian);
        let mut features = Vec::with_capacity(col_size);
        let mut idx = 0;
        for x in 0..resized_image.width() {
            for y in 0..resized_image.height() {

                let pixel = resized_image.get_pixel(x, y);
                let red = pixel[0];
                let green = pixel[1];
                let blue = pixel[2];

                features.push(red as f32 / 255.0);
                features.push(green as f32 / 255.0);
                features.push(blue as f32 / 255.0);
                // let pixel_color = PixelColor {
                //     red: red,
                //     green: green,
                //     blue: blue,
                // };
                // features.push(pixel_color);
                // if red > 20 && green > 20 && blue > 20 {
                //     data.set(i, idx, red / 255.0);
                //     data.set(i, idx + 1, green / 255.0);
                //     data.set(i, idx + 2, blue / 255.0);
                // }
                // idx += 3;
                // debug!("i: {}, idx: {}", i, idx);
            }
        }
        data.push(features);
        // if i == 1 {
        //     break;
        // }
    }

    // debug!("data.rows(): {}, data.cols(): {}", data.rows(), data.cols());
    debug!("data.len(): {}", data.len());
    debug!("labels.len(): {}", labels.len());

    let mut output_path = path::PathBuf::new();
    output_path.push("/tmp");
    output_path.push("output.ser");

    let output_file = fs::File::create(output_path.as_path()).unwrap();
    let mut writer = io::BufWriter::new(output_file);
    bincode::serialize_into(&mut writer, &data).unwrap();

    // let mut data = array::sparse::SparseRowArray::zeros(train_data.len(), col_size);
    //
    // for (row_idx, row) in train_data.into_iter().enumerate() {
    //     let mut idx = 0;
    //     for col in row.into_iter() {
    //         if col.red > 20 && col.green > 20 && col.blue > 20 {
    //             data.set(row_idx, idx, col.red as f32);
    //             data.set(row_idx, idx + 1, col.green as f32);
    //             data.set(row_idx, idx + 2, col.blue as f32);
    //         }
    //         idx += 3;
    //     }
    // }

    // let mut data: Vec<Vec<PixelColor>> = Vec::with_capacity(file_entries.len());

    debug!("wrote: {}", output_path.to_string_lossy());
    Ok((array::dense::Array::from(&data), array::dense::Array::from(labels)))
    // Ok((data, array::dense::Array::from(labels)))
}
