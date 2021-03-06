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

use image::GenericImageView;
use itertools::Itertools;
use rustlearn::array;
use serde::{Deserialize, Serialize};
use std::collections;
use std::fs;
use std::io;
use std::path;

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

pub fn crop_image(mut img: image::DynamicImage, from_left: u32, from_right: u32, from_top: u32, from_bottom: u32) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
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

    let images_by_region_and_category_map: collections::HashMap<(i32, i32), Vec<i32>> =
        train_metadata.annotations.iter().map(|x| ((x.region_id, x.category_id), x.image_id)).into_group_map();
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
        // let mut idx = 0;
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

pub fn preprocessing_step_4(mut img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let replacement_pixel = *img.get_pixel_mut(4, 40);

    // left border
    for y in 0..img.height() {
        for x in 0..55 {
            let pixel = img.get_pixel_mut(x, y);
            let red = pixel[0];
            let green = pixel[1];
            let blue = pixel[2];

            let tmp = vec![red, green, blue];
            let min = itertools::min(tmp.clone()).unwrap();
            let max = itertools::max(tmp.clone()).unwrap();

            if (max - min) <= 10 {
                pixel[0] = replacement_pixel[0];
                pixel[1] = replacement_pixel[1];
                pixel[2] = replacement_pixel[2];
            }
        }
    }

    // right border
    for y in 0..img.height() {
        for x in (img.width() - 55)..img.width() {
            let pixel = img.get_pixel_mut(x, y);
            let red = pixel[0];
            let green = pixel[1];
            let blue = pixel[2];

            let tmp = vec![red, green, blue];
            let min = itertools::min(tmp.clone()).unwrap();
            let max = itertools::max(tmp.clone()).unwrap();

            if (max - min) <= 10 {
                pixel[0] = replacement_pixel[0];
                pixel[1] = replacement_pixel[1];
                pixel[2] = replacement_pixel[2];
            }
        }
    }

    // bottom border
    for y in (img.height() - 40)..img.height() {
        for x in 0..img.width() {
            let pixel = img.get_pixel_mut(x, y);
            let red = pixel[0];
            let green = pixel[1];
            let blue = pixel[2];

            let tmp = vec![red, green, blue];
            let min = itertools::min(tmp.clone()).unwrap();
            let max = itertools::max(tmp.clone()).unwrap();

            if (max - min) <= 10 {
                pixel[0] = replacement_pixel[0];
                pixel[1] = replacement_pixel[1];
                pixel[2] = replacement_pixel[2];
            }
        }
    }

    // top border
    for y in 0..40 {
        for x in 0..img.width() {
            let pixel = img.get_pixel_mut(x, y);
            let red = pixel[0];
            let green = pixel[1];
            let blue = pixel[2];

            let tmp = vec![red, green, blue];
            let min = itertools::min(tmp.clone()).unwrap();
            let max = itertools::max(tmp.clone()).unwrap();

            if (max - min) <= 10 {
                pixel[0] = replacement_pixel[0];
                pixel[1] = replacement_pixel[1];
                pixel[2] = replacement_pixel[2];
            }
        }
    }
    img
}

pub fn preprocessing_step_3(mut img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let replacement_pixel = *img.get_pixel_mut(4, 40);

    for y in 0..img.height() {
        for x_window in (0..img.width()).collect::<Vec<u32>>().windows(19) {
            let mut window_data = Vec::new();

            for x in x_window.iter() {
                let pixel = img.get_pixel(*x, y);
                let red = pixel[0];
                let green = pixel[1];
                let blue = pixel[2];
                // debug!("x: {}, red: {}, green: {}, blue: {}", x, red, green, blue);
                window_data.push((red as f32, green as f32, blue as f32));
            }

            let range = itertools::min(x_window.clone()).unwrap()..itertools::max(x_window.clone()).unwrap();

            let reds: Vec<_> = window_data.iter().map(|e| e.0).collect();
            let reds_mean = statistical::mean(&reds);
            let reds_stddev = statistical::standard_deviation(&reds, None);

            let greens: Vec<_> = window_data.iter().map(|e| e.1).collect();
            let greens_mean = statistical::mean(&greens);
            let greens_stddev = statistical::population_standard_deviation(&greens, None);

            let blues: Vec<_> = window_data.iter().map(|e| e.2).collect();
            let blues_mean = statistical::mean(&blues);
            let blues_stddev = statistical::population_standard_deviation(&blues, None);

            // filter out mostly white pixels
            if reds_mean > 210.0 && greens_mean > 210.0 && blues_mean > 210.0 {
                continue;
            }

            // filter out mostly black pixels
            if reds_mean < 52.0 && greens_mean < 52.0 && blues_mean < 52.0 {
                continue;
            }

            // filter out non-common color range
            if reds_stddev > 9.0 || greens_stddev > 9.0 || blues_stddev > 9.0 {
                continue;
            }

            debug!(
                "range: {:?}, reds_mean: {}, greens_mean: {}, blues_mean: {}, reds_stddev: {}, greens_stddev: {}, blues_stddev: {}",
                range, reds_mean, greens_mean, blues_mean, reds_stddev, greens_stddev, blues_stddev
            );

            for x in x_window.iter() {
                let pixel = img.get_pixel_mut(*x, y);
                pixel[0] = replacement_pixel[0];
                pixel[1] = replacement_pixel[1];
                pixel[2] = replacement_pixel[2];
            }
        }
    }

    img
}

pub fn preprocessing_step_2(img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let mut img = image::imageops::flip_vertical(&img);
    image::imageops::invert(&mut img);
    let mut img = image::imageops::contrast(&mut img, 10.0);

    let mut cutoff = 0;
    for y in 10..(img.height() / 2) {
        let mut row_data = Vec::new();
        for x in 130..(img.width() - 130) {
            let pixel = img.get_pixel(x, y);
            let red = pixel[0];
            let blue = pixel[2];
            let green = pixel[1];

            if red < 110 && green < 110 && blue < 110 {
                continue;
            }

            // debug!("x: {}, y: {}, red: {}, green: {}, blue: {}", x, y, red, green, blue);
            row_data.push(red);
            row_data.push(green);
            row_data.push(blue);
        }
        let avg = row_data.iter().sum::<u8>() as f32 / row_data.len() as f32;
        debug!("y: {}, row_data average: {}", y, avg);
        if !avg.is_nan() && avg > 1f32 {
            cutoff = y;
            break;
        }
    }
    debug!("cutoff: {}", cutoff);

    let img_height = img.height();
    let img_width = img.width();

    let img = image::imageops::crop(&mut img, 0, cutoff, img_width, img_height).to_image();
    let mut img = image::imageops::flip_vertical(&img);

    let mut img = image::imageops::contrast(&mut img, -10.0);
    image::imageops::invert(&mut img);
    img
}

pub fn preprocessing_step_1(mut img: image::DynamicImage) -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let mut left_border = Vec::new();

    'outer: for y in 0..img.height() {
        if !(100u32..900u32).contains(&y) {
            continue 'outer;
        }
        'inner: for x in 0..img.width() {
            if !(10u32..40u32).contains(&x) {
                continue 'inner;
            }

            let pixel = img.get_pixel(x, y);
            let red = pixel[0];
            let green = pixel[1];
            let blue = pixel[2];

            debug!("x: {}, y: {}, red: {}, green: {}, blue: {}", x, y, red, green, blue);
            left_border.push(x);

            if red > 100 && green > 100 && blue > 100 {
                continue 'outer;
            }
        }
    }

    debug!("left_border max: {}", left_border.iter().max().unwrap());
    let from_left = left_border.iter().max().unwrap() + 1;

    let img_height = img.height();
    let img_width = img.width();

    let img = image::imageops::crop(&mut img, from_left, 0, img_width, img_height).to_image();

    let mut img = image::imageops::flip_horizontal(&img);

    let mut right_border = Vec::new();

    'outer: for y in 0..img.height() {
        if !(100u32..900u32).contains(&y) {
            continue;
        }
        'inner: for x in 0..img.width() {
            if !(10u32..40u32).contains(&x) {
                continue;
            }
            let pixel = img.get_pixel(x, y);
            let red = pixel[0];
            let green = pixel[1];
            let blue = pixel[2];

            debug!("x: {}, y: {}, red: {}, green: {}, blue: {}", x, y, red, green, blue);
            right_border.push(x);

            if red > 100 && green > 100 && blue > 100 {
                continue 'outer;
            }
        }
    }

    debug!("right_border max: {}", right_border.iter().max().unwrap());
    let from_right = right_border.iter().max().unwrap() + 1;

    let img = image::imageops::crop(&mut img, from_right, 0, img_width, img_height).to_image();

    let mut img = image::imageops::flip_horizontal(&img);

    let mut top_border = Vec::new();

    for y in 0..img.height() {
        if !(20u32..100u32).contains(&y) {
            continue;
        }
        let pixel = img.get_pixel(0u32, y);
        let red = pixel[0];
        let green = pixel[1];
        let blue = pixel[2];

        debug!("y: {}, red: {}, green: {}, blue: {}", y, red, green, blue);
        top_border.push(y);

        if red > 100 && green > 100 && blue > 100 {
            break;
        }
    }

    debug!("top_border max: {}", top_border.iter().max().unwrap());
    let from_top = top_border.iter().max().unwrap() + 1;

    let img_height = img.height();
    let img_width = img.width();

    let img = image::imageops::crop(&mut img, 0, from_top, img_width, img_height).to_image();

    let mut img = image::imageops::flip_vertical(&img);

    let mut bottom_border = Vec::new();

    for y in 0..img.height() {
        if !(10u32..50u32).contains(&y) {
            continue;
        }

        let pixel = img.get_pixel(0u32, y);
        let red = pixel[0];
        let green = pixel[1];
        let blue = pixel[2];

        debug!("y: {}, red: {}, green: {}, blue: {}", y, red, green, blue);
        bottom_border.push(y);

        if red > 100 && green > 100 && blue > 100 {
            break;
        }
    }

    debug!("bottom_border max: {}", bottom_border.iter().max().unwrap());
    let from_bottom = bottom_border.iter().max().unwrap() + 1;

    let img = image::imageops::crop(&mut img, 0, from_bottom, img_width, img_height).to_image();
    let img = image::imageops::flip_vertical(&img);
    let img = image::imageops::resize(&img, 600, 800, image::imageops::FilterType::CatmullRom);
    img
}
