use super::*;
use crate::preprocessing::bbox_conversion::{flip_bb_horizontally, flip_bb_vertically};
use crate::preprocessing::structs::{Bbox, CleanedImgLabels};
use failure::ResultExt;
use image::{DynamicImage, GenericImageView};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::path::PathBuf;

/// Yolo
pub struct YoloDataLoader {
    data_dir: PathBuf,
    imgs_and_labels_data: Vec<CleanedImgLabels>,
    buffer: Vec<(DynamicImage, Vec<Bbox>)>,
    buffer_size: u32,
    include_vertical_n_horizontal_flips: bool,
    shuffle: bool,
}

impl YoloDataLoader {
    pub fn new(
        data_json_file_path: &str,
        shuffle: bool,
        buffer_size: u32,
        include_vertical_n_horizontal_flips: bool,
    ) -> YoloDataLoader {
        let parent = Path::new(data_json_file_path)
            .parent()
            .expect("Error reading folder of yolo json data file")
            .to_owned();
        let file = File::open(data_json_file_path).expect("Json data file for Yolo not found!");
        let mut labels: Vec<CleanedImgLabels> =
            serde_json::from_reader(file).expect("Invalid json data file");

        if shuffle {
            labels.shuffle(&mut thread_rng());
        }

        YoloDataLoader {
            data_dir: parent,
            imgs_and_labels_data: labels,
            buffer: vec![],
            buffer_size,
            include_vertical_n_horizontal_flips,
            shuffle,
        }
    }
    pub fn len(&self) -> usize{
        if self.include_vertical_n_horizontal_flips{
            return self.imgs_and_labels_data.len()*3;
        }else{
            return self.imgs_and_labels_data.len();
        }
    }
    fn fill_buffer(&mut self) -> Result<(), ()> {
        let next_data = self.imgs_and_labels_data.pop().ok_or(())?;
        let mut data_dir = self.data_dir.clone();
        data_dir.push(next_data.img_filename);
        let mut new_data = vec![];
        let img = image::open(data_dir).expect("Error loading next img");
        let bbs = next_data.bboxes;
        new_data.push((img.clone(), bbs.clone()));
        if self.include_vertical_n_horizontal_flips {
            let img_flipped_v = img.flipv();
            let flipped_v_bb = flip_bb_vertically(&bbs, img.height());
            let img_flipped_h = img.fliph();
            let flipped_h_bb = flip_bb_horizontally(&bbs, img.width());
            new_data.push((img_flipped_v, flipped_v_bb));
            new_data.push((img_flipped_h, flipped_h_bb));
        }
        self.buffer.extend(new_data);
        Ok(())
    }
}

impl Iterator for YoloDataLoader {
    type Item = (DynamicImage, Vec<Bbox>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            while self.buffer.len() < self.buffer_size as usize {
                if self.fill_buffer().is_err() {
                    println!("All remaining images in buffer!");
                    break;
                }
            }
            if self.shuffle {
                println!("Shuffling!");
                self.buffer.shuffle(&mut thread_rng());
            }
        }

        self.buffer.pop()
    }
}

//impl<'a> DataLoader<'a> for YoloDataLoader{
//    type Data = String;
//
//    fn iter(&'a self) -> &'a [Self::Data] {
//        let a = self.imgs_and_labels_data.iter().map(|single|{
//           single.img_filename
//        });
//
//    }
//}
