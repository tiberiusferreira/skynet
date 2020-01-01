use crate::dataset::common_structs::{ImgFilenameWithBboxes, SimpleBbox};
use crate::dataset::DataLoader;
use image::{DynamicImage};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Yolo
pub struct YoloDataLoader {
    data_dir: PathBuf,
    imgs_and_labels_data: Vec<ImgFilenameWithBboxes>,
    max_elem_index: usize,
    next_element_index: usize,
}

impl YoloDataLoader {
    pub fn new(data_json_file_path: &str) -> YoloDataLoader {
        let parent = Path::new(data_json_file_path)
            .parent()
            .expect("Error reading folder of yolo json data file")
            .to_owned();
        let file = File::open(data_json_file_path).expect("Json data file for Yolo not found!");
        let labels: Vec<ImgFilenameWithBboxes> =
            serde_json::from_reader(file).expect("Invalid json data file");
        println!("Loaded ImgFilenameWithBboxes");
        YoloDataLoader {
            data_dir: parent,
            max_elem_index: labels.len(),
            imgs_and_labels_data: labels,
            next_element_index: 0,
        }
    }
}

impl Iterator for YoloDataLoader {
    type Item = (DynamicImage, Vec<SimpleBbox>);
    fn next(&mut self) -> Option<Self::Item> {
        let next_data = self.imgs_and_labels_data.pop()?;
        self.next_element_index += 1;
        let mut data_dir = self.data_dir.clone();
        println!("Loading {}", next_data.img_filename);
        data_dir.push(next_data.img_filename);
        let img = image::open(data_dir).expect("Error loading next img");
        Some((img, next_data.bboxes))
    }
}

impl DataLoader for YoloDataLoader {
    fn next_element_index(&self) -> usize {
        self.next_element_index
    }

    fn max_elem_index(&self) -> usize {
        self.max_elem_index
    }
}
