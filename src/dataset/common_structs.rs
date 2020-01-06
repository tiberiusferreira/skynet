use serde::{Deserialize, Serialize};
/// Frequently used structs in the provided data transformers/loaders/augmenters

/// An image with its Bounding Boxes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImgFilenameWithBboxes {
    pub img_filename: String,
    pub bboxes: Vec<SimpleBbox>,
}

/// A simple Bounding Box with a probability and class associated
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimpleBbox {
    pub top: i32,
    pub left: i32,
    pub height: u32,
    pub width: u32,
    pub prob: f64,
    pub class: u32,
}
