use image::{DynamicImage, ImageBuffer};
use tch::vision::image::save;
use tch::{Device, Kind, Tensor};
use crate::yolo_nn::DEVICE;
use itertools::Itertools;

pub mod common_structs;

pub mod data_transformers;
pub mod data_augmenters;
pub mod data_loaders;
pub mod iterator_adapters;

/// Extends the iterator trait to make sure the Dataset has methods display progress
pub trait DataLoader: Iterator {
    /// Returns the next element index, starting from 0
    fn next_element_index(&self) -> usize;
    /// Returns the index of the last element to be loaded
    fn max_elem_index(&self) -> usize;
}
