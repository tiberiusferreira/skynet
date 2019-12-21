use std::fs::File;
use std::path::Path;
pub mod yolo_dataset_loader;

/// Marker trait
trait DataLoader {}

impl<T> DataLoader for T where T: Iterator {}

trait DataTransformer<INPUT, OUTPUT> {
    fn transform<F>(&self, input: INPUT) -> OUTPUT;
}
