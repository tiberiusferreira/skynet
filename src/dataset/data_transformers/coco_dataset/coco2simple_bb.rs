use std::fs::File;
use super::coco_dataset_struct::*;

/// Takes the file exported by LabelBox and converts into a rust struct
pub fn read_annotations_file(filepath: &str) -> CocoDatasetJson {
    let input_file = File::open(filepath).expect("Could not open input file");
    serde_json::from_reader(input_file).expect("Input file did not contain a valid LabelBox json")
}