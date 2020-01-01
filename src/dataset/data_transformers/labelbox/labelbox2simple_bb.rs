use crate::dataset::common_structs::{ImgFilenameWithBboxes, SimpleBbox};
use std::fs::File;
use crate::dataset::data_transformers::labelbox::LabelBoxImageJson;

/// Takes the file exported by LabelBox and converts into a rust struct
pub fn labelbox_vec_from_exported_json_file(filepath: &str) -> Vec<LabelBoxImageJson> {
    let input_file = File::open(filepath).expect("Could not open input file");
    serde_json::from_reader(input_file).expect("Input file did not contain a valid LabelBox json")
}

/// Takes a single LabelBoxImageJson and returns a ImgFilenameWithBboxes with its Bounding Box
/// coordinates and sizes multiplied by width_multiplier and height_multiplier.
/// For example, if you scale the image width by 0.7, set width_multiplier to 0.7 and the Bounding
/// Box should stay in the right place in the scaled image.
pub fn labelbox_struct_to_img_filename_with_bboxes(
    file_labels: &LabelBoxImageJson,
    width_multiplier: f32,
    height_multiplier: f32,
) -> ImgFilenameWithBboxes {
    let img_filename = &file_labels.img_filename;
    let objects = &file_labels.label.objects;
    let bboxes: Vec<SimpleBbox> = objects
        .into_iter()
        .map(|obj| SimpleBbox {
            top: (obj.bbox.top as f32 * height_multiplier) as i32,
            left: (obj.bbox.left as f32 * width_multiplier) as i32,
            height: ((obj.bbox.height as f32) * height_multiplier) as u32,
            width: ((obj.bbox.width as f32) * width_multiplier) as u32,
            prob: 1.0,
            class: obj.value.clone(),
        })
        .collect();
    return ImgFilenameWithBboxes {
        img_filename: img_filename.clone(),
        bboxes,
    };
}
