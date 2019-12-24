//use std::fs::File;
//use crate::dataset::data_transformers::labelbox_structs::LabelBoxJsonRoot;
//use crate::dataset::common_structs::ImgFilenameWithBboxes;
//use crate::dataset::DataTransformer;
//
//pub fn read_labelbox_exported_json_from_file(filepath: &str) -> Vec<LabelBoxJsonRoot> {
//    let input_file = File::open(filepath).unwrap();
//    serde_json::from_reader(input_file).unwrap()
//}
//
//struct LabelBox2ImgFilenameWithBBoxes;
//
//impl DataTransformer for LabelBox2ImgFilenameWithBBoxes{
//    type InputData = LabelBoxJsonRoot;
//    type OutputData = ImgFilenameWithBboxes;
//
//    fn transform_data(&mut self, data: LabelBoxJsonRoot) -> Self::OutputData {
//        labelbox_struct_to_imgs_with_bb(&data, 1.0, 1.0)
//    }
//}
//
//pub fn labelbox_struct_to_imgs_with_bb(
//    file_labels: &LabelBoxJsonRoot,
//    width_multiplier: f32,
//    height_multiplier: f32,
//) -> ImgFilenameWithBboxes {
//    let img_filename = &file_labels.img_filename;
//    let objects = &file_labels.label.objects;
//    let bboxes: Vec<Bbox> = objects
//        .into_iter()
//        .map(|obj| Bbox {
//            top: (obj.bbox.top as f32 * height_multiplier) as i32,
//            left: (obj.bbox.left as f32 * width_multiplier) as i32,
//            height: ((obj.bbox.height as f32) * height_multiplier) as u32,
//            width: ((obj.bbox.width as f32) * width_multiplier) as u32,
//            prob: 1.0,
//            class: "0".to_string(),
//        })
//        .collect();
//    return ImgFilenameWithBboxes {
//        img_filename: img_filename.clone(),
//        bboxes,
//    };
//}