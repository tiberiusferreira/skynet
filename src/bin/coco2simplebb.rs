use image::{open, DynamicImage, FilterType, GenericImageView};
use imageproc::drawing::Blend;
use rayon::prelude::*;
use skynet::dataset::common_structs::{ImgFilenameWithBboxes, SimpleBbox};
use skynet::dataset::data_transformers::coco_dataset::{read_annotations_file, Annotation};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::Write;

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub struct ImgFileData {
    pub id: String,
    pub filename: String,
    pub path: String,
}

// None if width or height = 0 or < 10
fn annotation2simplebb(annotation: &Annotation) -> Option<SimpleBbox> {
    let out = SimpleBbox {
        top: *annotation.bbox.get(1).unwrap() as i32,
        left: *annotation.bbox.get(0).unwrap() as i32,
        height: *annotation.bbox.get(3).unwrap() as u32,
        width: *annotation.bbox.get(2).unwrap() as u32,
        prob: 1.0,
        class: annotation.category_id as u32,
    };
    if out.height < 10 || out.width < 10 {
        return None;
    } else {
        return Some(out);
    }
}

pub fn hashmap2img_filename_with_bboxes(
    hashmap: HashMap<ImgFileData, Vec<SimpleBbox>>,
) -> Vec<ImgFilenameWithBboxes> {
    hashmap
        .into_iter()
        .map(|(img_file_data, bb_vec)| ImgFilenameWithBboxes {
            img_filename: img_file_data.filename,
            bboxes: bb_vec,
        })
        .collect()
}
fn main() {
    let coco = read_annotations_file("coco/annotations_trainval2017/instances_train2017.json");
    println!("Done reading json");
    io::stdout().flush().unwrap();
    let only_people = coco.annotations.iter().filter(|e| e.category_id == 1);
    // Hashmap between an image_id and its BB
    let mut img_and_bb: HashMap<ImgFileData, Vec<SimpleBbox>> = HashMap::new();
    only_people.for_each(|annotation| {
        let img_file_data = img_id_to_img_file_data(annotation.image_id);
        let maybe_simple_bbox = annotation2simplebb(annotation);
        if let Some(simple_bbox) = maybe_simple_bbox {
            let mut bbox_vec = vec![simple_bbox];
            if let Some(existing) = img_and_bb.get(&img_file_data) {
                bbox_vec.extend_from_slice(existing.as_slice())
            }
            img_and_bb.insert(img_file_data, bbox_vec);
        }
    });
    let mut img_filename_with_bboxes_vec = hashmap2img_filename_with_bboxes(img_and_bb);
    println!("Resizing imgs and bboxes");
    io::stdout().flush().unwrap();
    img_filename_with_bboxes_vec
        .par_iter_mut()
        .for_each(|img_and_bbs| {
            let img_path = &format!("coco/train2017/{}", img_and_bbs.img_filename);
            let img = match open(img_path) {
                Ok(img) => img,
                Err(e) => {
                    println!("error opening img {}", img_path);
                    return;
                }
            };
            let (resized, width_ratio, height_ratio) = resize_img(img, 416);
            resized
                .save(&format!("coco/train/{}", img_and_bbs.img_filename))
                .expect("error saving img");
            let resized_bb = resize_bb(&img_and_bbs.bboxes, width_ratio, height_ratio);
            img_and_bbs.bboxes = resized_bb;
        });
    let file = File::create("coco/train/labels.json").unwrap();
    serde_json::to_writer_pretty(file, &img_filename_with_bboxes_vec).unwrap();
}

mod test {
    use futures::io::Error;
    use skynet::dataset::common_structs::ImgFilenameWithBboxes;
    use std::fs::File;

    #[test]
    fn draw_boxes() {
        let file = File::open("coco/test/labels.json").unwrap();
        let labels: Vec<ImgFilenameWithBboxes> =
            serde_json::from_reader(file).expect("Invalid json data file");
        for label in labels.iter().take(200) {
            let img_path = format!("coco/test/{}", label.img_filename);
            let out_path = format!("coco/test_bb_drawn/{}", label.img_filename);
            draw_bb_to_img_from_file(&img_path, &out_path, &label.bboxes);
        }
    }

    #[test]
    fn check_files() {
        let file = File::open("coco/train/labels.json").unwrap();
        let labels: Vec<ImgFilenameWithBboxes> =
            serde_json::from_reader(file).expect("Invalid json data file");
        //        let fil: Vec<&ImgFilenameWithBboxes>= labels.iter()
        //            .filter(|e| {
        //                if e.img_filename=="000000320612.jpg"{
        //                    println!("Found it!");
        //                    return false;
        //                }else{
        //                    return true;
        //                }
        //            }).collect();
        //        let new = File::create("fil_label.json").unwrap();
        //        serde_json::to_writer_pretty(new, &fil).unwrap();
        for label in labels.iter() {
            let path = format!("coco/train/{}", &label.img_filename);
            match File::open(path) {
                Ok(_) => {}
                Err(err) => {
                    println!("File {} not found! {}", label.img_filename, err);
                }
            }
        }
    }
}

fn resize_bb(
    bboxes: &Vec<SimpleBbox>,
    width_multiplier: f32,
    height_multiplier: f32,
) -> Vec<SimpleBbox> {
    bboxes
        .into_iter()
        .map(|obj| SimpleBbox {
            top: (obj.top as f32 * height_multiplier) as i32,
            left: (obj.left as f32 * width_multiplier) as i32,
            height: ((obj.height as f32) * height_multiplier) as u32,
            width: ((obj.width as f32) * width_multiplier) as u32,
            prob: obj.prob,
            class: obj.class.clone(),
        })
        .collect()
}

fn resize_img(img: DynamicImage, target_size: u32) -> (DynamicImage, f32, f32) {
    let (ori_width, ori_height) = (img.width(), img.height());
    let width_ratio = target_size as f32 / ori_width as f32;
    let height_ratio = target_size as f32 / ori_height as f32;
    let resized = img.resize_exact(target_size, target_size, FilterType::Nearest);
    (resized, width_ratio, height_ratio)
}

fn img_id_to_img_file_data(img_id: i64) -> ImgFileData {
    let path = format!("coco/train2017/{:012}.jpg", img_id);
    let filename = format!("{:012}.jpg", img_id);
    ImgFileData {
        id: img_id.to_string(),
        filename,
        path,
    }
}
