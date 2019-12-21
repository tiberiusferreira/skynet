use crate::preprocessing::structs::{Bbox, CleanedImgLabels, FileLabel};
use image::RgbaImage;
use imageproc::drawing::Blend;
use std::fs::File;
use tch::Tensor;
use tch::{Device, IndexOp, Kind, R3TensorGeneric};
//const OUTPUT_FILE: &str = "imgs/cleaned.json";

pub fn bbs_to_tensor(
    bbs: &Vec<Bbox>,
    grid_size: u32,
    original_img_size: u32,
    anchors: (u32, u32),
) -> R3TensorGeneric {
    let (anchor_width, anchor_height) = anchors;
    let tensor = tch::Tensor::zeros(
        &[6, grid_size as i64, grid_size as i64],
        (Kind::Float, Device::Cpu),
    );
    let grid_cell_size_pixels = original_img_size / grid_size;
    for bb in bbs {
        // Tensor layout is CenterX, CenterY, Width, Height, object_prob, class1_prob
        // WARNING! Width, Height is not the real values, they are related by:
        // Real Width = RW //// Real Height = RH
        // RW = exp(Width) * anchor_width // RH = exp(Height) * anchor_width
        // Width = ln(RW/anchor_width)
        let width = (bb.width as f32 / anchor_width as f32).ln();
        let height = (bb.height as f32 / anchor_height as f32).ln();
        // find out in which grid (x and y) the object center is
        let bb_left_center_percentage =
            (bb.left + (bb.width / 2) as i32) as f32 / original_img_size as f32; // something like 0.3
        let grid_cell_as_perc = 1. / grid_size as f32;
        let grids_cells_to_left = (bb_left_center_percentage / grid_cell_as_perc).floor() as u32;

        let bb_top_center_percentage =
            (bb.top + (bb.height / 2) as i32) as f32 / original_img_size as f32; // something like 0.3
        let grids_cells_to_top = (bb_top_center_percentage / grid_cell_as_perc).floor() as u32;

        tensor
            .i(0 as i64)
            .i(grids_cells_to_left as i64)
            .i(grids_cells_to_top as i64)
            .copy_(&Tensor::from(
                ((bb.left + bb.width as i32 / 2) as u32
                    - grids_cells_to_left * grid_cell_size_pixels) as f32
                    / grid_cell_size_pixels as f32,
            ));
        tensor
            .i(1 as i64)
            .i(grids_cells_to_left as i64)
            .i(grids_cells_to_top as i64)
            .copy_(&Tensor::from(
                ((bb.top + bb.height as i32 / 2) as u32
                    - grids_cells_to_top * grid_cell_size_pixels) as f32
                    / grid_cell_size_pixels as f32,
            ));
        tensor
            .i(2 as i64)
            .i(grids_cells_to_left as i64)
            .i(grids_cells_to_top as i64)
            .copy_(&Tensor::from(width));
        tensor
            .i(3 as i64)
            .i(grids_cells_to_left as i64)
            .i(grids_cells_to_top as i64)
            .copy_(&Tensor::from(height));
        tensor
            .i(4 as i64)
            .i(grids_cells_to_left as i64)
            .i(grids_cells_to_top as i64)
            .copy_(&Tensor::from(bb.prob));
        tensor
            .i(5 as i64)
            .i(grids_cells_to_left as i64)
            .i(grids_cells_to_top as i64)
            .copy_(&Tensor::from(bb.prob));
    }

    tensor.into()
}

pub struct GridPointsWithObjects {
    pub x: i64,
    pub y: i64,
    pub prob: f64,
}
pub fn objects_mask_tensor_from_target_tensor(
    tensor: R3TensorGeneric,
    grid_size: u32,
) -> Vec<GridPointsWithObjects> {
    let tensor = tensor.tensor;
    let grid_size = grid_size as i64;
    // Tensor should be Rank 3: [6, grid_size, grid_size]
    //                     0        1       2      3          4            5
    // Tensor layout is CenterX, CenterY, Width, Height, object_prob, class1_prob
    // select points where object_prob > 0.5

    let mut grid_points_with_objects: Vec<GridPointsWithObjects> = vec![];
    let objectness_tensor = tensor.i(4 as i64);
    for x in 0i64..grid_size {
        for y in 0i64..grid_size {
            let objectness = objectness_tensor.i(x).i(y).double_value(&[]);
            if objectness > 0.5 {
                grid_points_with_objects.push(GridPointsWithObjects {
                    x,
                    y,
                    prob: objectness,
                })
            }
        }
    }
    grid_points_with_objects
}

pub fn flip_bb_horizontally(bbs: &Vec<Bbox>, img_width: u32) -> Vec<Bbox> {
    let mut flipped = vec![];
    // we flip it in respect to the mid vertical line
    let half_width = (img_width / 2) as i32;
    for bbox in bbs {
        let original_over_half_width_left = bbox.left - half_width;
        let new_left = half_width - original_over_half_width_left - bbox.width as i32;
        flipped.push(Bbox {
            left: new_left,
            ..bbox.clone()
        })
    }
    flipped
}

pub fn flip_bb_vertically(bbs: &Vec<Bbox>, img_height: u32) -> Vec<Bbox> {
    let mut flipped = vec![];
    // we flip it in respect to the mid vertical line
    let half_height = (img_height / 2) as i32;
    for bbox in bbs {
        let original_over_half_height_top = bbox.top - half_height;
        let new_top = half_height - original_over_half_height_top - bbox.height as i32;
        flipped.push(Bbox {
            top: new_top,
            ..bbox.clone()
        })
    }
    flipped
}

pub fn bbs_from_tensor(
    tensor: R3TensorGeneric,
    grid_size: u32,
    original_img_size: u32,
    anchors: (u32, u32),
) -> Vec<Bbox> {
    let tensor = tensor.tensor;
    let grid_size = grid_size as i64;
    let (anchor_width, anchor_height) = anchors;
    let grid_cell_size_pixels: i64 = (original_img_size / grid_size as u32) as i64;
    // Tensor should be Rank 3: [6, grid_size, grid_size]
    //                     0        1       2      3          4            5
    // Tensor layout is CenterX, CenterY, Width, Height, object_prob, class1_prob
    // select points where object_prob > 0.5
    struct GridPointsWithObjects {
        x: i64,
        y: i64,
        prob: f64,
    }
    let mut grid_points_with_objects: Vec<GridPointsWithObjects> = vec![];
    let objectness_tensor = tensor.i(4 as i64);
    for x in 0i64..grid_size {
        for y in 0i64..grid_size {
            let objectness = objectness_tensor.i(x).i(y).double_value(&[]);
            if objectness > 0.1 {
                grid_points_with_objects.push(GridPointsWithObjects {
                    x,
                    y,
                    prob: objectness,
                })
            }
        }
    }
    let center_x_tensor = tensor.i(0 as i64);
    let center_y_tensor = tensor.i(1 as i64);
    let width_tensor = tensor.i(2 as i64);
    let height_tensor = tensor.i(3 as i64);
    let class_1_prob = tensor.i(5 as i64);
    let mut bbs = vec![];
    for point in grid_points_with_objects {
        let point_x_center = point.x * grid_cell_size_pixels
            + (center_x_tensor.i(point.x).i(point.y).double_value(&[])
                * grid_cell_size_pixels as f64) as i64;
        let point_y_center = point.y * grid_cell_size_pixels
            + (center_y_tensor.i(point.x).i(point.y).double_value(&[])
                * grid_cell_size_pixels as f64) as i64;
        let point_width = ((anchor_width as f64)
            * (width_tensor.i(point.x).i(point.y).double_value(&[])).exp())
            as u32;
        let point_height = ((anchor_height as f64)
            * height_tensor.i(point.x).i(point.y).double_value(&[]).exp())
            as u32;
        let _point_class_1_prob = class_1_prob.i(point.x).i(point.y).double_value(&[]);
        let top = (point_y_center - point_height as i64 / 2) as i32;
        let left = (point_x_center - point_width as i64 / 2) as i32;
        //        let left = point_x_center
        bbs.push(Bbox {
            top,
            left,
            height: point_height,
            width: point_width,
            prob: point.prob,
            class: "".to_string(),
        });
    }
    bbs
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, FilterType, GenericImageView};
    use rand::prelude::SliceRandom;
    use rand::thread_rng;
    use std::fs::File;

    #[test]
    fn test_to_tensor() {
        let grid_size = 13;
        let original_img_size = 416;
        let anchors = (70, 70);
        let bb = Bbox {
            top: 10,
            left: 200,
            height: 25,
            width: 30,
            prob: 1.0,
            class: "".to_string(),
        };

        let bb2 = Bbox {
            top: 100,
            left: 250,
            height: 36,
            width: 40,
            prob: 1.0,
            class: "".to_string(),
        };
        let expected = vec![bb, bb2];
        let tensor = bbs_to_tensor(&expected, grid_size, original_img_size, anchors);
        tensor.tensor.print();

        let bbs = bbs_from_tensor(tensor, grid_size, original_img_size, anchors);
        println!("{:#?}", bbs);
        assert_eq!(expected, bbs);
    }
    #[test]
    fn convert_labels() {
        const INPUT_FILE_ROOT: &str = "dataset/raw_samples";
        const INPUT_FILE: &str = "dataset/raw_samples/labelbox.json";

        let target_size = 416;
        let train_output_path = "dataset/train";
        let test_output_path = "dataset/test";
        let labelboxs = read_labelbox_from_file(INPUT_FILE);
        let mut img_n_labels: Vec<(CleanedImgLabels, DynamicImage)> = vec![];
        for label in labelboxs {
            let img_path = format!("{}/{}", INPUT_FILE_ROOT, label.img_filename.clone());
            let img = image::open(img_path).unwrap();
            let (ori_width, ori_height) = (img.width(), img.height());
            let width_ratio = target_size as f32 / ori_width as f32;
            let height_ratio = target_size as f32 / ori_height as f32;
            let resized = img.resize_exact(target_size, target_size, FilterType::CatmullRom);
            let bb = convert_labelbox_to_only_bbs(&label, width_ratio, height_ratio);
            img_n_labels.push((bb, resized));
        }

        img_n_labels.shuffle(&mut thread_rng());
        let (test, train) = img_n_labels.split_at(5);
        let mut test_labels = vec![];
        for (label, img) in test {
            test_labels.push(label);
            img.save(format!(
                "{}/{}",
                test_output_path,
                label.img_filename.clone()
            ));
        }
        let test_output_label_file =
            File::create(format!("{}/labels.json", test_output_path)).unwrap();
        serde_json::to_writer_pretty(test_output_label_file, &test_labels).unwrap();

        let mut train_labels = vec![];
        for (label, img) in train {
            train_labels.push(label);
            img.save(format!(
                "{}/{}",
                train_output_path,
                label.img_filename.clone()
            ));
        }
        let train_output_label_file =
            File::create(format!("{}/labels.json", train_output_path)).unwrap();
        serde_json::to_writer_pretty(train_output_label_file, &train_labels).unwrap();
    }
    #[test]
    fn test_horizontal_flip() {
        let img = image::open("code_test_data/test_img.jpg").unwrap();
        let file =
            File::open("code_test_data/labels.json").expect("Json data file for Yolo not found!");
        let mut labels: Vec<CleanedImgLabels> =
            serde_json::from_reader(file).expect("Invalid json data file");
        let bbs = &labels[0].bboxes;
        let mut blend = Blend(img.to_rgba());
        for single_bb in bbs {
            draw_bb_to_img(&mut blend, single_bb);
        }
        blend.0.save("code_test_data/hflip_test_original.jpg");

        let flipped_bbs = flip_bb_horizontally(bbs, img.width());
        let img = image::open("code_test_data/test_img.jpg").unwrap();
        let flipped_img = img.fliph();
        let mut blend = Blend(flipped_img.to_rgba());
        for single_bb in &flipped_bbs {
            draw_bb_to_img(&mut blend, single_bb);
        }
        blend.0.save("code_test_data/hflip_test.jpg");
    }

    #[test]
    fn test_vertical_flip() {
        let img = image::open("code_test_data/test_img.jpg").unwrap();
        let file =
            File::open("code_test_data/labels.json").expect("Json data file for Yolo not found!");
        let mut labels: Vec<CleanedImgLabels> =
            serde_json::from_reader(file).expect("Invalid json data file");
        let bbs = &labels[0].bboxes;
        let mut blend = Blend(img.to_rgba());
        for single_bb in bbs {
            draw_bb_to_img(&mut blend, single_bb);
        }
        blend.0.save("code_test_data/vflip_test_original.jpg");

        let flipped_bbs = flip_bb_vertically(bbs, img.height());
        let img = image::open("code_test_data/test_img.jpg").unwrap();
        let flipped_img = img.flipv();
        let mut blend = Blend(flipped_img.to_rgba());
        for single_bb in &flipped_bbs {
            draw_bb_to_img(&mut blend, single_bb);
        }
        blend.0.save("code_test_data/vflip_test.jpg");
    }
}

pub fn draw_bb_to_img(img: &mut Blend<RgbaImage>, bb: &Bbox) {
    let rec = imageproc::rect::Rect::at(bb.left as i32, bb.top as i32)
        .of_size(bb.width as u32, bb.height as u32);

    let color = image::Rgba([255, 0, 0, 90]);

    imageproc::drawing::draw_filled_rect_mut(img, rec, color);
}

pub fn read_labelbox_from_file(filepath: &str) -> Vec<FileLabel> {
    let input_file = File::open(filepath).unwrap();
    serde_json::from_reader(input_file).unwrap()
}

pub fn convert_labelbox_to_only_bbs(
    file_labels: &FileLabel,
    width_multiplier: f32,
    height_multiplier: f32,
) -> CleanedImgLabels {
    let img_filename = &file_labels.img_filename;
    let objects = &file_labels.label.objects;
    let bboxes: Vec<Bbox> = objects
        .into_iter()
        .map(|obj| Bbox {
            top: (obj.bbox.top as f32 * height_multiplier) as i32,
            left: (obj.bbox.left as f32 * width_multiplier) as i32,
            height: ((obj.bbox.height as f32) * height_multiplier) as u32,
            width: ((obj.bbox.width as f32) * width_multiplier) as u32,
            prob: 1.0,
            class: "0".to_string(),
        })
        .collect();
    return CleanedImgLabels {
        img_filename: img_filename.clone(),
        bboxes,
    };
}
