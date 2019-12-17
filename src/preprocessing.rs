pub mod structs;

use image::RgbaImage;
use imageproc::drawing::Blend;
use serde::{Deserialize, Serialize};
use std::fs::File;
use structs::*;
use tch::{Device, IndexOp, Kind, R3TensorGeneric, R4TensorGeneric, Tensor};

const INPUT_FILE: &str = "samples/export.json";
//const OUTPUT_FILE: &str = "imgs/cleaned.json";


pub fn to_tensor(
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
        let bb_left_center_percentage = (bb.left + (bb.width/2) as i32) as f32 / original_img_size as f32; // something like 0.3
        let grid_cell_as_perc = 1. / grid_size as f32;
        let grids_cells_to_left = (bb_left_center_percentage / grid_cell_as_perc).floor() as u32;

        let bb_top_center_percentage = (bb.top + (bb.height/2) as i32) as f32 / original_img_size as f32; // something like 0.3
        let grids_cells_to_top = (bb_top_center_percentage / grid_cell_as_perc).floor() as u32;

        tensor
            .i(0 as i64)
            .i(grids_cells_to_left as i64)
            .i(grids_cells_to_top as i64)
            .copy_(&Tensor::from(
                ((bb.left + bb.width as i32/2) as u32 - grids_cells_to_left * grid_cell_size_pixels)
                    as f32
                    / grid_cell_size_pixels as f32,
            ));
        tensor
            .i(1 as i64)
            .i(grids_cells_to_left as i64)
            .i(grids_cells_to_top as i64)
            .copy_(&Tensor::from(
                ((bb.top + bb.height as i32/2) as u32 - grids_cells_to_top * grid_cell_size_pixels) as f32
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
            .copy_(&Tensor::from(1.));
        tensor
            .i(5 as i64)
            .i(grids_cells_to_left as i64)
            .i(grids_cells_to_top as i64)
            .copy_(&Tensor::from(1.));
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
    original_img_size: u32,
    anchors: (u32, u32),
) -> Vec<GridPointsWithObjects> {
    let tensor = tensor.tensor;
    let grid_size = grid_size as i64;
    let (anchor_width, anchor_height) = anchors;
    let grid_cell_size_pixels: i64 = (original_img_size / grid_size as u32) as i64;
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
    //    let x_tensor = Tensor::of_slice(x_slice.as_slice());
    //    let y_tensor = Tensor::of_slice(y_slice.as_slice());
    //    x_tensor.print();
    //    (x_tensor, y_tensor)
}
pub fn from_tensor(
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
            if objectness > 0.5 {
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
            + (center_x_tensor.i(point.x).i(point.y).double_value(&[])*grid_cell_size_pixels as f64) as i64;
        let point_y_center = point.y * grid_cell_size_pixels
            + (center_y_tensor.i(point.x).i(point.y).double_value(&[])*grid_cell_size_pixels as f64) as i64;
        let point_width = ((anchor_width as f64)
            * (width_tensor.i(point.x).i(point.y).double_value(&[])).exp())
            as u32;
        let point_height = ((anchor_height as f64)
            * height_tensor.i(point.x).i(point.y).double_value(&[]).exp())
            as u32;
        let point_class_1_prob = class_1_prob.i(point.x).i(point.y).double_value(&[]);
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
    use image::imageops::resize;
    use image::{FilterType, GenericImageView};
    use std::env::var;

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
            prob: 0.0,
            class: "".to_string(),
        };

        let bb2 = Bbox {
            top: 100,
            left: 250,
            height: 36,
            width: 40,
            prob: 0.0,
            class: "".to_string(),
        };
        let tensor = to_tensor(vec![bb, bb2], grid_size, original_img_size, anchors);
        tensor.tensor.print();

        let bbs = from_tensor(tensor, grid_size, original_img_size, anchors);
        println!("{:#?}", bbs);
    }
    #[test]
    fn convert_labels() {
        let target_size = 416;
        let output_path = "processed_imgs";
        let labelboxs = read_labelbox_from_file(INPUT_FILE);
        let mut labels: Vec<CleanedImgLabels> = vec![];
        for label in labelboxs {
            let img_path = format!("samples/{}", label.img_filename.clone());
            let img = image::open(img_path).unwrap();
            let (ori_width, ori_height) = (img.width(), img.height());
            let width_ratio = (target_size as f32 / ori_width as f32);
            let height_ratio = (target_size as f32 / ori_height as f32);
            let resized = img.resize_exact(target_size, target_size, FilterType::CatmullRom);
            resized
                .save(format!("{}/{}", output_path, label.img_filename.clone()))
                .unwrap();

            let bb = convert_labelbox_to_only_bbs(&label, width_ratio, height_ratio);
            labels.push(bb);


        }
        let output_label_file =
            File::create(format!("{}/labels.json", output_path)).unwrap();
        serde_json::to_writer_pretty(output_label_file, &labels);
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
