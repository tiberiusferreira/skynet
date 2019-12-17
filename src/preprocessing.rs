mod structs;

use structs::*;
use serde::{Serialize, Deserialize};
use std::fs::File;
use imageproc::drawing::Blend;
use image::RgbaImage;
use tch::{Kind, Device, IndexOp, Tensor};

const INPUT_FILE: &str = "imgs/export.json";
//const OUTPUT_FILE: &str = "imgs/cleaned.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CleanedImgLabels {
    pub img_filename: String,
    pub bboxes: Vec<Bbox>
}


pub fn to_tensor(bbs: Vec<RawBbox>, grid_size: u32, original_img_size: u32, anchors: (u32, u32)) -> Tensor{
    let (anchor_width, anchor_height) = anchors;
    let tensor = tch::Tensor::zeros(&[6, grid_size as i64, grid_size as i64], (Kind::Float, Device::Cpu));

    for bb in bbs {
        // Tensor layout is CenterX, CenterY, Width, Height, object_prob, class1_prob
        // WARNING! Width, Height is not the real values, they are related by:
        // Real Width = RW //// Real Height = RH
        // RW = exp(Width) * anchor_width // RH = exp(Height) * anchor_width
        // Width = ln(RW/anchor_width)
        let width = (bb.width as f32 / anchor_width as f32).ln();
        let height = (bb.height as f32 / anchor_height as f32).ln();
        // find out in which grid (x and y) the object center is
        let bb_left_center_percentage = (bb.left as f32 / original_img_size as f32); // something like 0.3
        let grids_as_perc = (1. / grid_size as f32);
        let grids_to_left = (bb_left_center_percentage / grids_as_perc).floor() as u32;

        let bb_top_center_percentage = (bb.top as f32 / original_img_size as f32); // something like 0.3
        let grids_to_top = (bb_top_center_percentage / grids_as_perc).floor() as u32;

        tensor.i(0 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
            .copy_(&Tensor::from((bb.left as u32 - grids_to_left * grid_size) as f32));
        tensor.i(1 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
            .copy_(&Tensor::from((bb.top as u32 - grids_to_top * grid_size) as f32));
        tensor.i(2 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
            .copy_(&Tensor::from(width));
        tensor.i(3 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
            .copy_(&Tensor::from(height));
        tensor.i(4 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
            .copy_(&Tensor::from(1.));
        tensor.i(5 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
            .copy_(&Tensor::from(1.));
    }

    tensor
}

pub fn from_tensor(tensor: Tensor, grid_size: u32, original_img_size: u32, anchors: (u32, u32)){
    let grid_size = grid_size as i64;
    let (anchor_width, anchor_height) = anchors;

    // Tensor should be Rank 3: [6, grid_size, grid_size]
    //                     0        1       2      3          4            5
    // Tensor layout is CenterX, CenterY, Width, Height, object_prob, class1_prob
    // select points where object_prob > 0.5
    struct GridPointsWithObjects{
        x: i64,
        y: i64,
        prob: f64,
    }
    let mut grid_points_with_objects: Vec<GridPointsWithObjects> = vec![];
    let objectness_tensor = tensor.i(4 as i64);
    for x in 0i64..grid_size{
        for y in 0i64..grid_size{
            let objectness = objectness_tensor.i(x).i(y).double_value(&[]);
            if objectness > 0.5{
                grid_points_with_objects.push(GridPointsWithObjects{
                    x,
                    y,
                    prob: objectness
                })
            }
        }
    }
    let center_x_tensor = tensor.i(0 as i64);
    let center_y_tensor = tensor.i(1 as i64);
    let width_tensor = tensor.i(2 as i64);
    let height_tensor = tensor.i(3 as i64);
    let class_1_prob = tensor.i(5 as i64);
    for point in grid_points_with_objects{
        let point_x_center = center_x_tensor.i(point.x).i(point.y).double_value(&[]);
        let point_y_center = center_y_tensor.i(point.x).i(point.y).double_value(&[]);
        let point_width = width_tensor.i(point.x).i(point.y).double_value(&[]);
        let point_height = height_tensor.i(point.x).i(point.y).double_value(&[]);
        let point_class_1_prob = class_1_prob.i(point.x).i(point.y).double_value(&[]);
    }
    // WARNING! Width, Height is not the real values, they are related by:
    // Real Width = RW //// Real Height = RH
    // RW = exp(Width) * anchor_width // RH = exp(Height) * anchor_width
    // Width = ln(RW/anchor_width)
//    let rw = (bb.width as f32/anchor_width as f32).ln();
//    let height = (bb.height as f32/anchor_height as f32).ln();
//    // find out in which grid (x and y) the object center is
//    let bb_left_center_percentage = (bb.left as f32/original_img_size as f32); // something like 0.3
//    let grids_as_perc = (1./grid_size as f32);
//    let grids_to_left = (bb_left_center_percentage/grids_as_perc).floor() as u32;
//
//    let bb_top_center_percentage = (bb.top as f32/original_img_size as f32); // something like 0.3
//    let grids_to_top = (bb_top_center_percentage/grids_as_perc).floor() as u32;
//
//    tensor.i(0 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
//        .copy_(&Tensor::from((bb.left as u32 - grids_to_left*grid_size) as f32));
//    tensor.i(1 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
//        .copy_(&Tensor::from((bb.top as u32 - grids_to_top*grid_size) as f32));
//    tensor.i(2 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
//        .copy_(&Tensor::from(width));
//    tensor.i(3 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
//        .copy_(&Tensor::from(height));
//    tensor.i(4 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
//        .copy_(&Tensor::from(1.));
//    tensor.i(5 as i64).i(grids_to_left as i64).i(grids_to_top as i64)
//        .copy_(&Tensor::from(1.));
//
//    tensor
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_tensor() {
        let grid_size = 13;
        let original_img_size = 1220;
        let anchors = (70, 70);
        let bb = RawBbox {
            top: 10,
            left: 600,
            height: 20,
            width: 20
        };

        let bb2 = RawBbox {
            top: 100,
            left: 610,
            height: 20,
            width: 20
        };
        let tensor = to_tensor(vec![bb, bb2], 13, 1220, anchors);
        tensor.print();

        from_tensor(tensor, grid_size, original_img_size, anchors);
    }

}
//fn main() {
//    let labelbox = read_labelbox_from_file(INPUT_FILE);
//    let cleaned_bb = convert_labelbox_to_only_bbs(labelbox);
//
//    let mut img: Blend<RgbaImage> = Blend(image::open("samples/image0000_a.jpg").unwrap().to_rgba());
//
//    let first_img = (&cleaned_bb[0]);
//    for bb in first_img.bboxes.iter(){
//        draw_bb_to_img(&mut img, bb);
//    }
//
//
//    img.0.save("output.jpg").unwrap();
//
//}

pub fn draw_bb_to_img(img: &mut Blend<RgbaImage>, bb: &RawBbox){
    let rec = imageproc::rect::Rect::at(bb.left  as i32, bb.top as i32)
        .of_size(bb.width as u32, bb.height  as u32);

    let color = image::Rgba([255, 0, 0, 90]);

    imageproc::drawing::draw_filled_rect_mut(img, rec, color);
}

pub fn read_labelbox_from_file(filepath: &str) -> Vec<FileLabel>{
    let input_file = File::open(filepath).unwrap();
    serde_json::from_reader(input_file).unwrap()
}

pub fn convert_labelbox_to_only_bbs(file_labels: Vec<FileLabel>) -> Vec<CleanedImgLabels>{
    let mut cleaned_img_labels = vec![];
    for file in file_labels{
        let img_filename = file.img_filename;
        let objects = file.label.objects;
        let bboxes: Vec<Bbox> = objects.into_iter().map(|obj| Bbox{
            top: obj.bbox.top,
            left: obj.bbox.left,
            height: obj.bbox.height,
            width: obj.bbox.width,
            prob: 1.0,
            class: "0".to_string()
        }).collect();
        cleaned_img_labels.push(CleanedImgLabels{
            img_filename,
            bboxes
        });
    }
    cleaned_img_labels
//    let file = File::create(output_file_name).unwrap();
//    serde_json::to_writer_pretty(file, &cleaned_img_labels).unwrap();
}
