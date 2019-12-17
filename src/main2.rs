// The pre-trained weights can be downloaded here:
//   https://github.com/LaurentMazare/ocaml-torch/releases/download/v0.1-unstable/yolo-v3.ot
#![feature(const_generics)]
#[macro_use]
extern crate failure;
extern crate tch;

mod yolo;

use crate::yolo::coco_classes::NAMES;
use crate::yolo::network;
use crate::yolo::network::NetworkOutput;
use crate::yolo::postprocessing::report;
use rand::Rng;
use std::convert::TryFrom;
use std::path::Path;
use std::process::exit;
use tch::nn::ModuleT;
use tch::vision::image;
use tch::vision::imagenet::save_image;
use tch::{Device, IndexOp, Kind, R3Tensor, R3TensorGeneric, R4Tensor, Tensor, R4TensorGeneric, R2Tensor};

const CONFIG_NAME: &'static str = "yolo-v3.cfg";

//const CONFIG_NAME: &'static str = "yolo-v3.cfg";
//const CONFIDENCE_THRESHOLD: f64 = 0.5;
//const NMS_THRESHOLD: f64 = 0.4;
//
//#[derive(Debug, Clone, Copy)]
//struct Bbox {
//    xmin: f64,
//    ymin: f64,
//    xmax: f64,
//    ymax: f64,
//    confidence: f64,
//    class_index: usize,
//    class_confidence: f64,
//}
//
//// Intersection over union of two bounding boxes.
//fn iou(b1: &Bbox, b2: &Bbox) -> f64 {
//    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
//    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
//    let i_xmin = b1.xmin.max(b2.xmin);
//    let i_xmax = b1.xmax.min(b2.xmax);
//    let i_ymin = b1.ymin.max(b2.ymin);
//    let i_ymax = b1.ymax.min(b2.ymax);
//    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
//    i_area / (b1_area + b2_area - i_area)
//}
//
//// Assumes x1 <= x2 and y1 <= y2
//pub fn draw_rect(t: &mut Tensor, x1: i64, x2: i64, y1: i64, y2: i64) {
//    let color = Tensor::of_slice(&[0., 0., 1.]).view([3, 1, 1]);
//    t.narrow(2, x1, x2 - x1)
//        .narrow(1, y1, y2 - y1)
//        .copy_(&color)
//}

// Assumes x1 <= x2 and y1 <= y2
// expects tensor or size [3, width, height]
pub fn draw_rect_unnorm(
    t: &mut Tensor,
    xmin: i64,
    xmax: i64,
    ymin: i64,
    ymax: i64,
    color: &Tensor,
) {
    //    println!("Integer: {}", rng.gen_range(0, 255));
    let (width, height) = (t.size()[1], t.size()[2]);
    let xmin = xmin.max(0).min(width);
    let xmax = xmax.max(0).min(width);
    let ymin = ymin.max(0).min(height);
    let ymax = ymax.max(0).min(height);
    t.narrow(1, xmin, xmax - xmin)
        .narrow(2, ymin, ymax - ymin)
        .copy_(&color)
}

// expects tensor or size [3, width, height]
pub fn draw_bb_unnorm(img: &mut Tensor, xmin: i64, xmax: i64, ymin: i64, ymax: i64) {
    let mut rng = rand::thread_rng();
    let color = Tensor::of_slice(&[
        rng.gen_range(0, 255),
        rng.gen_range(0, 255),
        rng.gen_range(0, 255),
    ])
        .view([3, 1, 1]);
    draw_rect_unnorm(img, xmin, xmax, ymin, ymax.min(ymin + 2), &color);
    draw_rect_unnorm(img, xmin, xmax, ymin.max(ymax - 2), ymax, &color);
    draw_rect_unnorm(img, xmin, xmax.min(xmin + 2), ymin, ymax, &color);
    draw_rect_unnorm(img, xmin.max(xmax - 2), xmax, ymin, ymax, &color);
}

//pub fn report(
//    network_output: &Tensor,
//    original_img: &Tensor,
//    resized_img_width: i64,
//    resized_img_height: i64,
//) -> failure::Fallible<Tensor> {
//    let (npreds, pred_size) = network_output.size2()?;
//    let nclasses = (pred_size - 5) as usize;
//    // The bounding boxes grouped by (maximum) class index.
//    let mut bboxes: Vec<Vec<Bbox>> = (0..nclasses).map(|_| vec![]).collect();
//    // Extract the bounding boxes for which confidence is above the threshold.
//    for index in 0..npreds {
//        let single_pred = Vec::<f64>::from(network_output.get(index));
//        let confidence = single_pred[4];
//        // check if objectness confidence is high enough
//        if confidence > CONFIDENCE_THRESHOLD {
//            let mut class_index = 0;
//            // this this prediction, find out which one has highest class confidence
//            for i in 0..nclasses {
//                if single_pred[5 + i] > single_pred[5 + class_index] {
//                    class_index = i
//                }
//            }
//            if single_pred[class_index + 5] > 0. {
//                let x_bb_center = single_pred[0];
//                let y_bb_center = single_pred[0];
//                let width_bb = single_pred[2];
//                let height_bb = single_pred[3];
//                let x_bb_min = x_bb_center - width_bb / 2.;
//                let x_bb_max = x_bb_center + width_bb / 2.;
//                let y_bb_min = x_bb_center - height_bb / 2.;
//                let y_bb_max = x_bb_center + height_bb / 2.;
//                let class_conf = single_pred[5 + class_index];
//                let bbox = Bbox {
//                    xmin: single_pred[0] - single_pred[2] / 2.,
//                    ymin: single_pred[1] - single_pred[3] / 2.,
//                    xmax: single_pred[0] + single_pred[2] / 2.,
//                    ymax: single_pred[1] + single_pred[3] / 2.,
//                    confidence,
//                    class_index,
//                    class_confidence: single_pred[5 + class_index],
//                };
//                bboxes[class_index].push(bbox)
//            }
//        }
//    }
//    // Perform non-maximum suppression.
//    for bboxes_for_class in bboxes.iter_mut() {
//        // sort bboxes from lowest to higher confidence
//        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
//        let mut current_index = 0;
//        for index in 0..bboxes_for_class.len() {
//            let mut drop = false;
//            for prev_index in 0..current_index {
//                let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
//                if iou > NMS_THRESHOLD {
//                    drop = true;
//                    break;
//                }
//            }
//            if !drop {
//                bboxes_for_class.swap(current_index, index);
//                current_index += 1;
//            }
//        }
//        bboxes_for_class.truncate(current_index);
//    }
//    // Annotate the original image and print boxes information.
//    let (_, initial_h, initial_w) = original_img.size3()?;
//    let mut img = original_img.to_kind(tch::Kind::Float) / 255.;
//    let w_ratio = initial_w as f64 / resized_img_width as f64;
//    let h_ratio = initial_h as f64 / resized_img_height as f64;
//    for (class_index, bboxes_for_class) in bboxes.iter().enumerate() {
//        for b in bboxes_for_class.iter() {
//            println!("{}: {:?}", coco_classes::NAMES[class_index], b);
//            let xmin = ((b.xmin * w_ratio) as i64).max(0).min(initial_w - 1);
//            let ymin = ((b.ymin * h_ratio) as i64).max(0).min(initial_h - 1);
//            let xmax = ((b.xmax * w_ratio) as i64).max(0).min(initial_w - 1);
//            let ymax = ((b.ymax * h_ratio) as i64).max(0).min(initial_h - 1);
//            draw_rect(&mut img, xmin, xmax, ymin, ymax.min(ymin + 2));
//            draw_rect(&mut img, xmin, xmax, ymin.max(ymax - 2), ymax);
//            draw_rect(&mut img, xmin, xmax.min(xmin + 2), ymin, ymax);
//            draw_rect(&mut img, xmin.max(xmax - 2), xmax, ymin, ymax);
//        }
//    }
//    Ok((img * 255.).to_kind(tch::Kind::Uint8))
//}

//struct BBox {
//    center_x: i64,
//    center_y: i64,
//    width: i64,
//    height: i64,
//    opacity: f64,
//}
//
//struct RawBBox {
//    dim: i64,
//    row: i64,
//    col: i64,
//    bb_index: i64,
//    conf: f64,
//}
//
//fn draw_box() {
//    let mut img = image::load("test.jpg").unwrap();
//    /*
//        First anchors:
//        [(116, 90), (156, 198), (373, 326)]
//        [1, 255, 13, 13]
//    */
//    let mut bbs: Vec<RawBBox> = vec![];
//    let resized_width = 416;
//    let mut resized_img = image::resize(&img, resized_width, resized_width).unwrap();
//    let mut boxes: Vec<BBox> = Vec::with_capacity(3 * 13 * 13);
//    for boxes_in_y in 0..13 {
//        let grid_width = resized_width / 13;
//        let grid_y_start = boxes_in_y * grid_width;
//        let grid_y_center = grid_y_start + grid_width / 2;
//        for boxes_in_x in 0..13 {
//            let grid_x_start = boxes_in_x * grid_width;
//            let grid_x_center = grid_x_start + grid_width / 2;
//            let bb: &RawBBox = bbs.get((boxes_in_y * 13 + boxes_in_x) as usize).unwrap();
//            boxes.push(BBox {
//                center_x: grid_x_center,
//                center_y: grid_y_center,
//                width: grid_width,
//                height: grid_width,
//                opacity: 1.,
//            });
//        }
//    }
//    let anchors: Vec<(i64, i64)> = vec![(116, 90), (156, 198), (373, 326)];
//    for bb in boxes.iter().skip(7 * 13 + 3).take(1) {
//        for anchor in &anchors {
//            //            draw_bb_unnorm(&mut resized_img, bb.center_x-bb.width/2, bb.center_x+bb.width/2,
//            //                           bb.center_y-bb.width/2, bb.center_y+bb.width/2);
//            let (width, height) = anchor;
//            draw_bb_unnorm(
//                &mut resized_img,
//                bb.center_x - width / 2,
//                bb.center_x + width / 2,
//                bb.center_y - height / 2,
//                bb.center_y + height / 2,
//            );
//        }
//    }
//    //    draw_bb_unnorm(&mut img, 100, 200, 100, 200);
//    image::save(&resized_img, "out_res.jpg").unwrap();
//}

#[derive(Debug, Clone)]
struct SingleStructuredBB {
    // The grid should be square, so width == height
    anchor_box: Vec<(i64, i64)>,
    prediction_grid_dim: i64,
    grid_x: i64,
    grid_y: i64,
    grid_x_coef: f64,
    grid_y_coef: f64,
    bb_tw: f64,
    bb_th: f64,
    prob_contains_object_center: f64,
    classes_prob: Vec<f64>,
}

fn tensor_to_structured_output(tensor: &NetworkOutput) -> Vec<SingleStructuredBB> {
    let mut output: Vec<SingleStructuredBB> = Vec::new();
    let anchors = &tensor.anchor_boxes;
    let tensor = &tensor.single_scale_output.squeeze();
    let (grid_width, grid_height, nb_features) = tensor.size3().unwrap();
    for x in 0..grid_width {
        for y in 0..grid_height {
            let features_3_bb = tensor.get(x).get(y); //i((x, y));
            let length_each_bb = features_3_bb.size1().unwrap() / 3;
            for i in 0..3 {
                let offset = length_each_bb * i;
                output.push(SingleStructuredBB {
                    anchor_box: anchors.to_vec(),
                    prediction_grid_dim: grid_width,
                    grid_x: x,
                    grid_y: y,
                    grid_x_coef: features_3_bb.double_value(&[0 + offset]),
                    grid_y_coef: features_3_bb.double_value(&[1 + offset]),
                    bb_tw: features_3_bb.double_value(&[2 + offset]),
                    bb_th: features_3_bb.double_value(&[3 + offset]),
                    prob_contains_object_center: features_3_bb.double_value(&[4]),
                    classes_prob: Vec::<f64>::from(features_3_bb.slice(
                        0,
                        5 + offset,
                        length_each_bb + offset,
                        1,
                    )),
                })
            }
        }
    }
    output
}

/// Each output vec element is of size (N, W, W, Feat) where N is the batch size (should be 1)
/// W is the Width (or Height since they should be equal) of the grid which maps to the image space
/// wise, these are the divided by 32 or the resized image
/// and Feat is the feature map, of size 255 =  3*85, which are the features of the 3 BB
/// concatenated, for example, for resized size of 416: (Batch_size, 13, 13, 255) (Batch_size, 26, 26, 255) ...
fn nn_output_to_structured_output(output: Vec<NetworkOutput>) -> Vec<Vec<SingleStructuredBB>> {
    // First scale BB
    let structured_bb: Vec<Vec<SingleStructuredBB>> = output
        .iter()
        .map(|e| tensor_to_structured_output(&e))
        .collect();
    structured_bb
}

fn render_on_image(bbs: Vec<Vec<SingleStructuredBB>>) {
    let mut img = image::load("test2.jpg").unwrap();

    let resized_width = 416;
    let mut resized_img = image::resize(&img, resized_width, resized_width).unwrap();

    for single_scale_bbs in bbs {
        for bb in single_scale_bbs {
            if bb.prob_contains_object_center < 0.5 {
                continue;
            }
            let x_center_proportion =
                (bb.grid_x as f64 + bb.grid_x_coef) as f32 / bb.prediction_grid_dim as f32;
            let y_center_proportion =
                (bb.grid_y as f64 + bb.grid_y_coef) as f32 / bb.prediction_grid_dim as f32;
            let img_ratio = resized_width / bb.prediction_grid_dim;
            for anchor_box in bb.anchor_box {
                let (mut width, mut height) = anchor_box;
                width = ((width as f64) * bb.bb_tw.exp()) as i64;
                height = ((height as f64) * bb.bb_th.exp()) as i64;
                let mut max_arg = 0;
                let mut max_prob = 0.;
                for (index, prob) in bb.classes_prob.iter().enumerate() {
                    if prob > &max_prob {
                        max_prob = *prob;
                        max_arg = index;
                    }
                }
                println!("{}", NAMES[max_arg]);
                let x_min = (resized_width as f32 * x_center_proportion) as i64 - width / 2;
                let x_max = (resized_width as f32 * x_center_proportion) as i64 + width / 2;
                let y_min = (resized_width as f32 * y_center_proportion) as i64 - height / 2;
                let y_max = (resized_width as f32 * y_center_proportion) as i64 + height / 2;
                draw_bb_unnorm(&mut resized_img, x_min, x_max, y_min, y_max);
            }
        }
    }
    image::save(&resized_img, "testtt.jpg");
}

//    let mut boxes: Vec<BBox> = Vec::with_capacity(3 * 13 * 13);
//    for boxes_in_y in 0..13 {
//        let grid_width = resized_width / 13;
//        let grid_y_start = boxes_in_y * grid_width;
//        let grid_y_center = grid_y_start + grid_width / 2;
//        for boxes_in_x in 0..13 {
//            let grid_x_start = boxes_in_x * grid_width;
//            let grid_x_center = grid_x_start + grid_width / 2;
//            let bb: &RawBBox = bbs.get((boxes_in_y * 13 + boxes_in_x) as usize).unwrap();
//            boxes.push(BBox {
//                center_x: grid_x_center,
//                center_y: grid_y_center,
//                width: grid_width,
//                height: grid_width,
//                opacity: 1.,
//            });
//        }

//    let anchors: Vec<(i64, i64)> = vec![(116, 90), (156, 198), (373, 326)];
//    for bb in boxes.iter().skip(7 * 13 + 3).take(1) {
//        for anchor in &anchors {
//            //            draw_bb_unnorm(&mut resized_img, bb.center_x-bb.width/2, bb.center_x+bb.width/2,
//            //                           bb.center_y-bb.width/2, bb.center_y+bb.width/2);
//            let (width, height) = anchor;
//            draw_bb_unnorm(
//                &mut resized_img,
//                bb.center_x - width / 2,
//                bb.center_x + width / 2,
//                bb.center_y - height / 2,
//                bb.center_y + height / 2,
//            );
//        }
//    }
//    //    draw_bb_unnorm(&mut img, 100, 200, 100, 200);
//    image::save(&resized_img, "out_res.jpg").unwrap();
//}

//pub struct Matrix<T, const A: usize, const B: usize, const C: usize>{
//
//}

pub fn main() -> failure::Fallible<()> {
    //    let args: Vec<_> = std::env::args().collect();
    //    ensure!(args.len() >= 3, "usage: main yolo-v3.ot img.jpg ...");
    let a: R2Tensor<3, 2> = tch::Tensor::rand(&[3, 2], (Kind::Float, Device::Cpu)).into();
    //    let b = a.squeeze();

    let mut img = image::load("test.jpg").unwrap();
    exit(0);
    //    [(116, 90), (156, 198), (373, 326)]
    //    [1, 255, 13, 13]

    // Create the model and load the weights from the file.
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let darknet = network::parse_config(CONFIG_NAME)?;
    let model = darknet.build_model(&vs.root())?;
    vs.load("yolo-v3.ot")?;
    let imgs_path = "test2.jpg";
    let original_image_2: R3TensorGeneric = tch::rust_image::load_img_as_tensor_generic(imgs_path);
    let original_image = image::load(imgs_path)?;

    let net_width = darknet.width()?;
    let net_height = darknet.height()?;

    let resized_img: R3Tensor<3, 416, 416> = tch::rust_image::resize_img(&original_image_2);

    let image_as_batch_normalized: R4TensorGeneric =
        (resized_img.tensor.unsqueeze(0).to_kind(tch::Kind::Float) / 255.).into();
    /*
        width: 416 width/32: 13 size: [10647, 85]
        (((13*13*85)+26*26*85+52*52*85)*3)/85 = 10647
        Predicts 3 boxes for each grid square = 3*19*19 boxes, but it predicts at 3 scales, so
        3*3*19*19 = 3249
    */
    // Vec of 3 tensor of shape
    let predictions = model(&image_as_batch_normalized, false);
    let vec_per_scale_bbs = nn_output_to_structured_output(predictions);
    render_on_image(vec_per_scale_bbs);
    // (Batch_size, 13, 13, 255) (Batch_size, 26, 26, 255) ...

    //    let first_scale_dets = &predictions[0];

    // Here we would calculate loss and do backwards

    //    let (batch_size, t_row, t_col, features_each_grid_el) = first_scale_dets.size4().unwrap();
    //    // convert to BB to be drawn
    //    let mut bbs = vec![];
    //    for row in 0..t_row {
    //        for col in 0..t_col {
    //            let feature_map = first_scale_dets.squeeze().get(row).get(col).view((3, 85));
    //            for bb_index in 0..3 {
    //                //                    println!("row {:?} col {:?} bb_index {:?}", row, col, bb_index);
    //                bbs.push(RawBBox {
    //                    row,
    //                    col,
    //                    bb_index: bb_index,
    //                    conf: feature_map.get(bb_index).get(4).sigmoid().double_value(&[]),
    //                });
    //            }
    //        }
    //    }
    //
    //    println!("Pred: {:?}", predictions[0].size());
    //        predictions.logi
    //        let image = report(&predictions, &original_image, net_width, net_height)?;
    //        image::save(&image, format!("output-{:05}.jpg", index))?;
    //        println!("Converted {}", index);
    Ok(())
}
