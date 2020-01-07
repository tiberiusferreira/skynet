use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::exit;
use tch::{
    nn,
    nn::{FuncT, ModuleT},
    Device, Kind, R1Tensor, R2Tensor, R4Tensor, R4TensorGeneric, Tensor,
};
mod config;
mod config_file_parsing;
use crate::yolo_nn::network::original_yolo::config_file_parsing::{ConfigBlock, DarknetParsedFile};
use anyhow::{bail, ensure};
pub use config::YoloNetworkOutput;
use std::collections::BTreeMap;

// Apply f to a slice of tensor xs and replace xs values with f output.
fn slice_apply_and_set<F>(xs: &mut Tensor, start: i64, len: i64, f: F)
where
    F: FnOnce(&Tensor) -> Tensor,
{
    let mut slice = xs.narrow(2, start, len);
    let src = f(&slice);
    slice.copy_(&src)
}

fn detect(xs: &Tensor, image_height: i64, classes: i64, anchors: &Vec<(i64, i64)>) -> Tensor {
    let (bsize, _channels, conv_output_height, _width) = xs.size4().unwrap();
    let img_height_per_conv_output_height = image_height / conv_output_height;
    let grid_size = conv_output_height;
    // println!("{:?}  {:?}  {:?}  {:?}", image_height, conv_output_height, img_height_per_conv_output_height, grid_size);
    let nb_attrs_per_bb = 5 + classes;
    let nb_anchors_per_bb = anchors.len() as i64;
    let mut xs = xs //  (1, 255, 13, 13)
        .view((
            bsize,                               // 1
            nb_attrs_per_bb * nb_anchors_per_bb, // 85 * 3 = 255
            grid_size * grid_size,               // 13 * 13
        ))
        .transpose(1, 2) // (1, 13*13, 255)
        .contiguous()
        .view((
            bsize,
            grid_size * grid_size * nb_anchors_per_bb,
            nb_attrs_per_bb,
        )); // (1, 13*13*3, 85)
    let grid = Tensor::arange(grid_size, tch::kind::FLOAT_CPU);
    let a = grid.repeat(&[grid_size, 1]);
    // a = 0 1 2 ..
    //     0 1 2 ..
    //     0 1 2 ..
    let b = a.tr().contiguous();
    // b = 0 0 0 ..
    //     1 1 1 ..
    //     2 2 2 ..
    let x_offset = a.view((-1, 1));
    let y_offset = b.view((-1, 1));
    let xy_offset = Tensor::cat(&[x_offset, y_offset], 1)
        .repeat(&[1, nb_anchors_per_bb])
        .view((-1, 2))
        .unsqueeze(0);
    let anchors: Vec<f32> = anchors
        .iter()
        .flat_map(|&(x, y)| {
            vec![
                x as f32 / img_height_per_conv_output_height as f32,
                y as f32 / img_height_per_conv_output_height as f32,
            ]
            .into_iter()
        })
        .collect();
    let anchors = Tensor::of_slice(&anchors)
        .view((-1, 2))
        .repeat(&[grid_size * grid_size, 1])
        .unsqueeze(0);
    // xs here is: [box_coordinates objectness_score class_scores]
    // [tx, ty, tw, th,         po,         p1, p2, ...]
    // box_x, box_y, box_w, box_h = box coordinates in grid coordinates
    // box_x = sigmoid(tx) + c_x (x of top left of the grid cell)
    // box_y = sigmoid(ty) + c_y (y of top left of the grid cell)
    slice_apply_and_set(&mut xs, 0, 2, |xs| xs.sigmoid() + xy_offset);
    // first element is objectness probability, others are classes probabilities
    slice_apply_and_set(&mut xs, 4, 1 + classes, Tensor::sigmoid);
    // box_w = p_w*e^(t_w)
    // box_h = p_h*e^(t_h)
    slice_apply_and_set(&mut xs, 2, 2, |xs| xs.exp() * anchors);
    // Converting coordinates from grid (13x13 for example) to actual image size
    slice_apply_and_set(&mut xs, 0, 4, |xs| xs * img_height_per_conv_output_height);
    xs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yolo_nn::network::original_yolo::config::{DarknetConfig, YoloNetworkOutput};
    use crate::yolo_nn::yolo_bbox_conversion::{yolo_bbs_from_tensor2};
    use crate::yolo_nn::yolo_loss::yolo_loss2;
    use tch::vision::image::{load, resize, save};
    use tch::vision::imagenet::save_image;
    use crate::yolo_nn::helpers::img_drawing::draw_bb_to_img_from_file;
    use crate::dataset::common_structs::SimpleBbox;

    #[test]
    fn test_network_v2_works() {
        let yolo = DarknetConfig::new("yolo-v3_modif.cfg").unwrap();
        let (mut vs, model) = yolo.build_model().unwrap();
        vs.load("yolo-v3_modif.ot").unwrap();
        vs.freeze();
        for (a, t) in vs.variables().iter_mut().filter(|(s, t)| s.contains("custom")){
            t.set_requires_grad(true);
        }
        let img = load("000000181303.jpg").unwrap();
        let ground_truth_bb = SimpleBbox{
            top: 94,
            left: 148,
            height: 263,
            width: 39,
            prob: 1.0,
            class: 0
        };
        let img = resize(&img, 416, 416).unwrap();
        img.set_requires_grad(false);
        save(&img, "resized.jpg").unwrap();
        let img = img.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;

        let out = model(&img.into(), true);
        let mut out: Vec<YoloNetworkOutput> = out
            .into_iter()
            .map(|e| YoloNetworkOutput {
                single_scale_output: e.single_scale_output.squeeze(),
                anchor_boxes: e.anchor_boxes,
            })
            .collect();
        let mut bbs = vec![];
        for scale_pred in out.iter().skip(1).take(1) {
            let new_bbs = yolo_bbs_from_tensor2(scale_pred, 416);
            bbs.extend_from_slice(new_bbs.as_slice());
        }
        draw_bb_to_img_from_file("resized.jpg", "resized.jpg", &bbs);
        let loss = yolo_loss2(vec![ground_truth_bb], &out[1], 416);



        for (a, t) in vs.variables().iter().filter(|(s, t)| t.requires_grad()){
            println!("GRAD {:?} {:?}", a,t );
        }



        println!("{:?}", loss.double_value(&[]));


    }
}
//
//    pub fn build_model(
//        &self,
//        vs: &nn::Path,
//    ) -> Result<Box<dyn Fn(&R4TensorGeneric, bool) -> Vec<NetworkOutput>>, anyhow::Error> {
//        let mut blocks: Vec<(i64, Bl)> = vec![];
//        let mut prev_channels: i64 = 3;
//        for (index, block) in self.config_blocks.iter().enumerate() {
//            let channels_and_bl = match block.block_type.as_str() {
//                "convolutional" => conv(vs / index, index, prev_channels, &block)?,
//                "upsample" => upsample(prev_channels)?,
//                "shortcut" => shortcut(index, prev_channels, &block)?,
//                "route" => route(index, &blocks, &block)?,
//                "yolo" => yolo(prev_channels, &block)?,
//                otherwise => bail!("unsupported block type {}", otherwise),
//            };
//            prev_channels = channels_and_bl.0;
//            blocks.push(channels_and_bl);
//        }
//        let image_height = self.height()?;
//        let func = Box::new(
//            move |n_3_416_416: &R4TensorGeneric, train: bool| -> Vec<NetworkOutput> {
//                let mut layer_outputs: Vec<Tensor> = vec![];
//                let mut detections: Vec<NetworkOutput> = vec![];
//                for (_, b) in blocks.iter() {
//                    let ys = match b {
//                        Bl::Layer(l) => {
//                            let xs = layer_outputs.last().unwrap_or(&n_3_416_416.tensor);
//                            l.forward_t(&xs, train)
//                        }
//                        Bl::Route(layers) => {
//                            let layers: Vec<_> =
//                                layers.iter().map(|&i| &layer_outputs[i]).collect();
//                            Tensor::cat(&layers, 1)
//                        }
//                        Bl::Shortcut(from) => {
//                            layer_outputs.last().unwrap() + layer_outputs.get(*from).unwrap()
//                        }
//                        Bl::Yolo(classes, anchors) => {
//                            println!("{:?}", anchors);
//                            let last_output: R4TensorGeneric =
//                                layer_outputs.last().unwrap_or(&n_3_416_416.tensor).shallow_clone().into();
//
//                            println!("last_output size = {:?}", last_output.tensor.size());
//                            //                        detections.push(detect(last_output, image_height, *classes, anchors));
//                            /*
//                            Originally indexed as (13x13 is an example) [1, 255, 13, 13]
//                            which makes we index a given feature of the 255 possible features
//                            3(number of bb)*85(features per bb (ex: objectness prob)) and then indexing the col and row for its value
//                            We change it so we can index a row and col [13, 13] and get all the features
//                            of the 3 bounding boxes for that location
//                            */
//
//                            //                        last_output.index(&[]);
//                            let transposed: R4TensorGeneric = last_output.tensor
//                                .transpose(1, 2)
//                                .transpose(2, 3) // (Batch_size, 13, 13, 255)
//                                .contiguous().into();
//                            let (n, w, h, feat_len) = transposed.tensor.size4().unwrap();
//                            // From (Batch_size, 13, 13, 255) from the dimension 3, only select
//                            // the 5th one onward, which are the objectness and classes probabilities
//                            // elements (4, 5, ..)
//                            transposed.tensor.narrow(3, 4, feat_len - 4).sigmoid_();
//                            // Also apply sigmoid to x and y of the BB elements (0 and 1)
//                            transposed.tensor.narrow(3, 0, 2).sigmoid_();
//                            detections.push(NetworkOutput {
//                                single_scale_output: transposed.tensor,
//                                anchor_boxes: anchors.to_vec(),
//                            });
//                            Tensor::default()
//                        }
//                    };
//                    layer_outputs.push(ys);
//                }
//                detections
//            },
//        );
//        Ok(func)
//    }
