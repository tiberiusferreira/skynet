use std::fs::File;

use imageproc::drawing::Blend;
use tch::nn::{ModuleT, OptimizerConfig};
use tch::{nn, Device, Kind, Reduction, Tensor, IndexOp};

use crate::preprocessing::bbox_conversion::{
    bbs_to_tensor, flip_bb_horizontally, flip_bb_vertically, objects_mask_tensor_from_target_tensor,
};
use crate::preprocessing::dataset::yolo_dataset_loader::YoloDataLoader;
use crate::preprocessing::from_img_to_tensor;
use crate::preprocessing::structs::{Bbox, CleanedImgLabels};
use image::{DynamicImage, GenericImageView};
use std::process::exit;
use tch::vision::image::{resize, save};

mod preprocessing;

const NB_CLASSES: i64 = 1;


pub fn leaky_relu_with_slope(t1: &Tensor) -> Tensor{
    let t1 = t1 * 0.1;
    t1.max1(&t1)
}
fn net(vs: &nn::Path) -> impl ModuleT {
    let conv_cfg = nn::ConvConfig {
        padding: 1,
        ..Default::default()
    };
    let conv_cfg_no_pad = nn::ConvConfig {
        padding: 0,
        ..Default::default()
    };

    nn::seq_t()
        .add(nn::conv2d(vs, 3, 16, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 16, Default::default()))
        .add_fn(|x|  leaky_relu_with_slope(x))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 16, 32, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 32, Default::default()))
        .add_fn(|x|  leaky_relu_with_slope(x))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 32, 64, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 64, Default::default()))
        .add_fn(|x|  leaky_relu_with_slope(x))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 64, 128, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 128, Default::default()))
        .add_fn(|x|  leaky_relu_with_slope(x))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 128, 256, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 256, Default::default()))
        .add_fn(|x|  leaky_relu_with_slope(x))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 256, 512, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 512, Default::default()))
        .add_fn(|x|  leaky_relu_with_slope(x))
        //        .add_fn(|x| x.max_pool2d(&[2, 2], &[1, 1], &[1, 1], &[1, 1], false))
        .add(nn::conv2d(vs, 512, 1024, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 1024, Default::default()))
        .add_fn(|x|  leaky_relu_with_slope(x))
        .add(nn::conv2d(vs, 1024, 256, 1, conv_cfg_no_pad))
        .add(nn::batch_norm2d(vs, 256, Default::default()))
        .add_fn(|x|  leaky_relu_with_slope(x))
        .add(nn::conv2d(vs, 256, 512, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 512, Default::default()))
        .add_fn(|x|  leaky_relu_with_slope(x))
        // Output is (x, y, w, h, object_prob, class1_prob, class2_prob, ...)
        .add(nn::conv2d(vs, 512, NB_CLASSES + 5, 1, Default::default()))
        .add_fn(|x| x.shallow_clone()) // Linear activation
        // normalize output of probabilities
        .add_fn(|x| {
            let (_batch, features, _, _) = x.size4().unwrap();
            let nb_classes = features - 5;

            // Center x and y
            let x_y = x.narrow(1, 0, 2).sigmoid();
            // width, height, can be negative because real Width = exp(width) * anchor_width
            // same for Height
            let rest = x.narrow(1, 2, 2);
            // Object confidence and class probabilities
            let probs = x.narrow(1, 4, nb_classes + 1).sigmoid();
            Tensor::cat(&[x_y, rest, probs], 1)
        })
}

fn main() -> failure::Fallible<()> {
    let mut store = nn::VarStore::new(Device::Cpu);
    let net = net(&store.root());
    let mut opt = nn::Adam::default().build(&store, 1e-3)?;
    store.load("variables.ot").unwrap();
    let nb_epochs = 10;
    for epochs in 0..nb_epochs {
        let label_n_images_filepath = "dataset/train/labels.json";
        let data_loader = YoloDataLoader::new(label_n_images_filepath, true, 1000, true);
        let data_len = data_loader.len();
        use preprocessing::augmentation::*;
        let augmented_dataset = data_loader
            .map(|(img, bbs)| {
                let bright = random_change_brightness(&img, -40, 40);
                let contrast = random_change_contrast(&img, -15., 15.);
                let hue = random_hue_rotation(&img, -35, 35);
                let blurred = random_blur(&img, 0.5, 1.0);
                vec![
                    (img, bbs.clone()),
                    (bright, bbs.clone()),
                    (contrast, bbs.clone()),
                    (hue, bbs.clone()),
                    (blurred, bbs),
                ]
                    .into_iter()
            })
            .flatten();
        let augmented_data_len = data_len*4;
        use itertools::Itertools;
        let batch_size = 64;
        for (batch_index, batch) in augmented_dataset.chunks(batch_size).into_iter().enumerate() {
            let start_batch = std::time::Instant::now();
            let batch: Vec<(DynamicImage, Vec<Bbox>)> = batch.collect();
            let (img_batch, bbox_batch): (Vec<&DynamicImage>, Vec<&Vec<Bbox>>) = batch.iter().fold(
                (vec![], vec![]),
                |(mut vec_img_acc, mut vec_bb_acc), new| {
                    let (img, bbox_vec) = new;
                    vec_img_acc.push(img);
                    vec_bb_acc.push(bbox_vec);
                    (vec_img_acc, vec_bb_acc)
                },
            );

            let (ch, width, height) = from_img_to_tensor(img_batch[0]).size3().unwrap();
            let batch_size = img_batch.len();
            let img_batch_tensor =
                Tensor::empty(&[batch_size as i64, ch as i64, width as i64, height as i64],
                              (Kind::Uint8, Device::Cpu));
            for (index, img) in img_batch.iter().enumerate(){
                let img_as_tensor = from_img_to_tensor(img);
                img_batch_tensor.i(index as i64).copy_(&img_as_tensor);
            }
            let normalized_img_tensor_batch =
                img_batch_tensor.to_kind(tch::Kind::Float) / 255.;
            let normalized_img_tensor_batch = normalized_img_tensor_batch.set_requires_grad(true);

            let sample_desired = bbs_to_tensor(bbox_batch[0], 13, 416, (70, 70));
            let (a, b, c) = sample_desired.tensor.size3().unwrap();
            let desired_batch = Tensor::empty(&[batch_size as i64, a as i64, b as i64, c as i64], (Kind::Float, Device::Cpu));
            for (index, bb) in bbox_batch.iter().enumerate(){
                let desired = bbs_to_tensor(&bb, 13, 416, (70, 70));
                desired_batch.i(index as i64).copy_(&(desired.tensor));
            }
            let batch_output = net.forward_t(&normalized_img_tensor_batch, true);
            let loss = yolo_loss(desired_batch, batch_output);
            opt.backward_step(&loss);
            //            if index % 300 == 0 {
            println!("Loss {:?} {}/{}", loss, batch_index, augmented_data_len/batch_size);
            //            }
            //            if index % 300 == 0 {
            print_test_loss(&net);
            //            }
        }
        store.save("variables.ot").unwrap();
        println!("Saved Weights!");
    }

    Ok(())
}

pub fn print_test_loss(net: &impl ModuleT) {
    let label_n_images_filepath = "dataset/test/labels.json";
    let data_loader = YoloDataLoader::new(label_n_images_filepath, false, 1, false);
    for (index, (img, bb_vec)) in data_loader.enumerate() {
        let img_as_tensor = from_img_to_tensor(&img);
        let img_as_normalized_tensor_batch =
            img_as_tensor.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;
        let img_as_normalized_tensor_batch =
            img_as_normalized_tensor_batch.set_requires_grad(false);
        let desired = bbs_to_tensor(&bb_vec, 13, 416, (70, 70))
            .tensor
            .unsqueeze(0);
        let output = net.forward_t(&img_as_normalized_tensor_batch, true);
        let loss = yolo_loss(desired, output);
        println!("Test loss for img {} = {}", index, loss.double_value(&[]));
    }
}
pub fn yolo_loss(desired: Tensor, output: Tensor) -> Tensor {
    let target_object_prob = desired.narrow(1, 4, 1);
    let output_object_prob = output.narrow(1, 4, 1);
    let objectness_loss = output_object_prob.binary_cross_entropy::<Tensor>(
        &target_object_prob,
        None,
        Reduction::Mean,
    );

    let (batch_size, _, _, _) = desired.size4().unwrap();
    let mut others_loss: Tensor = Tensor::from(0.);
    for prediction_index in 0..batch_size{
        // TODO objects_mask_tensor_from_target_tensor assumes it take only one sample, not a batch
        let grid_points_with_obj = objects_mask_tensor_from_target_tensor(desired.i(prediction_index).into(), 13);
        for point in grid_points_with_obj {
            let target_others = desired
                .narrow(0, prediction_index, 1)
                .narrow(1, 0, 4)
                .narrow(2, point.x, 1)
                .narrow(3, point.y, 1);
            let output_others = output
                .narrow(0, prediction_index, 1)
                .narrow(1, 0, 4)
                .narrow(2, point.x, 1)
                .narrow(3, point.y, 1);

            others_loss += output_others.mse_loss(&target_others, Reduction::Sum);
        }
    }
    others_loss = others_loss/batch_size;
    objectness_loss + others_loss
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::bbox_conversion::{bbs_from_tensor, draw_bb_to_img};

    #[test]
    fn test_network_works() {
        let mut store = nn::VarStore::new(Device::Cpu);
        let net = net(&store.root());

        store.load("variables.ot").unwrap();
        let test_resized_img_path = "dataset/test/m5.jpg";
        let train_resized_img_path = "dataset/train/image0000.jpg";
        let resized_img_path = train_resized_img_path;
        let resized = tch::vision::image::load(resized_img_path).unwrap();

        let img_as_batch = resized.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;
        let output = net.forward_t(&img_as_batch, true);
        let out = output.squeeze();
        let bb = bbs_from_tensor(out.into(), 13, 416, (70, 70));
        //        let bb = from_tensor(desired.squeeze().into(), 13, 416, (70, 70));

        let img = image::open(resized_img_path).unwrap();
        let mut blend = Blend(img.to_rgba());
        for single_bb in &bb {
            draw_bb_to_img(&mut blend, single_bb);
        }
        blend.0.save("maybe_worked.jpg").unwrap();
        println!("{:?}", bb);
    }
}
