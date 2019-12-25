use lazy_static::*;
use std::fs::File;

use imageproc::drawing::Blend;
use tch::nn::{Adam, ModuleT, Optimizer, OptimizerConfig, VarStore};
use tch::{nn, Device, IndexOp, Kind, Reduction, Tensor};

use super::dataset;
use super::dataset::iterator_adapters::*;
use crate::dataset::common_structs::SimpleBbox;
use crate::dataset::data_augmenters::image_augmentations::default_augmentation;
use crate::dataset::data_loaders::yolo_dataset_loader::YoloDataLoader;
use crate::dataset::data_transformers::bbox_conversion::bbs_to_tensor;
use crate::dataset::data_transformers::img2tensor::from_img_to_tensor;
use crate::dataset::DataLoader;
use image::{DynamicImage, GenericImageView};
use std::process::exit;
use tch::vision::image::{resize, save};
pub mod yolo_loss;
use crate::yolo_nn::yolo_loss::yolo_loss;

const NB_CLASSES: i64 = 1;

lazy_static! {
    pub static ref DEVICE: Device = {
        if tch::Cuda::is_available(){
            println!("Using CPU");
            Device::Cpu
        }else{
            println!("Using GPU");
            Device::Cpu
        }
    };
}
//pub const DEVICE: Device = {
//    if true {
//        //tch::Cuda::is_available(){
//        Device::Cpu
//    } else {
//        Device::Cpu
//    }
//};

pub fn leaky_relu_with_slope(t1: &Tensor) -> Tensor {
    let t1 = t1 * 0.1;
    t1.max1(&t1)
}

fn yolo_net(vs: &nn::Path) -> impl ModuleT {
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
        .add_fn(|x| leaky_relu_with_slope(x))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 16, 32, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 32, Default::default()))
        .add_fn(|x| leaky_relu_with_slope(x))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 32, 64, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 64, Default::default()))
        .add_fn(|x| leaky_relu_with_slope(x))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 64, 128, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 128, Default::default()))
        .add_fn(|x| leaky_relu_with_slope(x))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 128, 256, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 256, Default::default()))
        .add_fn(|x| leaky_relu_with_slope(x))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 256, 512, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 512, Default::default()))
        .add_fn(|x| leaky_relu_with_slope(x))
        //        .add_fn(|x| x.max_pool2d(&[2, 2], &[1, 1], &[1, 1], &[1, 1], false))
        .add(nn::conv2d(vs, 512, 1024, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 1024, Default::default()))
        .add_fn(|x| leaky_relu_with_slope(x))
        .add(nn::conv2d(vs, 1024, 256, 1, conv_cfg_no_pad))
        .add(nn::batch_norm2d(vs, 256, Default::default()))
        .add_fn(|x| leaky_relu_with_slope(x))
        .add(nn::conv2d(vs, 256, 512, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 512, Default::default()))
        .add_fn(|x| leaky_relu_with_slope(x))
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

pub fn yolo_trainer() -> failure::Fallible<()> {
    let net_params_store = nn::VarStore::new(*DEVICE);
    let mut network = yolo_net(&net_params_store.root());
    let mut opt = nn::Adam::default().build(&net_params_store, 1e-3)?;
    //    net_params_store.load("variables.ot").unwrap();
    let nb_epochs = 10;
    for epochs in 0..nb_epochs {
        let label_n_images_filepath = "dataset/train/labels.json";
        let data_loader = YoloDataLoader::new(label_n_images_filepath);
        let augmented = data_loader
            .map(|(img, bboxes)| {
                let augmented_imgs = default_augmentation(img);
                let out: Vec<(DynamicImage, Vec<SimpleBbox>)> = augmented_imgs
                    .into_iter()
                    .map(|img| (img, bboxes.clone()))
                    .collect();
                out
            })
            .flatten()
            .shuffling(1000)
            .dataset_batching(16);
        for batch in augmented {
            let start_batch = std::time::Instant::now();
            // Help IDE
            let batch: Vec<(DynamicImage, Vec<SimpleBbox>)> = batch;
            train_single_batch(batch, &mut network, &mut opt);
            println!("Batch took {} s", start_batch.elapsed().as_secs());
        }
        println!("Epoch {} of {}", epochs, nb_epochs);
        net_params_store.save("variables.ot").unwrap();
        println!("Saved Weights!");
    }

    Ok(())
}

fn train_single_batch(
    batch: Vec<(DynamicImage, Vec<SimpleBbox>)>,
    network: &mut impl ModuleT,
    opt: &mut Optimizer<Adam>,
) {
    let (img_batch, bbox_batch): (Vec<&DynamicImage>, Vec<&Vec<SimpleBbox>>) = batch.iter().fold(
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
    let img_batch_tensor = Tensor::empty(
        &[batch_size as i64, ch as i64, width as i64, height as i64],
        (Kind::Uint8, *DEVICE),
    );
    for (index, img) in img_batch.iter().enumerate() {
        let img_as_tensor = from_img_to_tensor(img);
        img_batch_tensor.i(index as i64).copy_(&img_as_tensor);
    }
    let normalized_img_tensor_batch = img_batch_tensor.to_kind(tch::Kind::Float) / 255.;
    let normalized_img_tensor_batch = normalized_img_tensor_batch
        .set_requires_grad(true)
        .to_device(*DEVICE);

    let sample_desired = bbs_to_tensor(bbox_batch[0], 13, 416, (70, 70));
    let (a, b, c) = sample_desired.tensor.size3().unwrap();
    let desired_batch = Tensor::empty(
        &[batch_size as i64, a as i64, b as i64, c as i64],
        (Kind::Float, *DEVICE),
    );
    for (index, bb) in bbox_batch.iter().enumerate() {
        let desired = bbs_to_tensor(&bb, 13, 416, (70, 70));
        desired_batch.i(index as i64).copy_(&(desired.tensor));
    }
    let batch_output = network.forward_t(&normalized_img_tensor_batch, true);
    let loss = yolo_loss(desired_batch, batch_output).to_device(*DEVICE);
    opt.backward_step(&loss);
    println!("Loss {:?}", loss);
    print_test_loss(network);
}
pub fn print_test_loss(net: &impl ModuleT) {
    let label_n_images_filepath = "dataset/test/labels.json";
    let mut data_loader = YoloDataLoader::new(label_n_images_filepath);
    while let Some((img, bb_vec)) = data_loader.next() {
        let img_as_tensor = from_img_to_tensor(&img);
        let img_as_normalized_tensor_batch =
            img_as_tensor.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;
        let img_as_normalized_tensor_batch =
            img_as_normalized_tensor_batch.set_requires_grad(false);
        let desired = bbs_to_tensor(&bb_vec, 13, 416, (70, 70))
            .tensor
            .unsqueeze(0);
        let output = net.forward_t(&img_as_normalized_tensor_batch, true);
        let loss = yolo_loss::yolo_loss(desired, output);
        println!("Test loss for img = {}", loss.double_value(&[]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //    use crate::dataset::bbox_conversion::{bbs_from_tensor, draw_bb_to_img};

    //    #[test]
    //    fn test_network_works() {
    //        let mut store = nn::VarStore::new(DEVICE);
    //        let net = yolo_net(&store.root());
    //
    //        store.load("variables.ot").unwrap();
    //        let test_resized_img_path = "dataset/test/m32.jpg";
    //        let train_resized_img_path = "dataset/train/image0000.jpg";
    //        let resized_img_path = train_resized_img_path;
    //        let resized = tch::vision::image::load(resized_img_path).unwrap();
    //
    //        let img_as_batch = resized.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;
    //        let output = net.forward_t(&img_as_batch, true);
    //        let out = output.squeeze();
    //        let bb = bbs_from_tensor(out.into(), 13, 416, (70, 70));
    //        //        let bb = from_tensor(desired.squeeze().into(), 13, 416, (70, 70));
    //
    //        let img = image::open(resized_img_path).unwrap();
    //        let mut blend = Blend(img.to_rgba());
    //        for single_bb in &bb {
    //            draw_bb_to_img(&mut blend, single_bb);
    //        }
    //        blend.0.save("maybe_worked.jpg").unwrap();
    //        println!("{:?}", bb);
    //    }
}