use lazy_static::*;

use tch::nn::{ModuleT, Optimizer, OptimizerConfig};
use tch::{nn, Device, IndexOp, Kind, Reduction, Tensor, R4TensorGeneric};

use super::dataset::iterator_adapters::*;
use crate::dataset::common_structs::SimpleBbox;
use crate::dataset::data_loaders::yolo_dataset_loader::YoloDataLoader;
use crate::dataset::data_transformers::img2tensor::from_img_to_tensor;
use image::DynamicImage;
pub mod network;
pub mod yolo_bbox_conversion;
pub mod yolo_loss;
mod helpers;
use crate::dataset::data_augmenters::image_augmentations::default_augmentation;
use crate::yolo_nn::yolo_bbox_conversion::{bbs_to_tensor, yolo_bbs_from_tensor2};
use crate::yolo_nn::yolo_loss::{yolo_loss, yolo_loss2};
use network::micro_yolo_net;
use tch::vision::image::save;
use crate::yolo_nn::network::{YoloNetworkOutput, DarknetConfig};
use std::process::exit;
use itertools::Itertools;

const NB_CLASSES: i64 = 1;

lazy_static! {
    pub static ref DEVICE: Device = {
        if tch::Cuda::is_available() {
            println!("Using GPU");
            Device::Cuda(0)
        } else {
            println!("Using CPU");
            Device::Cpu
        }
    };
}

pub fn leaky_relu_with_slope(t1: &Tensor) -> Tensor {
    let t2 = t1 * 0.1;
    t1.max1(&t2)
}

#[cfg(test)]
mod loss_tests {
    use super::*;
    use tch::vision::image::load;

    #[test]
    fn network_loss() {
        let network = DarknetConfig::new("yolo-v3_modif.cfg", *DEVICE).unwrap();
        let (mut vs, model) = network.build_model().unwrap();
        vs.load("yolo-v3_modif_trainned.ot").unwrap();
        let img = load("code_test_data/test_img.jpg").unwrap().unsqueeze(0);
        let img_batch_tensor = img.to_kind(tch::Kind::Float) / 255.;

        let bb = SimpleBbox {
            top: 67,
            left: 54,
            height: 130,
            width: 146,
            prob: 0.0,
            class: 0,
        };
        let ground_truth = vec![bb];
        let out = model(&img_batch_tensor.into(), true);
        let mut loss= Tensor::from(0.).to_device(*DEVICE);

        let start = std::time::Instant::now();
        for scale_pred in out.iter() {
            // each prediction scale is a tensor containing all predictions at this scale
            let single_img_pred = &scale_pred.single_scale_output.i(0);
            let output = YoloNetworkOutput{
                single_scale_output: single_img_pred.shallow_clone(),
                anchor_boxes: scale_pred.anchor_boxes.clone()
            };
            loss += yolo_loss2(&ground_truth, &output, 416, *DEVICE);
        }
        println!("Took: {}ms calculating loss", start.elapsed().as_millis());
        println!("Loss: {}", loss.double_value(&[]));
        //3.072609091710629
        //3.0726102585447266


    }
//    println!("Took: {}ms in single call", start.elapsed().as_millis());

}

pub fn output_loss(out: &Vec<YoloNetworkOutput>, mini_batch_bbs: Vec<&Vec<SimpleBbox>>) -> Tensor{
    let mut batch_loss = Tensor::from(0.).to_device(*DEVICE);
    for scale_pred in out.iter() {
        // each prediction scale is a tensor containing all predictions at this scale
        // for this batch
        for single_img_pred_index in 0..mini_batch_bbs.len(){
            let single_img_pred = &scale_pred.single_scale_output.i(single_img_pred_index as i64);
            let output = YoloNetworkOutput{
                single_scale_output: single_img_pred.shallow_clone(),
                anchor_boxes: scale_pred.anchor_boxes.clone()
            };
            let start = std::time::Instant::now();
            batch_loss += yolo_loss2(&mini_batch_bbs[single_img_pred_index], &output, 416, *DEVICE);
        }
    }
    batch_loss
}
pub fn yolo_trainer() -> failure::Fallible<()> {
    //    let mut net_params_store = nn::VarStore::new(*DEVICE);
    const BATCH_SIZE: usize = 128; //128
    const MINI_BATCH_SIZE: usize = 16; //16
    let network = DarknetConfig::new("yolo-v3_modif.cfg", *DEVICE).unwrap();
    let (mut vs, model) = network.build_model().unwrap();


//    vs.load("yolo-v3_modif_trainned.ot").unwrap();
//    vs.load("yolo-v3_modif_trainned.ot").unwrap();
//    vs.load("yolo-v3.ot").unwrap();
    vs.load("../content/gdrive/My Drive/yolo-v3_modif_trainned.ot").unwrap();

    vs.freeze();
    for (a, t) in vs.variables().iter_mut().filter(|(s, t)| s.contains("custom")){
        t.set_requires_grad(true);
    }
    let mut opt = nn::Sgd::default().build(&vs, 1e-3)?;
    let nb_epochs = 200;
    let mut done_batches = 0;
    for epochs in 0..nb_epochs {
        let label_n_images_filepath = "labelbox_dataset/train/labels.json";
//        let label_n_images_filepath = "coco/test/labels.json";
        let start_epoch = std::time::Instant::now();
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
            .shuffling(BATCH_SIZE*5)
            .dataset_batching(BATCH_SIZE);
        for batch in augmented {
            // Help IDE
            let batch: Vec<(DynamicImage, Vec<SimpleBbox>)> = batch;
            let actual_batch_size = batch.len();
            // get img dimensions
            let (ch, width, height) = from_img_to_tensor(&batch[0].0).size3().unwrap();

            let mut batch_loss = Tensor::from(0.).to_device(*DEVICE);

            // MINI Batches
            for mini_batch in batch.iter().dataset_batching(MINI_BATCH_SIZE){
                let actual_mini_batch_size = mini_batch.len();
                let mini_batch_start = std::time::Instant::now();
                let img_mini_batch_tensor = Tensor::zeros(
                    &[actual_mini_batch_size as i64, ch as i64, width as i64, height as i64],
                    (Kind::Uint8, *DEVICE),
                );

                let mut mini_batch_bbs = vec![];
                for (index, (img, bb)) in mini_batch.iter().enumerate(){
                    let img = from_img_to_tensor(img);
                    img_mini_batch_tensor.i(index as i64).copy_(&img);
                    mini_batch_bbs.push(bb);
                }
                let img_mini_batch_tensor = img_mini_batch_tensor.to_kind(tch::Kind::Float) / 255.;


                let out = model(&img_mini_batch_tensor.into(), true);

                batch_loss += output_loss(&out, mini_batch_bbs);
            }

            let loss = batch_loss / actual_batch_size as i64;

            println!("Loss: {} for batch of {}", loss.double_value(&[]), actual_batch_size);
            opt.backward_step(&loss);
            done_batches += 1;
//            if done_batches % 10 == 0 {

//            }
        }
        println!("Done Batches: {}", done_batches);
        println!("Epoch took {} s", start_epoch.elapsed().as_secs_f32());
        vs.save("../content/gdrive/My Drive/yolo-v3_modif_trainned.ot").unwrap();
//        vs.save("yolo-v3_modif_trainned.ot").unwrap();

        println!("Saved Weights!");
        println!("Epoch {} of {}", epochs, nb_epochs);
        print_test_loss(&model);
    }

    Ok(())
}


pub fn print_test_loss(net: &Box<dyn Fn(&R4TensorGeneric, bool) -> Vec<YoloNetworkOutput>>) {
    //    let label_n_images_filepath = "dataset/test/labels.json";
    let label_n_images_filepath = "labelbox_dataset/test/labels.json";
    let mut data_loader = YoloDataLoader::new(label_n_images_filepath);
    let mut loss = Tensor::from(0.).to_device(*DEVICE);
    let mut i = 0;
    while let Some((img, bb_vec)) = data_loader.next() {
        let img_as_tensor = from_img_to_tensor(&img);
        let img_as_normalized_tensor_batch =
            img_as_tensor.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;
        let img_as_normalized_tensor_batch =
            img_as_normalized_tensor_batch.set_requires_grad(false);
        let out = net(&img_as_normalized_tensor_batch.into(), false);

        loss += output_loss(&out, vec![&bb_vec]);
        i += 1;
    }

    println!("Test loss for img = {}", loss.double_value(&[])/i as f64);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yolo_nn::yolo_bbox_conversion::{bbs_from_tensor};
    use imageproc::drawing::Blend;
    use crate::yolo_nn::helpers::img_drawing::{draw_bb_to_img, draw_bb_to_img_from_file};

    #[test]
    fn test_network_works() {
        let network = DarknetConfig::new("yolo-v3_modif.cfg", *DEVICE).unwrap();
        let (mut vs, model) = network.build_model().unwrap();
        vs.load("yolo-v3_modif_trainned.ot").unwrap();
        println!("Loaded");
        let label_n_images_filepath = "labelbox_dataset/train/labels.json";
//        let label_n_images_filepath = "coco/test/labels.json";
        let data_loader = YoloDataLoader::new(label_n_images_filepath);
        let mut i = 0;
        for (mut img, bb) in data_loader.take(10) {
            let tensor = from_img_to_tensor(&img);
            let img_as_batch = tensor.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;


            let out = model(&img_as_batch.into(), true);
            let mut out: Vec<YoloNetworkOutput> = out
                .into_iter()
                .map(|e| YoloNetworkOutput {
                    single_scale_output: e.single_scale_output.squeeze(),
                    anchor_boxes: e.anchor_boxes,
                })
                .collect();
            let mut bbs = vec![];
            for scale_pred in out.iter().skip(0).take(3) {
                let new_bbs = yolo_bbs_from_tensor2(scale_pred, 416);
                bbs.extend_from_slice(new_bbs.as_slice());
            }
            let path = format!("test_results/{}.jpg", i);
            for bb in &bbs{
                draw_bb_to_img(&mut img, bb);
            }
            img.save(path);
            i += 1;
        }
    }
}
