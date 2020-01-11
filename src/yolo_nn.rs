use lazy_static::*;

use tch::nn::{Adam, ModuleT, Optimizer, OptimizerConfig};
use tch::{nn, Device, IndexOp, Kind, Reduction, Tensor};

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
        let network = DarknetConfig::new("yolo-v3_modif.cfg").unwrap();
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
            loss += yolo_loss2(&ground_truth, &output, 416);
        }
        println!("Took: {}ms calculating loss", start.elapsed().as_millis());
        println!("Loss: {}", loss.double_value(&[]));
        //3.072609091710629
        //3.0726102585447266


    }
//    println!("Took: {}ms in single call", start.elapsed().as_millis());

}

pub fn yolo_trainer() -> failure::Fallible<()> {
    //    let mut net_params_store = nn::VarStore::new(*DEVICE);
    const BATCH_SIZE: usize = 128;
    const MINI_BATCH_SIZE: usize = 16;
    let network = DarknetConfig::new("yolo-v3_modif.cfg").unwrap();
    let (mut vs, model) = network.build_model().unwrap();
//    vs.load("yolo-v3_modif_trainned.ot").unwrap();
    vs.load("yolo-v3.ot").unwrap();
    vs.freeze();
    for (a, t) in vs.variables().iter_mut().filter(|(s, t)| s.contains("custom")){
        t.set_requires_grad(true);
    }
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
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

                for scale_pred in out.iter() {
                    // each prediction scale is a tensor containing all predictions at this scale
                    // for this batch
                    for single_img_pred_index in 0..actual_mini_batch_size{
                        let single_img_pred = &scale_pred.single_scale_output.i(single_img_pred_index as i64);
                        let output = YoloNetworkOutput{
                            single_scale_output: single_img_pred.shallow_clone(),
                            anchor_boxes: scale_pred.anchor_boxes.clone()
                        };
                        let start = std::time::Instant::now();
                        batch_loss += yolo_loss2(&mini_batch_bbs[single_img_pred_index], &output, 416, *DEVICE);
//                        println!("Took: {}ms in single call", start.elapsed().as_millis());
                    }
                }

                println!("Done minibatch of {}. Speed = {:0.2}img/s", actual_mini_batch_size, actual_mini_batch_size as f32/mini_batch_start.elapsed().as_secs_f32());
            }

            let loss = batch_loss / actual_batch_size as i64;
            println!("Loss: {} for batch of {}", loss.double_value(&[]), actual_batch_size);
            opt.backward_step(&loss);



//            let loss = train_single_batch(batch, &mut network, &mut opt);
            done_batches += 1;
            if done_batches % 10 == 0 {
                vs.save("yolo-v3_modif_trainned.ot").unwrap();
                println!("Saved Weights!");
                println!("Done Batches: {}", done_batches);
//                println!("Loss {:?}", loss);

            }
        }
        println!("Epoch took {} s", start_epoch.elapsed().as_secs_f32());
        println!("Epoch {} of {}", epochs, nb_epochs);
    }

    Ok(())
}

fn train_single_batch(
    batch: Vec<(DynamicImage, Vec<SimpleBbox>)>,
    network: &mut impl ModuleT,
    opt: &mut Optimizer<Adam>,
) -> f64 {
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
    let img_batch_tensor = Tensor::zeros(
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

    let sample_desired = bbs_to_tensor(bbox_batch[0], 13, 416, (70, 70), *DEVICE);
    let (a, b, c) = sample_desired.tensor.size3().unwrap();

    let desired_batch = Tensor::zeros(
        &[batch_size as i64, a as i64, b as i64, c as i64],
        (Kind::Float, *DEVICE),
    );
    for (index, bb) in bbox_batch.iter().enumerate() {
        let desired = bbs_to_tensor(&bb, 13, 416, (70, 70), *DEVICE);
        desired_batch.i(index as i64).copy_(&(desired.tensor));
    }
    let batch_output = network.forward_t(&normalized_img_tensor_batch, true);

    let loss = yolo_loss(desired_batch, batch_output).to_device(*DEVICE);
    opt.backward_step(&loss);
    loss.double_value(&[])
}

pub fn print_test_loss(net: &impl ModuleT) {
    //    let label_n_images_filepath = "dataset/test/labels.json";
    let label_n_images_filepath = "coco/test/labels.json";
    let mut data_loader = YoloDataLoader::new(label_n_images_filepath);
    while let Some((img, bb_vec)) = data_loader.next() {
        let img_as_tensor = from_img_to_tensor(&img);
        let img_as_normalized_tensor_batch =
            img_as_tensor.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;
        let img_as_normalized_tensor_batch =
            img_as_normalized_tensor_batch.set_requires_grad(false);
        let desired = bbs_to_tensor(&bb_vec, 13, 416, (70, 70), *DEVICE)
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
    use crate::yolo_nn::yolo_bbox_conversion::{bbs_from_tensor};
    use imageproc::drawing::Blend;
    use crate::yolo_nn::helpers::img_drawing::{draw_bb_to_img, draw_bb_to_img_from_file};

    #[test]
    fn test_network_works() {
        let network = DarknetConfig::new("yolo-v3_modif.cfg").unwrap();
        let (mut vs, model) = network.build_model().unwrap();
        vs.load("yolo-v3_modif_trainned.ot").unwrap();


        let label_n_images_filepath = "labelbox_dataset/test/labels.json";
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
