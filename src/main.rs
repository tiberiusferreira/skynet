use std::fs::File;

use imageproc::drawing::Blend;
use rand::Rng;
use tch::nn::{Module, ModuleT, OptimizerConfig};
use tch::rust_image::resize_img;
use tch::{kind, nn, no_grad, vision, Device, IndexOp, Kind, Reduction, Tensor};

use crate::preprocessing::{
    draw_bb_to_img, from_tensor, objects_mask_tensor_from_target_tensor, to_tensor
};
use std::process::exit;
use crate::preprocessing::structs::CleanedImgLabels;
use tch::vision::image::resize;

mod preprocessing;

const NB_CLASSES: i64 = 1;

fn net(vs: &nn::Path) -> impl ModuleT {
    let conv_cfg = nn::ConvConfig {
        padding: 1,
        ..Default::default()
    };
    nn::seq_t()
        .add(nn::conv2d(vs, 3, 16, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 16, Default::default()))
        .add_fn(|x| x.leaky_relu()) // change to slope 0.1
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 16, 32, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 32, Default::default()))
        .add_fn(|x| x.leaky_relu()) // change to slope 0.1
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 32, 64, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 64, Default::default()))
        .add_fn(|x| x.leaky_relu()) // change to slope 0.1
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 64, 128, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 128, Default::default()))
        .add_fn(|x| x.leaky_relu()) // change to slope 0.1
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 128, 256, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 256, Default::default()))
        .add_fn(|x| x.leaky_relu()) // change to slope 0.1
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 256, 512, 3, conv_cfg))
        .add(nn::batch_norm2d(vs, 512, Default::default()))
        .add_fn(|x| x.leaky_relu()) // change to slope 0.1
        //        .add_fn(|x| x.max_pool2d(&[2, 2], &[1, 1], &[0, 0], &[1, 1], false))
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
    let mut opt = nn::Adam::default().build(&store, 5. * 1e-3)?;

    let label_file = File::open("processed_imgs/labels.json").unwrap();
    let labels: Vec<CleanedImgLabels> = serde_json::from_reader(label_file).unwrap();
    store.load("variables.ot").unwrap();
    for i in 0..1 {
        for label in &labels {
            let original_image = tch::vision::image::load(format!("processed_imgs/{}", label.img_filename))?;

            let img_as_batch = original_image.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;

//            println!("Input: {:?}", img_as_batch.size());
            let img_as_batch = img_as_batch.set_requires_grad(true);

            let desired = to_tensor(&label.bboxes, 13, 416, (70, 70))
                .tensor
                .unsqueeze(0);

            for i in 0..2 {
                let output = net.forward_t(&img_as_batch, true);
                let target_object_prob = desired.narrow(1, 4, 1);
                let output_object_prob = output.narrow(1, 4, 1);
                let objectness_loss = output_object_prob.binary_cross_entropy::<Tensor>(
                    &target_object_prob,
                    None,
                    Reduction::Sum,
                );

                let grid_points_with_obj =
                    objects_mask_tensor_from_target_tensor(desired.squeeze().into(), 13, 416, (70, 70));
                let mut others_loss: Tensor = Tensor::from(0.);
                for point in grid_points_with_obj {
                    let target_others = desired
                        .narrow(1, 0, 4)
                        .narrow(2, point.x, 1)
                        .narrow(3, point.y, 1);
                    let output_others = output
                        .narrow(1, 0, 4)
                        .narrow(2, point.x, 1)
                        .narrow(3, point.y, 1);

                    others_loss += output_others.mse_loss(&target_others, Reduction::Sum);
                }
                let loss = objectness_loss + others_loss;
                opt.backward_step(&loss);
                println!("Loss {:?}", loss);
            }
        }
    }

    store.save("variables.ot").unwrap();

    let original_image = tch::vision::image::load("new_samples/image0005_resized.jpg")?;

    let img_as_batch = original_image.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;
    let output = net.forward_t(&img_as_batch, true);
    let out = output.squeeze();
    let bb = from_tensor(out.into(), 13, 416, (70, 70));
//        let bb = from_tensor(desired.squeeze().into(), 13, 416, (70, 70));

    let img = image::open("new_samples/image0005_resized.jpg").unwrap();
    let mut blend = Blend(img.to_rgba());
    for single_bb in &bb {
        draw_bb_to_img(&mut blend, single_bb);
    }
    blend.0.save("maybe_worked5.jpg").unwrap();
    println!("{:?}", bb);
    //    println!("Desired {:?}", desired.size());

    Ok(())
}
