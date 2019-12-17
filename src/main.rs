mod preprocessing;
use rand::Rng;
use tch::nn::{Module, ModuleT, OptimizerConfig};
use tch::rust_image::resize_img;
use tch::vision::image;
use tch::vision::imagenet::save_image;
use tch::{kind, nn, no_grad, vision, Device, Kind, Reduction, Tensor};

const NB_CLASSES: i64 = 1;

fn net(vs: &nn::Path) -> impl ModuleT {
    let conv_cfg = nn::ConvConfig {
        padding: 1,
        ..Default::default()
    };
    nn::seq_t()
        .add(nn::conv2d(vs, 3, 16, 3, conv_cfg))
        .add_fn(|x| x.leaky_relu()) // change to slope 0.1
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 16, 32, 3, conv_cfg))
        .add_fn(|x| x.leaky_relu()) // change to slope 0.1
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 32, 64, 3, conv_cfg))
        .add_fn(|x| x.leaky_relu()) // change to slope 0.1
        .add_fn(|x| x.max_pool2d_default(2))
        // Output is (x, y, w, h, object_prob, class1_prob, class2_prob, ...)
        .add(nn::conv2d(vs, 64, NB_CLASSES + 5, 1, Default::default()))
        .add_fn(|x| x.shallow_clone()) // Linear activation
        // normalize output of probabilities
        .add_fn(|x| {
            let (batch, features, _, _) = x.size4().unwrap();
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

    let original_image = image::load("test.jpg")?;

    let resized_img = image::resize(&original_image, 416, 416).unwrap();
    let img_as_batch = resized_img.unsqueeze(0).to_kind(tch::Kind::Float) / 255.;

    println!("Input: {:?}", img_as_batch.size());
    let img_as_batch = img_as_batch.set_requires_grad(true);

    let desired = Tensor::zeros(&[1, 6, 52, 52], (Kind::Float, Device::Cpu));

    for i in 0..5 {
        let output = net.forward_t(&img_as_batch, true);
        let loss = output.mse_loss(&desired, Reduction::Sum);
        opt.backward_step(&loss);
        //        println!("Output {:?}", output.size());
        println!("Loss {:?}", loss);
    }
    let output = net.forward_t(&img_as_batch, true);
    output
        .narrow(1, 0, 6)
        .narrow(2, 10, 1)
        .narrow(3, 10, 1)
        .print();
    println!("Desired {:?}", desired.size());

    Ok(())
}
