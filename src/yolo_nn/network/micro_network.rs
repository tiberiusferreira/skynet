use super::*;

pub fn micro_yolo_net(vs: &nn::Path) -> impl ModuleT {
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
