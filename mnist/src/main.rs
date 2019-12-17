use tch::nn::{Module, OptimizerConfig, ModuleT};
use tch::vision::imagenet::save_image;
use tch::{kind, nn, no_grad, vision, Device, Kind, Tensor};
use rand::Rng;

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES: i64 = 128;
const LABELS: i64 = 10;

fn net(vs: &nn::Path) -> impl ModuleT {
//    let mut vs = nn::VarStore::new(Device::Cpu);

    nn::seq_t()
        .add_fn(|x| x.view([-1, 1, 28, 28]))
        .add(nn::conv2d(vs, 1, 32, 5, Default::default()))
        .add_fn(|x| x.max_pool2d_default(2))
        .add(nn::conv2d(vs, 32, 64, 5, Default::default()))
        .add_fn(|x| x.max_pool2d_default(2))
        .add_fn(|x| x.view([-1, 1024]))
        .add(nn::linear(vs,1024,1024,Default::default()))
        .add_fn(|x| x.relu())
        .add_fn_t(|x, train| x.dropout(0.5, train))
        .add(nn::linear(vs,1024,10,Default::default()))
}
fn main() -> failure::Fallible<()> {

    let dataset = vision::mnist::load_dir("data").unwrap();
    println!("train-images: {:?}", dataset.train_images.size());
    println!("train-labels: {:?}", dataset.train_labels.size());
    println!("test-images: {:?}", dataset.test_images.size());
    println!("test-labels: {:?}", dataset.test_labels.size());

    let mut store = nn::VarStore::new(Device::Cpu);

    let net = net(&store.root());
    store.load("variables").unwrap();
    let mut opt = nn::Adam::default().build(&store, 1e-3)?;

//    let test_accuracy = net
//        .forward_t(&dataset.test_images, false)
//        .accuracy_for_logits(&dataset.test_labels);
//    println!("test acc: {:5.2}%", 100. * f64::from(&test_accuracy));

    //    save_image(&ten, "test.jpg").unwrap();
    let batch_size = 256*1;
    for epoch in 1..=20 {
        let mut loss= Tensor::from(0);
        for (index, (bimages, blabels)) in dataset.train_iter(batch_size).shuffle().enumerate() {
             loss = net
                .forward_t(&bimages, true)
                .cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
            println!("{} of {}", index, dataset.train_images.size().get(0).unwrap()/batch_size)
        }
        let test_accuracy = net
            .forward_t(&dataset.test_images, false)
            .accuracy_for_logits(&dataset.test_labels);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        );
    }

    let mut rng = rand::thread_rng();
    let index = rng.gen_range(0, 10_000-1);
    println!("Index = {}", index);
    let ten = dataset.test_images.get(index).reshape(&[28, 28]);
    for i in 0..ten.size()[0] {
        for j in 0..ten.size()[1] {
            print!(
                "{0:4} ",
                (ten.get(i).get(j).double_value(&[]) * 255.0) as u32
            );
        }
        println!();
    }
    let pred = net.forward_t(&dataset.test_images.get(index), false).argmax(-1, false);
    println!("predicted = {}", pred.double_value(&[]));

    store.save("variables").unwrap();

    Ok(())
}
