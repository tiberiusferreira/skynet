use skynet::yolo_nn::yolo_trainer;
use tch::IndexOp;

fn main() {
    yolo_trainer().unwrap();
}
