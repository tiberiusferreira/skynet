#[macro_use]
extern crate lazy_static;
use skynet::yolo_nn::yolo_trainer;
fn main() {
    yolo_trainer().unwrap();
}
