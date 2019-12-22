use image::{DynamicImage, ImageBuffer};
use tch::vision::image::save;
use tch::{Device, Kind, Tensor};
use crate::yolo_nn::DEVICE;
pub mod augmentation;
pub mod bbox_conversion;
pub mod dataset;
pub mod structs;


pub fn from_img_to_tensor(img: &DynamicImage) -> Tensor {
    let rbg = img.to_rgb();
    let width = rbg.width();
    let height = rbg.height();
    let tensor = tch::Tensor::zeros(
        &[3, width as i64, height as i64],
        (Kind::Uint8, DEVICE),
    );
    let raw_data_vec = rbg.clone().into_raw();
    let img_as_tensor = Tensor::of_data_size(
        raw_data_vec.as_slice(),
        &[width as i64, height as i64, 3],
        Kind::Uint8,
    )
    .transpose(0, 2)
    .transpose(1, 2);
    img_as_tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::IndexOp;

    #[test]
    fn test_to_tensor() {
        let img_path = "code_test_data/test_img.jpg";
        let image_as_tensor = tch::vision::image::load(img_path).unwrap();
        let (channels, width, height) = image_as_tensor.size3().unwrap();
        let image_as_dyn = image::open(img_path).unwrap();

        let img_converted_to_tensor = from_img_to_tensor(&image_as_dyn);

        for ch in 0..channels {
            for x in 0..width {
                for y in 0..height {
                    // accept up to 2 (of 255) of divergence between pytorch and our conversion
                    assert!(
                        (image_as_tensor.i(ch).i(x).i(y).double_value(&[])
                            - img_converted_to_tensor.i(ch).i(x).i(y).double_value(&[]))
                        .abs()
                            < 2.
                    );
                }
            }
        }
    }
}
