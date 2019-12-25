use image::DynamicImage;
use rand::{thread_rng, Rng};

// Reasonable values are -30 and +30, max is 255, min is 0
pub fn random_change_brightness(img: &DynamicImage, min: i32, max: i32) -> DynamicImage {
    let value = thread_rng().gen_range(min, max);
    img.brighten(value)
}

// Reasonable values are -10 and +10
pub fn random_change_contrast(img: &DynamicImage, min: f32, max: f32) -> DynamicImage {
    let value = thread_rng().gen_range(min, max);
    img.adjust_contrast(value)
}

// Reasonable values are -30 and +30
pub fn random_hue_rotation(img: &DynamicImage, min: i32, max: i32) -> DynamicImage {
    let value = thread_rng().gen_range(min, max);
    img.huerotate(value)
}

// Reasonable values are 0.5 to 1.0
pub fn random_blur(img: &DynamicImage, min: f32, max: f32) -> DynamicImage {
    let value = thread_rng().gen_range(min, max);
    img.blur(value)
}

// Turns 1 image into 5
pub fn default_augmentation(img: DynamicImage) -> Vec<DynamicImage> {
    let bright = random_change_brightness(&img, -40, 40);
    let contrast = random_change_contrast(&img, -15., 15.);
    let hue = random_hue_rotation(&img, -35, 35);
    let blurred = random_blur(&img, 0.5, 1.0);
    vec![img, bright, contrast, hue, blurred]
}

//// TODO Make it more flexible in respect to which augmentations to apply
//pub struct ImageAugmenter;
//
//
//impl DataTransformer for ImageAugmenter{
//    type InputData = DynamicImage;
//    type OutputData = Vec<DynamicImage>;
//
//    fn transform_data(&mut self, data: Self::InputData) -> Self::OutputData {
//        default_augmentation(data)
//    }
//}
