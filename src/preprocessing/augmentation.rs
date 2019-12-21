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
