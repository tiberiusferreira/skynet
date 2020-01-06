use imageproc::drawing::{Blend, Canvas};
use image::{RgbaImage, DynamicImage};
use crate::dataset::common_structs::SimpleBbox;

pub fn draw_bb_to_img(img: &mut DynamicImage, bb: &SimpleBbox) {
    draw_bb_to_img_with_color(img, bb, [255, 0, 0, 90]);
}

pub fn draw_bb_to_img_with_color(img: &mut DynamicImage, bb: &SimpleBbox, rgba_color: [u8; 4]) {
    let mut img_blend = Blend(img.to_rgba());
    let rec = imageproc::rect::Rect::at(bb.left as i32, bb.top as i32)
        .of_size(bb.width as u32, bb.height as u32);

    let color = image::Rgba(rgba_color);

    imageproc::drawing::draw_hollow_rect_mut(&mut img_blend, rec, color);
    std::mem::swap(img, &mut DynamicImage::ImageRgba8(img_blend.0));
}

pub fn draw_bb_to_img_from_file(img_path: &str, out_path: &str, bbs: &Vec<SimpleBbox>) {
    let mut img = image::open(img_path).unwrap();
    for bb in bbs {
        draw_bb_to_img(&mut img, bb);
    }
    img.save(out_path).expect("Error saving img with BB");
}



pub fn draw_grid_to_img(img: &mut DynamicImage, grid_size: u32) {
    let mut img_blend = Blend(img.to_rgba());
    let (img_width, img_height) = img_blend.dimensions();
    let single_cell_width = img_width as f32/grid_size  as f32;
    let color = image::Rgba([255, 0, 0, 255]);

    for i in 0..=grid_size{
        // vertical lines
        let x = i as f32 *single_cell_width;
        let y_start = 0.;
        let y_end = img_height as f32;
        imageproc::drawing::draw_line_segment_mut(&mut img_blend, (x, y_start), (x, y_end), color);

        // horizontal lines
        let y = i as f32 *single_cell_width;
        let x_start = 0.;
        let x_end = img_width as f32;
        imageproc::drawing::draw_line_segment_mut(&mut img_blend, (x_start, y), (x_end, y), color);
    }
    std::mem::swap(img, &mut DynamicImage::ImageRgba8(img_blend.0));
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_DATA_DIR: &str = "code_test_data";

    #[test]
    fn test_draw_grid_to_img() {
        let mut img = image::open(format!("{}/test_img.jpg", TEST_DATA_DIR)).unwrap();
        draw_grid_to_img(&mut img, 5);
        img.save(format!("{}/img_grid_5_5.jpg", TEST_DATA_DIR)).unwrap();
    }
}