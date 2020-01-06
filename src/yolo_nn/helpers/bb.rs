use crate::dataset::common_structs::SimpleBbox;

/// TODO cleanup conversions
fn bb_max_min_x_y(bb: &SimpleBbox) -> (f32, f32, f32, f32) {
    let bb_xmax = bb.left as u32 + bb.width;
    let bb_xmin = bb.left;
    let bb_ymax = bb.top as u32 + bb.height;
    let bb_ymin = bb.top;
    (
        bb_xmax as f32,
        bb_xmin as f32,
        bb_ymax as f32,
        bb_ymin as f32,
    )
}

pub fn iou_bbs(bb1: &SimpleBbox, bb2: &SimpleBbox) -> f32 {
    let (bb1_xmax, bb1_xmin, bb1_ymax, bb1_ymin) = bb_max_min_x_y(&bb1);
    let (bb2_xmax, bb2_xmin, bb2_ymax, bb2_ymin) = bb_max_min_x_y(&bb2);

    // Intersection over union of two bounding boxes.
    let b1_area = (bb1_xmax - bb1_xmin) * (bb1_ymax - bb1_ymin);
    let b2_area = (bb2_xmax - bb2_xmin) * (bb2_ymax - bb2_ymin);
    let i_xmin = bb1_xmin.max(bb2_xmin);
    let i_xmax = bb1_xmax.min(bb2_xmax);
    let i_ymin = bb1_ymin.max(bb2_ymin);
    let i_ymax = bb1_ymax.min(bb2_ymax);
    let i_area = (i_xmax - i_xmin).max(0.) * (i_ymax - i_ymin).max(0.);
    i_area / (b1_area + b2_area - i_area + 10e-4)
}
