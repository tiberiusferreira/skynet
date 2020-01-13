use super::*;
use crate::yolo_nn::network::YoloNetworkOutput;
use crate::yolo_nn::yolo_bbox_conversion::{objects_mask_tensor_from_target_tensor, bb_to_yolo_norm_coords, GridPointsWithObjects};
use crate::yolo_nn::helpers::bb::iou_bbs;


#[derive(Clone, Debug)]
struct GridXYIoU {
    pub grids_to_the_left_of_bb_center: u32,
    pub grids_above_of_bb_center: u32,
    // Each anchor IOU and the anchor BB in real pixel values according to original image size
    pub anchors_iou_bb: Vec<(f32, SimpleBbox)>,
    // for 3 anchors should be 0, 1 or 2
    pub best_anchor_index: usize,
    pub index_anchors_iou_above_50_but_not_best: Vec<(usize)>,
}

fn get_grid_x_y_iou(
    bb: &SimpleBbox,
    original_img_size: u32,
    grid_size: u32,
    anchors: Vec<(i64, i64)>,
) -> GridXYIoU {
    let grid_cell_size_pixels = original_img_size / grid_size;
    let bb_left_center_pixels = bb.left as u32 + bb.width / 2;
    let bb_top_center_pixels = bb.top as u32 + bb.height / 2;
    let grids_to_the_left_of_bb_center =
        (bb_left_center_pixels as f32 / grid_cell_size_pixels as f32).floor() as u32;
    let grids_above_of_bb_center =
        (bb_top_center_pixels as f32 / grid_cell_size_pixels as f32).floor() as u32;
    let grid_x_center_pixels =
        ((grids_to_the_left_of_bb_center as f32 + 0.5) * grid_cell_size_pixels as f32) as i32;
    let grid_y_center_pixels =
        ((grids_above_of_bb_center as f32 + 0.5) * grid_cell_size_pixels as f32) as i32;
    let anchors_iou: Vec<(f32, SimpleBbox)> = anchors
        .clone()
        .into_iter()
        .map(|(width, height)| {
            let anchor_as_bb = (SimpleBbox {
                top: (grid_y_center_pixels - (height / 2) as i32).max(0),
                left: (grid_x_center_pixels - (width / 2) as i32).max(0),
                height: height as u32,
                width: width as u32,
                prob: 0.0,
                class: 0,
            });

            (iou_bbs(&bb, &anchor_as_bb), anchor_as_bb)
        })
        .collect();

    let mut max_iou = 0.;
    let mut max_index = 0;
    let mut index_anchors_iou_above_50_but_not_best = vec![];
    for (index, iou_bb) in anchors_iou.iter().enumerate(){
        if iou_bb.0 > max_iou{
            max_iou = iou_bb.0;
            max_index = index;
        }
    }

    for (index, iou_bb) in anchors_iou.iter().enumerate(){
        if iou_bb.0 > 0.5 && index!=max_index{
            index_anchors_iou_above_50_but_not_best.push(index);
        }
    }

    GridXYIoU {
        grids_to_the_left_of_bb_center,
        grids_above_of_bb_center,
        anchors_iou_bb: anchors_iou,
        best_anchor_index: max_index,
        index_anchors_iou_above_50_but_not_best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yolo_nn::helpers::img_drawing::{draw_bb_to_img, draw_bb_to_img_with_color, draw_grid_to_img};

    #[test]
    fn test_iou_bbs() {
        let bb = SimpleBbox {
            top: 100,
            left: 100,
            height: 50,
            width: 50,
            prob: 0.0,
            class: 0,
        };
        let independent_bb = SimpleBbox {
            top: 150,
            left: 150,
            height: 50,
            width: 50,
            prob: 0.0,
            class: 0,
        };
        let half_vertical_overlap_bb = SimpleBbox {
            top: 75,
            left: 100,
            height: 50,
            width: 50,
            prob: 0.0,
            class: 0,
        };
        let half_horizontal_overlap_bb = SimpleBbox {
            top: 100,
            left: 75,
            height: 50,
            width: 50,
            prob: 0.0,
            class: 0,
        };
        let mut img = image::open("code_test_data/test_img.jpg").unwrap();
        // For debugging
//        helpers::img_drawing::draw_bb_to_img_with_color(&mut img, &bb, [255, 0, 0, 100]);
//        helpers::img_drawing::draw_bb_to_img_with_color(&mut img, &independent_bb, [0, 255, 0, 100]);
//        helpers::img_drawing::draw_bb_to_img_with_color(&mut img, &half_vertical_overlap_bb, [0, 0, 255, 100]);
//        helpers::img_drawing::draw_bb_to_img_with_color(&mut img, &half_horizontal_overlap_bb, [0, 0, 255/2, 100]);
        img.save("code_test_data/bb_img.jpg").unwrap();
        assert_eq!(iou_bbs(&bb, &independent_bb), 0.);
        assert!((iou_bbs(&bb, &bb) - 1.).abs() < 0.01);
        assert!((iou_bbs(&bb, &half_vertical_overlap_bb) - 0.3333).abs() < 0.01);
        assert!((iou_bbs(&bb, &half_horizontal_overlap_bb) - 0.3333).abs() < 0.01);
    }

    #[test]
    fn test_get_grid_x_y_iou() {
        let mut img = image::open("code_test_data/test_img.jpg").unwrap();
        let bb = SimpleBbox {
            top: 100,
            left: 110,
            height: 50,
            width: 50,
            prob: 0.0,
            class: 0,
        };
        let original_img_size = 416;
        let grid_size = 5;
        let anchors = vec![(62,45),  (59,119),  (116,90),  (156,198),  (373,326)]; // 116,90,  156,198,  373,326
        draw_bb_to_img_with_color(&mut img, &bb, [255, 0, 0, 255]);
        draw_grid_to_img(&mut img, 5);
        let grid_x_y_iou = get_grid_x_y_iou(&bb, original_img_size, grid_size, anchors);
        for (iou, anchor_bb) in &grid_x_y_iou.anchors_iou_bb{
            draw_bb_to_img_with_color(&mut img, anchor_bb, [0, 255, 0, (255.*(*iou)) as u8]);
            println!("{:?} {:?}", iou, anchor_bb);
        }
        assert_eq!(grid_x_y_iou.anchors_iou_bb[0].0, 0.6202142);
        img.save("code_test_data/img_test_get_grid_x_y_iou.jpg").unwrap();
    }


    #[test]
    fn test_playground(){
        let mut t = tch::Tensor::zeros(&[3, 3, 3], (Kind::Float, Device::Cpu));

        let mut counter = 0;
        for i in 0..3{
            for j in 0..3 {
                t.i(i).i(j).copy_(&Tensor::from(counter));
                counter += 1;
            }
        }
//        t.print();

        let mask = tch::Tensor::ones(&[3], (Kind::Bool, Device::Cpu));
//        mask.i(1).i(2).copy_(&Tensor::from(false));
//        mask.i(2).i(0).copy_(&Tensor::from(false));

        let t = t.masked_select(&mask);
        println!("{:?}", t.size());
//        let t = t.view([-1, 3]);
        t.print();
        exit(0);

//        t.i(1).i(0).copy_(&Tensor::from(10.));

        let t_mask = tch::Tensor::zeros(&[3, 3], (Kind::Bool, Device::Cpu));
        let mask = [false, false, false, true];

        t_mask.i(1).i(0).copy_(&Tensor::from(true));
//        t_mask.i(1).i(0).copy_(&Tensor::from(true));
        let mut t_sel = t.masked_select(&t_mask);

        let size = 20000;
        let mut t_1 =  tch::Tensor::zeros(&[size], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let start = std::time::Instant::now();

        let t2 = t_1.shallow_clone()*10;
//        t_1*10;

        t_1.i(1).backward();

        println!("Took {}ms", start.elapsed().as_millis());

        let mut t_2 =  tch::Tensor::zeros(&[size], (Kind::Float, Device::Cpu));
        let start = std::time::Instant::now();
        for i in 0..size{
//            let value = Tensor::from(10);
            t_2.i(i).copy_(&Tensor::from(10));// += value.shallow_clone();
        }
        println!("Second took {}ms", start.elapsed().as_millis());
        println!("{}", t_1.i(523).double_value(&[]));
        println!("{}", t_2.i(523).double_value(&[]));
//        println!("Test!!");
    }

}

#[derive(Clone, Debug)]
struct BboxWithGridXyIoU {
    bb: SimpleBbox,
    grid_xy_iou: GridXYIoU,
}

fn single_grid_loss(features_tensor: Tensor, original_img_size: u32, grid_size: u32, anchors: Vec<(i64, i64)>, obj_in_this_grid: Option<&BboxWithGridXyIoU>, device: Device) -> (Tensor, Tensor) {

    let nb_anchors = 3;
    let nb_features = features_tensor.size1().expect("features tensor is not of Rank 1");
    let features_per_anchor = nb_features/nb_anchors;
    let nb_classes = (features_per_anchor) - 5;

    let start = std::time::Instant::now();
    let features_tensor_3_85 = features_tensor.reshape(&[nb_anchors, features_per_anchor]);
//    println!("Reshape costs: {}ms", start.elapsed().as_nanos()/1000); // ~ 4ms

    let start = std::time::Instant::now();
    let mut objectness_loss = Tensor::from(0.).to_device(device);
    let mut total_loss = Tensor::from(0.).to_device(device);
    match obj_in_this_grid {
        Some(object) => {
            // get best anchor
            let iou_vec = &object.grid_xy_iou.anchors_iou_bb;
            let mut best_anchor_index = 0;
            let mut tmp_best_iou = 0.;
            for (i, (iou, _bb)) in iou_vec.iter().enumerate() {
                if *iou > tmp_best_iou {
                    best_anchor_index = i;
                    tmp_best_iou = *iou;
                }
            }
            for anchor_index in 0..nb_anchors as usize{
                let output_tensor_for_anchor_85 = features_tensor_3_85.i(anchor_index as i64);
                if anchor_index == best_anchor_index {
                    // do full loss for this anchor
                    let output_tensor_for_anchor_85 = features_tensor_3_85.i(anchor_index as i64);
                    // Class Prob Loss
                    let output_tensor_classes_80 = output_tensor_for_anchor_85.narrow(0, 5, nb_classes);
                    let classes_prob = Tensor::zeros(&[nb_classes], (Kind::Float, device));
                    classes_prob.i(object.bb.class as i64).copy_(&Tensor::from(1.).to_device(device));
//                    println!("Expected class tensor/Actual");
//                    classes_prob.print();
//                    output_tensor_classes_80.print();
                    let class_loss = output_tensor_classes_80.binary_cross_entropy::<Tensor>(&classes_prob, None, Reduction::Mean);
//                    class_loss.print();
                    // Position Loss
                    let anchor = anchors[anchor_index];
                    let desired_coords = bb_to_yolo_norm_coords(&object.bb, grid_size, original_img_size, (anchor.0 as u32, anchor.1 as u32), device);

                    let output_coords = output_tensor_for_anchor_85.narrow(0, 0, 4);
                    let coords_loss = output_coords.mse_loss(&desired_coords, Reduction::Mean);
//                    println!("Coords loss");
//                    coords_loss.print();
                    // Objectness Loss
                    let objectness_loss = output_tensor_for_anchor_85.i(4).mse_loss(&Tensor::from(1.).to_device(device).to_kind(Kind::Float), Reduction::Mean);
//                    println!("Obj loss");
//                    objectness_loss.print();
                    /// TODO, check if should be this way or without avg (/3)
                    let local_loss = (objectness_loss + coords_loss + class_loss);
//                    println!("Total loss = ");
//                    local_loss.print();
                    total_loss += local_loss;
                } else if iou_vec[anchor_index].0 >= 0.5 {
                    // IOU >= 0.5, but not best
                    continue;
                } else {
                    // IOU < 0.5, only objectness loss
                    let local_objectness_loss = output_tensor_for_anchor_85.i(4).mse_loss(&Tensor::from(0.).to_device(device).to_kind(Kind::Float), Reduction::Mean);
//                    println!("IOU < 0.5 loss");
//                    objectness_loss.print();
                    total_loss += local_objectness_loss;
                }
            }
//            println!("Grid with Obj loss = {}", total_loss.double_value(&[]));
        }
        None => {
            for anchor_index in 0..nb_anchors {
                let local_objectness_loss = features_tensor_3_85
                    .i(anchor_index as i64).i(4)
                    .mse_loss(&Tensor::from(0.).to_kind(Kind::Float), Reduction::Mean);
                if local_objectness_loss.double_value(&[]) > 0.5{
//                    println!("Objectness loss = {}", objectness_loss.double_value(&[]));
                }
                total_loss += local_objectness_loss.shallow_clone();
                objectness_loss += local_objectness_loss;
            }
        }
    }
//    println!("Rest costs: {}ms", start.elapsed().as_nanos()/1000); //~125ms
    (total_loss, objectness_loss)
}
pub fn yolo_loss2(
    ground_truth: &Vec<SimpleBbox>,
    network_output: &YoloNetworkOutput,
    original_img_size: u32,
    device: Device
) -> Tensor {
    let tensor = network_output.single_scale_output.shallow_clone().to_device(device);
    let (grid_width, grid_height, nb_features) =
        tensor.size3().expect("Expected tensor to have Rank 3");
    let features_per_anchor = nb_features/3;
    let nb_classes = (features_per_anchor) - 5;

    let grid_size = grid_width as u32; // we assume grid_width = grid_height

    let mut bbox_with_grid_xy_iou = vec![];
    for bb in ground_truth {
        let grid_xy_iou = get_grid_x_y_iou(
            bb,
            original_img_size,
            grid_size,
            network_output.anchor_boxes.clone(),
        );
        bbox_with_grid_xy_iou.push(BboxWithGridXyIoU {
            bb: bb.clone(),
            grid_xy_iou,
        });
    }
//    println!("{:#?}", bbox_with_grid_xy_iou);
    // Ok so here we have the X and Y of the objects in the grid, and each anchor IOU
    // 1 - when obj and best IOU                        -> all losses
    // 2 - when obj and anchor IOU > 0.5 but not best   -> no loss
    // 3 - when obj and anchor IOU < 0.5                -> only objectness loss
    // Ok so result is [13, 13, 3, 85]
    // 2 Masks: for 1 and 3
    // but to apply losses to 1 we need to know the order in which the objects are returned
    // Could create a select mask for objs [13, 13, 3, 85]

    let tensor_3 = tensor.reshape(&[grid_width, grid_height, 3, nb_features/3]);
    // No Objs mask
    let no_obj_mask = Tensor::ones(&[grid_width, grid_height, 3, nb_features/3], (Kind::Bool, device));
    let false_tensor = Tensor::from(false).to_device(device);
    for bb in &bbox_with_grid_xy_iou{
        let x = bb.grid_xy_iou.grids_to_the_left_of_bb_center as i64;
        let y = bb.grid_xy_iou.grids_above_of_bb_center as i64;
        let best_anchor_index = bb.grid_xy_iou.best_anchor_index as i64;
        let grid_el = no_obj_mask.i(y).i(x);
        grid_el.i(best_anchor_index).copy_(&false_tensor);
        for bb in &bb.grid_xy_iou.index_anchors_iou_above_50_but_not_best{
            grid_el.i(*bb as i64).copy_(&false_tensor);
        }
    }

    let no_obj_loss = tensor_3.masked_select(&no_obj_mask).reshape(&[-1, nb_features/3]);
    let only_objectness = no_obj_loss.narrow(1, 4, 1);

    let (elements, size) = only_objectness.size2().unwrap();
    let target_objectness = Tensor::zeros(&[elements, size], (Kind::Float, device));

    let new_loss = only_objectness.mse_loss(&target_objectness, Reduction::Mean);

    let mut total_time = 0;
    let mut total_existing_obj_loss = Tensor::from(0.).to_device(device);


    for bb in &bbox_with_grid_xy_iou{
        let x = bb.grid_xy_iou.grids_to_the_left_of_bb_center as i64;
        let y = bb.grid_xy_iou.grids_above_of_bb_center as i64;
        let features_tensor = tensor.i(y as i64).i(x as i64);
        let grid_el_loss = single_grid_loss(features_tensor, original_img_size, grid_size, network_output.anchor_boxes.clone(), Some(bb), device);
        total_existing_obj_loss += grid_el_loss.0;
    }
    let nb_objs = bbox_with_grid_xy_iou.len() as i64;
    let nb_objs_predictions = nb_objs*network_output.anchor_boxes.len() as i64;
    total_existing_obj_loss = total_existing_obj_loss/nb_objs_predictions;

    total_existing_obj_loss + new_loss
}

pub fn yolo_loss(desired: Tensor, output: Tensor) -> Tensor {
    let target_object_prob = desired.narrow(1, 4, 1);

    let output_object_prob = output.narrow(1, 4, 1);

    let objectness_loss = output_object_prob.binary_cross_entropy::<Tensor>(
        &target_object_prob,
        None,
        Reduction::Mean,
    );

    let (batch_size, _, _, _) = desired.size4().unwrap();
    let mut others_loss: Tensor = Tensor::from(0.).to_device(*DEVICE);
    for prediction_index in 0..batch_size {
        // TODO objects_mask_tensor_from_target_tensor assumes it take only one sample, not a batch
        let grid_points_with_obj =
            objects_mask_tensor_from_target_tensor(desired.i(prediction_index).into(), 13);
        for point in grid_points_with_obj {
            let target_others = desired
                .narrow(0, prediction_index, 1)
                .narrow(1, 0, 4)
                .narrow(2, point.x, 1)
                .narrow(3, point.y, 1);
            let output_others = output
                .narrow(0, prediction_index, 1)
                .narrow(1, 0, 4)
                .narrow(2, point.x, 1)
                .narrow(3, point.y, 1);

            others_loss += output_others.mse_loss(&target_others, Reduction::Sum);
        }
    }
    others_loss = others_loss / batch_size;
    objectness_loss + others_loss
}
