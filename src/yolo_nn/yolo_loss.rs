use super::*;
use crate::dataset::data_transformers::bbox_conversion::objects_mask_tensor_from_target_tensor;

pub fn yolo_loss(desired: Tensor, output: Tensor) -> Tensor {
    let target_object_prob = desired.narrow(1, 4, 1);
    let output_object_prob = output.narrow(1, 4, 1);
    let objectness_loss = output_object_prob.binary_cross_entropy::<Tensor>(
        &target_object_prob,
        None,
        Reduction::Mean,
    );

    let (batch_size, _, _, _) = desired.size4().unwrap();
    let mut others_loss: Tensor = Tensor::from(0.).to_device(DEVICE);
    for prediction_index in 0..batch_size{
        // TODO objects_mask_tensor_from_target_tensor assumes it take only one sample, not a batch
        let grid_points_with_obj = objects_mask_tensor_from_target_tensor(desired.i(prediction_index).into(), 13);
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
    others_loss = others_loss/batch_size;
    objectness_loss + others_loss
}
