use std::fs::File;
use std::io::Write;

use image::{DynamicImage, FilterType, GenericImageView};
use rand::prelude::SliceRandom;
use rand::thread_rng;

use skynet::dataset::common_structs::ImgFilenameWithBboxes;
use skynet::dataset::data_transformers::labelbox::{
    labelbox_struct_to_img_filename_with_bboxes, labelbox_vec_from_exported_json_file,
    LabelBoxImageJson,
};

const RAW_SAMPLES_DIR: &str = "dataset/raw_samples";

fn main() {
    // Load labelbox exported json
    let labelbox_data =
        labelbox_vec_from_exported_json_file(&format!("{}/labelbox.json", RAW_SAMPLES_DIR));
    // Download labelbox images to folder
    download_missing_imgs_to_folder(&labelbox_data);
    // Now we split the data into train and test datasets and resize images and BBs to 416x416 pixels
    split_into_train_and_test_resizing_imgs_and_bb(labelbox_data);
}

fn split_into_train_and_test_resizing_imgs_and_bb(labelbox_data: Vec<LabelBoxImageJson>) {
    let target_size = 416;
    let train_output_path = "dataset/train";
    let test_output_path = "dataset/test";
    if File::open(train_output_path).is_ok() {
        std::fs::remove_dir_all(train_output_path).expect("Error removing old train data");
    }
    if File::open(test_output_path).is_ok() {
        std::fs::remove_dir_all(test_output_path).expect("Error removing old test data");
    }
    std::fs::create_dir_all(train_output_path).expect("Error creating train data dir");
    std::fs::create_dir_all(test_output_path).expect("Error creating test data dir");
    let mut img_n_labels: Vec<(ImgFilenameWithBboxes, DynamicImage)> = vec![];
    for label in labelbox_data {
        let img_path = format!("{}/{}", RAW_SAMPLES_DIR, label.img_filename.clone());
        let img = image::open(img_path).unwrap();
        let (ori_width, ori_height) = (img.width(), img.height());
        let width_ratio = target_size as f32 / ori_width as f32;
        let height_ratio = target_size as f32 / ori_height as f32;
        let resized = img.resize_exact(target_size, target_size, FilterType::Nearest);
        let bb = labelbox_struct_to_img_filename_with_bboxes(&label, width_ratio, height_ratio);
        img_n_labels.push((bb, resized));
    }

    img_n_labels.shuffle(&mut thread_rng());
    let (test, train) = img_n_labels.split_at(5);
    let mut test_labels = vec![];
    for (label, img) in test {
        test_labels.push(label);
        img.save(format!(
            "{}/{}",
            test_output_path,
            label.img_filename.clone()
        ))
        .expect("Error saving image");
    }
    let test_output_label_file = File::create(format!("{}/labels.json", test_output_path)).unwrap();
    serde_json::to_writer_pretty(test_output_label_file, &test_labels).unwrap();

    let mut train_labels = vec![];
    for (label, img) in train {
        train_labels.push(label);
        img.save(format!(
            "{}/{}",
            train_output_path,
            label.img_filename.clone()
        ))
        .expect("Error saving img.");
    }
    let train_output_label_file =
        File::create(format!("{}/labels.json", train_output_path)).unwrap();
    serde_json::to_writer_pretty(train_output_label_file, &train_labels).unwrap();
}
/// Download all missing files of labelbox.json into raw_samples folder
fn download_missing_imgs_to_folder(labelbox_data: &Vec<LabelBoxImageJson>) {
    let client = surf::Client::new();
    let mut file_to_download_fut = vec![];
    for label in labelbox_data {
        let file_path = format!("{}/{}", RAW_SAMPLES_DIR, label.img_filename);
        if let Err(_) = File::open(&file_path) {
            let download_url = label.labeled_data_download_url.clone();
            let download_and_save_futures = async {
                let download_url = download_url;
                println!("Downloading {}", file_path);
                let resp = client.get(&download_url).await;
                match resp {
                    Ok(mut res) => {
                        let response_bytes = res
                            .body_bytes()
                            .await
                            .expect("Error getting response bytes");
                        let mut downloaded_file = File::create(file_path).unwrap();
                        downloaded_file
                            .write(&response_bytes)
                            .expect("Error writting bytes to file");
                    }
                    Err(e) => {
                        println!("Err {}", e);
                    }
                }
            };
            file_to_download_fut.push(download_and_save_futures);
        }
    }
    let all_futs = futures::future::join_all(file_to_download_fut);
    async_std::task::block_on(all_futs);
}
