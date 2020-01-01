use serde::{Deserialize, Serialize};
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CocoDatasetJson {
    pub info: Info,
    pub licenses: Vec<License>,
    pub images: Vec<Image>,
    pub annotations: Vec<Annotation>,
    pub categories: Vec<Category>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Info {
    pub description: String,
    pub url: String,
    pub version: String,
    pub year: i64,
    pub contributor: String,
    pub date_created: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct License {
    pub url: String,
    pub id: i64,
    pub name: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Image {
    pub license: i64,
    pub file_name: String,
    pub coco_url: String,
    pub height: i64,
    pub width: i64,
    pub date_captured: String,
    pub flickr_url: String,
    pub id: i64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub segmentation: ::serde_json::Value,
    pub area: f64,
    pub iscrowd: i64,
    pub image_id: i64,
    pub bbox: Vec<f64>,
    pub category_id: i64,
    pub id: i64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Category {
    pub supercategory: String,
    pub id: i64,
    pub name: String,
}
