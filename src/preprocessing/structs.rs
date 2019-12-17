use serde::{Serialize, Deserialize};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileLabel {
    #[serde(rename = "ID")]
    pub id: String,
    #[serde(rename = "DataRow ID")]
    pub data_row_id: String,
    #[serde(rename = "Labeled Data")]
    pub labeled_data: String,
    #[serde(rename = "Label")]
    pub label: Label,
    #[serde(rename = "Created By")]
    pub created_by: String,
    #[serde(rename = "Project Name")]
    pub project_name: String,
    #[serde(rename = "Created At")]
    pub created_at: String,
    #[serde(rename = "Updated At")]
    pub updated_at: String,
    #[serde(rename = "Seconds to Label")]
    pub seconds_to_label: f64,
    #[serde(rename = "External ID")]
    pub img_filename: String,
    #[serde(rename = "Agreement")]
    pub agreement: serde_json::Value,
    #[serde(rename = "Benchmark Agreement")]
    pub benchmark_agreement: serde_json::Value,
    #[serde(rename = "Benchmark ID")]
    pub benchmark_id: serde_json::Value,
    #[serde(rename = "Benchmark Reference ID")]
    pub benchmark_reference_id: serde_json::Value,
    #[serde(rename = "Dataset Name")]
    pub dataset_name: String,
    #[serde(rename = "Reviews")]
    pub reviews: Vec<serde_json::Value>,
    #[serde(rename = "View Label")]
    pub view_label: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Label {
    pub objects: Vec<Object>,
    pub classifications: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Object {
    #[serde(rename = "featureId")]
    pub feature_id: String,
    #[serde(rename = "schemaId")]
    pub schema_id: String,
    pub title: String,
    pub value: String,
    pub color: String,
    pub bbox: RawBbox,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawBbox {
    pub top: i32,
    pub left: i32,
    pub height: u32,
    pub width: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bbox {
    pub top: i32,
    pub left: i32,
    pub height: u32,
    pub width: u32,
    pub prob: f64,
    pub class: String,
}


#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CleanedImgLabels {
    pub img_filename: String,
    pub bboxes: Vec<Bbox>
}