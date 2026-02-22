use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EnhanceModelKind {
    ParameterPredictor,
    ImageToImage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDescriptor {
    pub id: String,
    pub name: String,
    pub description: String,
    pub kind: EnhanceModelKind,
    pub version: String,
    pub builtin: bool,
    pub requires_api_key: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EnhanceResult {
    #[serde(rename_all = "camelCase")]
    Parameters {
        values: HashMap<String, f64>,
    },
    #[serde(rename_all = "camelCase")]
    Image {
        result_file_id: String,
        mime_type: String,
    },
}

#[derive(Debug, Clone)]
pub struct ImageStats {
    pub mean_luminance: f64,
    pub median_luminance: f64,
    pub stddev_luminance: f64,
    pub r_mean: f64,
    pub g_mean: f64,
    pub b_mean: f64,
    pub r_histogram: [u32; 256],
    pub g_histogram: [u32; 256],
    pub b_histogram: [u32; 256],
    pub lum_histogram: [u32; 256],
    pub percentile_1: f64,
    pub percentile_5: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
    pub shadow_clip_pct: f64,
    pub highlight_clip_pct: f64,
    pub saturation_mean: f64,
    pub color_cast: (f64, f64), // (warm/cool shift, green/magenta shift)
}
