use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Request body for POST /api/v1/optimize/run
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizeRequest {
    pub file_id: String,
    pub strength: Option<f64>,
    pub denoise: Option<bool>,
    pub enhance: Option<bool>,
    pub masks: Option<bool>,
}

/// Normalized configuration derived from OptimizeRequest
#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    /// Strength 0.0–1.0 (default 0.5)
    pub strength: f64,
    /// Enable AI denoise pass
    pub denoise: bool,
    /// Enable HDRNet enhance pass
    pub enhance: bool,
    /// Enable smart mask computation
    pub masks: bool,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            strength: 0.5,
            denoise: true,
            enhance: true,
            masks: true,
        }
    }
}

impl From<&OptimizeRequest> for OptimizeConfig {
    fn from(req: &OptimizeRequest) -> Self {
        Self {
            strength: req.strength.unwrap_or(50.0) / 100.0,
            denoise: req.denoise.unwrap_or(true),
            enhance: req.enhance.unwrap_or(true),
            masks: req.masks.unwrap_or(true),
        }
    }
}

/// Individual mask data for a detected region
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskData {
    pub mask_type: String,
    pub width: u32,
    pub height: u32,
    #[serde(skip)]
    pub data: Vec<f32>,
}

/// Set of computed masks from segmentation models
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskSet {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<MaskData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sky: Option<MaskData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skin: Option<MaskData>,
}

impl Default for MaskSet {
    fn default() -> Self {
        Self {
            subject: None,
            sky: None,
            skin: None,
        }
    }
}

/// Result of the optimize pipeline
#[derive(Debug)]
pub struct OptimizeResult {
    pub image_data: Vec<u8>,
    pub mime_type: String,
    pub masks: MaskSet,
    pub applied_params: HashMap<String, f64>,
}

/// Status of an individual optimize model
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizeModelStatus {
    pub id: String,
    pub name: String,
    pub available: bool,
    pub file_size_mb: Option<f64>,
}

/// Tile configuration for tiled inference
#[derive(Debug, Clone)]
pub struct TileConfig {
    pub size: u32,
    pub overlap: u32,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            size: 512,
            overlap: 32,
        }
    }
}

/// Model manifest for optimize models (extends existing manifest pattern)
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
#[allow(dead_code)]
pub struct OptimizeModelManifest {
    pub id: String,
    pub name: String,
    pub publisher: String,
    pub description: String,
    pub adapter: String,
    pub file: String,
    #[serde(default)]
    pub input_size: Option<[u32; 2]>,
    #[serde(default)]
    pub normalize: Option<OptimizeNormalizeConfig>,
    #[serde(default)]
    pub tile_size: Option<u32>,
    #[serde(default)]
    pub tile_overlap: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OptimizeNormalizeConfig {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// Serializable result for job completion
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizeResultData {
    pub result_file_id: String,
    pub applied_params: HashMap<String, f64>,
    pub masks: OptimizeMasksSummary,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizeMasksSummary {
    pub subject: bool,
    pub sky: bool,
    pub skin: bool,
}

impl From<&MaskSet> for OptimizeMasksSummary {
    fn from(masks: &MaskSet) -> Self {
        Self {
            subject: masks.subject.is_some(),
            sky: masks.sky.is_some(),
            skin: masks.skin.is_some(),
        }
    }
}
