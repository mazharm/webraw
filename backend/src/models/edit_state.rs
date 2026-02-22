use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditState {
    pub schema_version: u32,
    pub asset_id: String,
    pub global: GlobalAdjustments,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub film_sim: Option<FilmSimState>,
    #[serde(default)]
    pub local_adjustments: Vec<LocalAdjustment>,
    #[serde(default)]
    pub ai_layers: Vec<AiLayer>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<History>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snapshots: Option<Vec<Snapshot>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlobalAdjustments {
    #[serde(default)]
    pub exposure: f64,
    #[serde(default)]
    pub contrast: f64,
    #[serde(default)]
    pub highlights: f64,
    #[serde(default)]
    pub shadows: f64,
    #[serde(default)]
    pub whites: f64,
    #[serde(default)]
    pub blacks: f64,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub tint: f64,
    #[serde(default)]
    pub vibrance: f64,
    #[serde(default)]
    pub saturation: f64,
    #[serde(default)]
    pub texture: f64,
    #[serde(default)]
    pub clarity: f64,
    #[serde(default)]
    pub dehaze: f64,
    #[serde(default)]
    pub tone_curve: ToneCurve,
    #[serde(default)]
    pub hsl: HashMap<String, HslValues>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grading: Option<ColorGrading>,
    #[serde(default)]
    pub sharpening: Sharpening,
    #[serde(default)]
    pub denoise: Denoise,
    #[serde(default)]
    pub optics: Optics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crop: Option<Crop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transform: Option<Transform>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effects: Option<Effects>,
}

fn default_temperature() -> f64 {
    5500.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToneCurve {
    pub mode: String,
    #[serde(default)]
    pub points: Vec<CurvePoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parametric: Option<ParametricCurve>,
}

impl Default for ToneCurve {
    fn default() -> Self {
        Self {
            mode: "POINT".to_string(),
            points: vec![
                CurvePoint { x: 0.0, y: 0.0 },
                CurvePoint { x: 1.0, y: 1.0 },
            ],
            parametric: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvePoint {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricCurve {
    pub highlights: f64,
    pub lights: f64,
    pub darks: f64,
    pub shadows: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HslValues {
    pub h: f64,
    pub s: f64,
    pub l: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorGrading {
    pub shadows: HslValues,
    pub mids: HslValues,
    pub highlights: HslValues,
    pub balance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sharpening {
    pub amount: f64,
    pub radius: f64,
    pub detail: f64,
    pub masking: f64,
}

impl Default for Sharpening {
    fn default() -> Self {
        Self {
            amount: 40.0,
            radius: 1.0,
            detail: 25.0,
            masking: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Denoise {
    pub luma: f64,
    pub chroma: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enhanced: Option<bool>,
}

impl Default for Denoise {
    fn default() -> Self {
        Self {
            luma: 0.0,
            chroma: 0.0,
            enhanced: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Optics {
    pub enable: bool,
    pub distortion: f64,
    pub vignette: f64,
    pub ca: f64,
}

impl Default for Optics {
    fn default() -> Self {
        Self {
            enable: false,
            distortion: 0.0,
            vignette: 0.0,
            ca: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crop {
    pub angle: f64,
    pub rect: CropRect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CropRect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub vertical: f64,
    pub horizontal: f64,
    pub rotate: f64,
    pub aspect: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Effects {
    pub grain_amount: f64,
    pub grain_size: f64,
    pub vignette_amount: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FilmSimState {
    pub id: String,
    pub strength: f64,
    pub grain_amount: f64,
    pub grain_size: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bw_filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalAdjustment {
    pub id: String,
    #[serde(rename = "type")]
    pub adjustment_type: String,
    pub mask: serde_json::Value,
    pub params: LocalAdjustmentParams,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct LocalAdjustmentParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exposure: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contrast: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub highlights: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadows: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub whites: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blacks: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tint: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vibrance: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub saturation: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub texture: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clarity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dehaze: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sharpening_amount: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denoise_luma: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiLayer {
    pub id: String,
    pub asset_id: String,
    pub opacity: f64,
    pub blend_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_id: Option<String>,
    pub meta: AiLayerMeta,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiLayerMeta {
    pub provider: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt_hash: Option<String>,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct History {
    pub head_id: String,
    pub nodes: HashMap<String, HistoryNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryNode {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub label: String,
    pub snapshot: serde_json::Value,
    pub ts: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    pub id: String,
    pub name: String,
    pub state_hash: String,
    pub created_at: String,
}

impl EditState {
    pub fn default_for_asset(asset_id: &str) -> Self {
        Self {
            schema_version: 1,
            asset_id: asset_id.to_string(),
            global: GlobalAdjustments::default(),
            film_sim: None,
            local_adjustments: vec![],
            ai_layers: vec![],
            history: None,
            snapshots: None,
        }
    }
}

impl Default for GlobalAdjustments {
    fn default() -> Self {
        Self {
            exposure: 0.0,
            contrast: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            temperature: 5500.0,
            tint: 0.0,
            vibrance: 0.0,
            saturation: 0.0,
            texture: 0.0,
            clarity: 0.0,
            dehaze: 0.0,
            tone_curve: ToneCurve::default(),
            hsl: HashMap::new(),
            grading: None,
            sharpening: Sharpening::default(),
            denoise: Denoise::default(),
            optics: Optics::default(),
            crop: None,
            transform: None,
            effects: None,
        }
    }
}
