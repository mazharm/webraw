use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use image::DynamicImage;
use image::GenericImageView;

use crate::models::auto_enhance::{
    EnhanceModelKind, EnhanceResult, ImageStats, ModelDescriptor,
};
use crate::models::config::AppConfig;
use crate::models::error::AppError;

// ---------------------------------------------------------------------------
// RunContext — passed to every model run
// ---------------------------------------------------------------------------

pub struct RunContext {
    pub api_key: Option<String>,
    pub config: Arc<AppConfig>,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait EnhanceModel: Send + Sync {
    fn descriptor(&self) -> ModelDescriptor;
    fn requires_api_key(&self) -> bool;
    async fn run(
        &self,
        img: &DynamicImage,
        stats: &ImageStats,
        ctx: &RunContext,
    ) -> Result<EnhanceResult, AppError>;
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

pub struct AutoEnhanceService {
    models: Vec<Arc<dyn EnhanceModel>>,
}

impl AutoEnhanceService {
    pub fn new(models_dir: &str) -> Self {
        let mut models: Vec<Arc<dyn EnhanceModel>> = vec![
            Arc::new(BuiltinAutoFix),
            Arc::new(GeminiAutoFix),
        ];

        // Scan for ONNX models (future expansion)
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "onnx") {
                    tracing::info!(
                        "Found ONNX model {:?} but ONNX runtime is not compiled in; skipping",
                        path
                    );
                }
            }
        } else {
            tracing::debug!("Models directory '{}' not found; using built-in models only", models_dir);
        }

        tracing::info!("AutoEnhanceService loaded {} models", models.len());
        let _ = &mut models; // suppress unused mut if no ONNX
        Self { models }
    }

    pub fn list_models(&self) -> Vec<ModelDescriptor> {
        self.models.iter().map(|m| m.descriptor()).collect()
    }

    pub fn get_model(&self, id: &str) -> Option<Arc<dyn EnhanceModel>> {
        self.models.iter().find(|m| m.descriptor().id == id).cloned()
    }
}

// ---------------------------------------------------------------------------
// Stats computation
// ---------------------------------------------------------------------------

pub fn compute_image_stats(img: &DynamicImage) -> ImageStats {
    let (w, h) = img.dimensions();
    let total_pixels = (w as u64) * (h as u64);
    if total_pixels == 0 {
        return zero_stats();
    }

    let mut r_hist = [0u32; 256];
    let mut g_hist = [0u32; 256];
    let mut b_hist = [0u32; 256];
    let mut lum_hist = [0u32; 256];

    let mut r_sum: f64 = 0.0;
    let mut g_sum: f64 = 0.0;
    let mut b_sum: f64 = 0.0;
    let mut lum_sum: f64 = 0.0;
    let mut lum_sq_sum: f64 = 0.0;
    let mut sat_sum: f64 = 0.0;

    let rgb8 = img.to_rgb8();
    for pixel in rgb8.pixels() {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        r_hist[r as usize] += 1;
        g_hist[g as usize] += 1;
        b_hist[b as usize] += 1;

        let rf = r as f64 / 255.0;
        let gf = g as f64 / 255.0;
        let bf = b as f64 / 255.0;

        let lum = 0.2126 * rf + 0.7152 * gf + 0.0722 * bf;
        let lum_byte = (lum * 255.0).clamp(0.0, 255.0) as u8;
        lum_hist[lum_byte as usize] += 1;

        r_sum += rf;
        g_sum += gf;
        b_sum += bf;
        lum_sum += lum;
        lum_sq_sum += lum * lum;

        let max_c = rf.max(gf).max(bf);
        let min_c = rf.min(gf).min(bf);
        let sat = if max_c > 0.0 {
            (max_c - min_c) / max_c
        } else {
            0.0
        };
        sat_sum += sat;
    }

    let n = total_pixels as f64;
    let mean_lum = lum_sum / n;
    let variance = (lum_sq_sum / n) - (mean_lum * mean_lum);
    let stddev_lum = variance.max(0.0).sqrt();

    let r_mean = r_sum / n;
    let g_mean = g_sum / n;
    let b_mean = b_sum / n;
    let sat_mean = sat_sum / n;

    // Percentiles from luminance histogram
    let p1 = percentile_from_hist(&lum_hist, total_pixels, 0.01);
    let p5 = percentile_from_hist(&lum_hist, total_pixels, 0.05);
    let p95 = percentile_from_hist(&lum_hist, total_pixels, 0.95);
    let p99 = percentile_from_hist(&lum_hist, total_pixels, 0.99);

    // Median luminance
    let median_lum = percentile_from_hist(&lum_hist, total_pixels, 0.50);

    // Clipping
    let shadow_clip = lum_hist[0] as f64 / n * 100.0;
    let highlight_clip = lum_hist[255] as f64 / n * 100.0;

    // Color cast: how much R/B deviate from G (warm/cool), and G from average of R+B (tint)
    let avg_rgb = (r_mean + g_mean + b_mean) / 3.0;
    let warm_cool = if avg_rgb > 0.0 {
        (r_mean - b_mean) / avg_rgb
    } else {
        0.0
    };
    let green_magenta = if avg_rgb > 0.0 {
        (g_mean - (r_mean + b_mean) / 2.0) / avg_rgb
    } else {
        0.0
    };

    ImageStats {
        mean_luminance: mean_lum,
        median_luminance: median_lum,
        stddev_luminance: stddev_lum,
        r_mean,
        g_mean,
        b_mean,
        r_histogram: r_hist,
        g_histogram: g_hist,
        b_histogram: b_hist,
        lum_histogram: lum_hist,
        percentile_1: p1,
        percentile_5: p5,
        percentile_95: p95,
        percentile_99: p99,
        shadow_clip_pct: shadow_clip,
        highlight_clip_pct: highlight_clip,
        saturation_mean: sat_mean,
        color_cast: (warm_cool, green_magenta),
    }
}

fn percentile_from_hist(hist: &[u32; 256], total: u64, percentile: f64) -> f64 {
    let target = (total as f64 * percentile) as u64;
    let mut cumulative: u64 = 0;
    for (i, &count) in hist.iter().enumerate() {
        cumulative += count as u64;
        if cumulative >= target {
            return i as f64 / 255.0;
        }
    }
    1.0
}

fn zero_stats() -> ImageStats {
    ImageStats {
        mean_luminance: 0.0,
        median_luminance: 0.0,
        stddev_luminance: 0.0,
        r_mean: 0.0,
        g_mean: 0.0,
        b_mean: 0.0,
        r_histogram: [0; 256],
        g_histogram: [0; 256],
        b_histogram: [0; 256],
        lum_histogram: [0; 256],
        percentile_1: 0.0,
        percentile_5: 0.0,
        percentile_95: 0.0,
        percentile_99: 0.0,
        shadow_clip_pct: 0.0,
        highlight_clip_pct: 0.0,
        saturation_mean: 0.0,
        color_cast: (0.0, 0.0),
    }
}

// ---------------------------------------------------------------------------
// Built-in model: Auto Fix (exposure + color + detail in one)
// ---------------------------------------------------------------------------

struct BuiltinAutoFix;

#[async_trait]
impl EnhanceModel for BuiltinAutoFix {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            id: "builtin".to_string(),
            name: "Built-in (Algorithmic)".to_string(),
            description: "One-click fix: exposure, contrast, white balance, vibrance, clarity, and dehaze — no API key needed".to_string(),
            kind: EnhanceModelKind::ParameterPredictor,
            version: "1.0.0".to_string(),
            builtin: true,
            requires_api_key: false,
        }
    }

    fn requires_api_key(&self) -> bool {
        false
    }

    async fn run(
        &self,
        _img: &DynamicImage,
        stats: &ImageStats,
        _ctx: &RunContext,
    ) -> Result<EnhanceResult, AppError> {
        let mut params = HashMap::new();

        // --- Exposure ---
        let target_median = 0.45;
        let exposure_ev =
            ((target_median / stats.median_luminance.max(0.01)).ln()) / (2.0_f64.ln());
        let exposure = exposure_ev.clamp(-3.0, 3.0);
        params.insert("exposure".to_string(), round2(exposure));

        // --- Contrast ---
        let target_stddev = 0.18;
        let contrast =
            ((target_stddev - stats.stddev_luminance) / target_stddev * 50.0).clamp(-30.0, 40.0);
        params.insert("contrast".to_string(), round2(contrast));

        // --- Highlights ---
        if stats.highlight_clip_pct > 0.5 {
            let highlights = (-stats.highlight_clip_pct * 10.0).clamp(-100.0, 0.0);
            params.insert("highlights".to_string(), round2(highlights));
        }

        // --- Shadows ---
        if stats.shadow_clip_pct > 0.5 {
            let shadows = (stats.shadow_clip_pct * 10.0).clamp(0.0, 100.0);
            params.insert("shadows".to_string(), round2(shadows));
        }

        // --- Whites ---
        let whites = ((0.90 - stats.percentile_95) * 200.0).clamp(-60.0, 60.0);
        if whites.abs() > 5.0 {
            params.insert("whites".to_string(), round2(whites));
        }

        // --- Blacks ---
        let blacks = ((stats.percentile_5 - 0.10) * 200.0).clamp(-60.0, 60.0);
        if blacks.abs() > 5.0 {
            params.insert("blacks".to_string(), round2(blacks));
        }

        // --- Temperature (white balance) ---
        let avg = (stats.r_mean + stats.g_mean + stats.b_mean) / 3.0;
        if avg > 0.01 {
            let rb_ratio = stats.r_mean / stats.b_mean.max(0.001);
            let temp = (5500.0 / rb_ratio).clamp(3000.0, 9000.0);
            if (temp - 5500.0).abs() > 200.0 {
                params.insert("temperature".to_string(), round2(temp));
            }

            let rb_avg = (stats.r_mean + stats.b_mean) / 2.0;
            let g_ratio = stats.g_mean / rb_avg.max(0.001);
            let tint = ((g_ratio - 1.0) * 80.0).clamp(-50.0, 50.0);
            if tint.abs() > 3.0 {
                params.insert("tint".to_string(), round2(tint));
            }
        }

        // --- Vibrance ---
        let target_sat = 0.30;
        let vibrance =
            ((target_sat - stats.saturation_mean) / target_sat * 40.0).clamp(-20.0, 40.0);
        if vibrance.abs() > 3.0 {
            params.insert("vibrance".to_string(), round2(vibrance));
        }

        // --- Clarity (boost mid-tone contrast when image is flat) ---
        let clarity = ((0.20 - stats.stddev_luminance) / 0.20 * 30.0).clamp(0.0, 35.0);
        if clarity > 5.0 {
            params.insert("clarity".to_string(), round2(clarity));
        }

        // --- Dehaze (simple heuristic: low-contrast + elevated blacks → hazy) ---
        let haze_indicator = (stats.percentile_5 - 0.05).max(0.0) * 5.0
            + (0.15 - stats.stddev_luminance).max(0.0) * 3.0;
        let dehaze = (haze_indicator * 40.0).clamp(0.0, 40.0);
        if dehaze > 5.0 {
            params.insert("dehaze".to_string(), round2(dehaze));
        }

        Ok(EnhanceResult::Parameters { values: params })
    }
}

// ---------------------------------------------------------------------------
// Gemini AI model: sends image to Gemini vision API
// ---------------------------------------------------------------------------

struct GeminiAutoFix;

#[async_trait]
impl EnhanceModel for GeminiAutoFix {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            id: "gemini".to_string(),
            name: "Gemini AI".to_string(),
            description: "AI-powered analysis using Google Gemini vision — requires API key"
                .to_string(),
            kind: EnhanceModelKind::ParameterPredictor,
            version: "1.0.0".to_string(),
            builtin: false,
            requires_api_key: true,
        }
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    async fn run(
        &self,
        img: &DynamicImage,
        _stats: &ImageStats,
        ctx: &RunContext,
    ) -> Result<EnhanceResult, AppError> {
        let api_key = ctx
            .api_key
            .as_deref()
            .ok_or(AppError::AiInvalidKey)?;

        // Encode image as JPEG base64
        let mut jpeg_buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut jpeg_buf, image::ImageFormat::Jpeg)
            .map_err(|e| AppError::AutoEnhanceError(format!("Failed to encode proxy JPEG: {}", e)))?;
        let b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            jpeg_buf.into_inner(),
        );

        let gemini_model = &ctx.config.gemini_model;
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            gemini_model, api_key
        );

        let prompt = r#"Analyze this photograph and recommend optimal development parameters.
Return ONLY a JSON object with numeric values for these keys:
- exposure (-5 to 5, EV stops)
- contrast (-100 to 100)
- highlights (-100 to 100)
- shadows (-100 to 100)
- whites (-100 to 100)
- blacks (-100 to 100)
- temperature (2000 to 50000, Kelvin)
- tint (-150 to 150)
- vibrance (-100 to 100)
- saturation (-100 to 100)
- clarity (-100 to 100)
- dehaze (-100 to 100)
- texture (-100 to 100)"#;

        let body = serde_json::json!({
            "contents": [{
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": b64
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseModalities": ["TEXT"]
            }
        });

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AppError::AutoEnhanceError(format!("Gemini request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(AppError::AiInvalidKey);
            }
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                return Err(AppError::AiQuotaExceeded);
            }
            return Err(AppError::AutoEnhanceError(format!(
                "Gemini API error {}: {}",
                status, text
            )));
        }

        let resp_json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AppError::AutoEnhanceError(format!("Failed to parse Gemini response: {}", e)))?;

        // Extract text from candidates[0].content.parts[0].text
        let text = resp_json
            .pointer("/candidates/0/content/parts/0/text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                AppError::AutoEnhanceError("No text in Gemini response".to_string())
            })?;

        let parsed: HashMap<String, f64> = serde_json::from_str(text).map_err(|e| {
            AppError::AutoEnhanceError(format!(
                "Failed to parse Gemini JSON: {} — raw: {}",
                e, text
            ))
        })?;

        Ok(EnhanceResult::Parameters { values: parsed })
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
