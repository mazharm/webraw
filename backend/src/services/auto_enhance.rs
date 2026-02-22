use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use image::DynamicImage;
use image::GenericImageView;
use ndarray::Array4;
use serde::Deserialize;

use crate::models::auto_enhance::{
    EnhanceModelKind, EnhanceResult, ImageStats, ModelDescriptor,
};
use crate::models::config::AppConfig;
use crate::models::error::AppError;

// ---------------------------------------------------------------------------
// RunContext — passed to every model run
// ---------------------------------------------------------------------------

pub struct RunContext {
    pub api_key: Option<String>,        // Gemini
    pub anthropic_key: Option<String>,  // Claude
    pub openai_key: Option<String>,     // OpenAI
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
            Arc::new(ClaudeAutoFix),
            Arc::new(GeminiAutoFix),
            Arc::new(OpenAiAutoFix),
        ];

        // Scan for model manifest JSON files and load ONNX models
        let models_path = Path::new(models_dir);
        if let Ok(entries) = std::fs::read_dir(models_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "json") {
                    match Self::load_manifest_model(&path, models_path) {
                        Ok(model) => {
                            tracing::info!("Loaded ONNX model '{}' from {:?}", model.descriptor().name, path);
                            models.push(model);
                        }
                        Err(e) => {
                            tracing::warn!("Skipping manifest {:?}: {}", path, e);
                        }
                    }
                }
            }
        } else {
            tracing::debug!("Models directory '{}' not found; using built-in models only", models_dir);
        }

        tracing::info!("AutoEnhanceService loaded {} models", models.len());
        Self { models }
    }

    fn load_manifest_model(
        manifest_path: &Path,
        models_dir: &Path,
    ) -> Result<Arc<dyn EnhanceModel>, String> {
        let json_str = std::fs::read_to_string(manifest_path)
            .map_err(|e| format!("Failed to read manifest: {}", e))?;
        let manifest: ModelManifest = serde_json::from_str(&json_str)
            .map_err(|e| format!("Failed to parse manifest: {}", e))?;

        let onnx_path = models_dir.join(&manifest.file);
        if !onnx_path.exists() {
            return Err(format!("ONNX file '{}' not found", manifest.file));
        }

        let variant = match manifest.adapter.as_str() {
            "nima_aesthetic" => NimaVariant::Aesthetic,
            "nima_technical" => NimaVariant::Technical,
            other => return Err(format!("Unknown adapter type: {}", other)),
        };

        let session = ort::session::Session::builder()
            .and_then(|b| b.with_intra_threads(2))
            .and_then(|b| b.commit_from_file(&onnx_path))
            .map_err(|e| format!("Failed to load ONNX session: {}", e))?;

        Ok(Arc::new(NimaAutoFix {
            session: parking_lot::Mutex::new(session),
            variant,
            manifest,
        }))
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
            publisher: None,
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
// Claude AI model: sends image to Anthropic Messages API with vision
// ---------------------------------------------------------------------------

struct ClaudeAutoFix;

#[async_trait]
impl EnhanceModel for ClaudeAutoFix {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            id: "claude".to_string(),
            name: "Claude AI".to_string(),
            description: "AI-powered analysis using Claude vision — requires API key"
                .to_string(),
            kind: EnhanceModelKind::ParameterPredictor,
            version: "1.0.0".to_string(),
            builtin: false,
            requires_api_key: true,
            publisher: Some("Anthropic".to_string()),
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
            .anthropic_key
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

        let claude_model = &ctx.config.claude_model;
        let url = "https://api.anthropic.com/v1/messages";

        let prompt = ANALYSIS_PROMPT;

        let body = serde_json::json!({
            "model": claude_model,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        });

        let client = reqwest::Client::new();
        let resp = client
            .post(url)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AppError::AutoEnhanceError(format!("Claude request failed: {}", e)))?;

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
                "Claude API error {}: {}",
                status, text
            )));
        }

        let resp_json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AppError::AutoEnhanceError(format!("Failed to parse Claude response: {}", e)))?;

        // Extract text from content[0].text
        let text = resp_json
            .pointer("/content/0/text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                AppError::AutoEnhanceError("No text in Claude response".to_string())
            })?;

        let json_str = strip_markdown_fences(text);

        let parsed: HashMap<String, f64> = serde_json::from_str(json_str).map_err(|e| {
            AppError::AutoEnhanceError(format!(
                "Failed to parse Claude JSON: {} — raw: {}",
                e, text
            ))
        })?;

        Ok(EnhanceResult::Parameters { values: parsed })
    }
}

// ---------------------------------------------------------------------------
// Gemini AI model: sends image to Google Generative Language API with vision
// ---------------------------------------------------------------------------

struct GeminiAutoFix;

#[async_trait]
impl EnhanceModel for GeminiAutoFix {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            id: "gemini".to_string(),
            name: "Gemini AI".to_string(),
            description: "AI-powered analysis using Gemini vision — requires API key"
                .to_string(),
            kind: EnhanceModelKind::ParameterPredictor,
            version: "1.0.0".to_string(),
            builtin: false,
            requires_api_key: true,
            publisher: Some("Google".to_string()),
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

        let prompt = ANALYSIS_PROMPT;

        let body = serde_json::json!({
            "contents": [{
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }]
        });

        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .header("content-type", "application/json")
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

        let json_str = strip_markdown_fences(text);

        let parsed: HashMap<String, f64> = serde_json::from_str(json_str).map_err(|e| {
            AppError::AutoEnhanceError(format!(
                "Failed to parse Gemini JSON: {} — raw: {}",
                e, text
            ))
        })?;

        Ok(EnhanceResult::Parameters { values: parsed })
    }
}

// ---------------------------------------------------------------------------
// OpenAI GPT-4o model: sends image to OpenAI Chat Completions API with vision
// ---------------------------------------------------------------------------

struct OpenAiAutoFix;

#[async_trait]
impl EnhanceModel for OpenAiAutoFix {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            id: "openai".to_string(),
            name: "OpenAI GPT-4o".to_string(),
            description: "AI-powered analysis using GPT-4o vision — requires API key"
                .to_string(),
            kind: EnhanceModelKind::ParameterPredictor,
            version: "1.0.0".to_string(),
            builtin: false,
            requires_api_key: true,
            publisher: Some("OpenAI".to_string()),
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
            .openai_key
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

        let openai_model = &ctx.config.openai_model;
        let url = "https://api.openai.com/v1/chat/completions";

        let prompt = ANALYSIS_PROMPT;

        let body = serde_json::json!({
            "model": openai_model,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:image/jpeg;base64,{}", b64)
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        });

        let client = reqwest::Client::new();
        let resp = client
            .post(url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| AppError::AutoEnhanceError(format!("OpenAI request failed: {}", e)))?;

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
                "OpenAI API error {}: {}",
                status, text
            )));
        }

        let resp_json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| AppError::AutoEnhanceError(format!("Failed to parse OpenAI response: {}", e)))?;

        // Extract text from choices[0].message.content
        let text = resp_json
            .pointer("/choices/0/message/content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                AppError::AutoEnhanceError("No text in OpenAI response".to_string())
            })?;

        let json_str = strip_markdown_fences(text);

        let parsed: HashMap<String, f64> = serde_json::from_str(json_str).map_err(|e| {
            AppError::AutoEnhanceError(format!(
                "Failed to parse OpenAI JSON: {} — raw: {}",
                e, text
            ))
        })?;

        Ok(EnhanceResult::Parameters { values: parsed })
    }
}

// ---------------------------------------------------------------------------
// Shared prompt and helpers for AI providers
// ---------------------------------------------------------------------------

const ANALYSIS_PROMPT: &str = r#"Analyze this photograph and recommend optimal development parameters.
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

fn strip_markdown_fences(text: &str) -> &str {
    let s = text.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    let s = s.trim();
    let s = s.strip_suffix("```").unwrap_or(s);
    s.trim()
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

// ---------------------------------------------------------------------------
// Model manifest (deserialized from JSON sidecar)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelManifest {
    id: String,
    name: String,
    publisher: String,
    description: String,
    adapter: String,
    file: String,
    input_size: [u32; 2],
    normalize: NormalizeConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct NormalizeConfig {
    mean: [f32; 3],
    std: [f32; 3],
}

// ---------------------------------------------------------------------------
// NIMA CNN provider
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum NimaVariant {
    Aesthetic,
    Technical,
}

struct NimaAutoFix {
    session: parking_lot::Mutex<ort::session::Session>,
    variant: NimaVariant,
    manifest: ModelManifest,
}

#[async_trait]
impl EnhanceModel for NimaAutoFix {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            id: self.manifest.id.clone(),
            name: self.manifest.name.clone(),
            description: self.manifest.description.clone(),
            kind: EnhanceModelKind::ParameterPredictor,
            version: "1.0.0".to_string(),
            builtin: false,
            requires_api_key: false,
            publisher: Some(self.manifest.publisher.clone()),
        }
    }

    fn requires_api_key(&self) -> bool {
        false
    }

    async fn run(
        &self,
        img: &DynamicImage,
        stats: &ImageStats,
        _ctx: &RunContext,
    ) -> Result<EnhanceResult, AppError> {
        // Preprocess image for ONNX model
        let input_tensor =
            preprocess_for_onnx(img, self.manifest.input_size, &self.manifest.normalize);

        // Run inference
        let input_value = ort::value::Tensor::from_array(input_tensor).map_err(|e| {
            AppError::AutoEnhanceError(format!("Failed to create ONNX input tensor: {}", e))
        })?;
        let probs: Vec<f32> = {
            let mut session = self.session.lock();
            let outputs = session
                .run(ort::inputs![input_value])
                .map_err(|e| {
                    AppError::AutoEnhanceError(format!("ONNX inference failed: {}", e))
                })?;
            let (_shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| {
                    AppError::AutoEnhanceError(format!("Failed to extract ONNX output: {}", e))
                })?;
            data.to_vec()
        };

        // Compute mean score (weighted sum: scores 1–10)
        let mean_score: f32 = probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i as f32 + 1.0) * p)
            .sum();

        tracing::info!(
            "NIMA {:?} score: {:.2} (from {} classes)",
            self.variant,
            mean_score,
            probs.len()
        );

        // Compute correction strength: score >= 6 → ~0, score = 2 → 1.0
        let correction_strength = ((6.0_f32 - mean_score) / 4.0_f32).clamp(0.0_f32, 1.0_f32) as f64;

        // Apply heuristic mapping scaled by correction_strength
        let params = match self.variant {
            NimaVariant::Aesthetic => nima_aesthetic_params(stats, correction_strength),
            NimaVariant::Technical => nima_technical_params(stats, correction_strength),
        };

        Ok(EnhanceResult::Parameters { values: params })
    }
}

/// Preprocess an image for ONNX inference: resize, normalize, return NCHW tensor.
fn preprocess_for_onnx(
    img: &DynamicImage,
    size: [u32; 2],
    norm: &NormalizeConfig,
) -> Array4<f32> {
    let resized = img.resize_exact(size[0], size[1], image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    let (w, h) = (size[0] as usize, size[1] as usize);
    let mut tensor = Array4::<f32>::zeros((1, 3, h, w));

    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                tensor[[0, c, y, x]] = (val - norm.mean[c]) / norm.std[c];
            }
        }
    }

    tensor
}

/// Aesthetic NIMA: biases toward vibrance, saturation, contrast, temperature/tint.
fn nima_aesthetic_params(stats: &ImageStats, strength: f64) -> HashMap<String, f64> {
    let mut params = HashMap::new();

    // Vibrance
    let target_sat = 0.30;
    let vibrance = ((target_sat - stats.saturation_mean) / target_sat * 40.0).clamp(-20.0, 40.0);
    if (vibrance * strength).abs() > 1.0 {
        params.insert("vibrance".to_string(), round2(vibrance * strength));
    }

    // Saturation (lighter touch)
    let saturation = ((target_sat - stats.saturation_mean) / target_sat * 20.0).clamp(-15.0, 25.0);
    if (saturation * strength).abs() > 1.0 {
        params.insert("saturation".to_string(), round2(saturation * strength));
    }

    // Contrast
    let target_stddev = 0.18;
    let contrast =
        ((target_stddev - stats.stddev_luminance) / target_stddev * 50.0).clamp(-30.0, 40.0);
    if (contrast * strength).abs() > 2.0 {
        params.insert("contrast".to_string(), round2(contrast * strength));
    }

    // Temperature (white balance)
    let avg = (stats.r_mean + stats.g_mean + stats.b_mean) / 3.0;
    if avg > 0.01 {
        let rb_ratio = stats.r_mean / stats.b_mean.max(0.001);
        let temp = (5500.0 / rb_ratio).clamp(3000.0, 9000.0);
        let temp_delta = temp - 5500.0;
        if (temp_delta * strength).abs() > 100.0 {
            params.insert(
                "temperature".to_string(),
                round2(5500.0 + temp_delta * strength),
            );
        }

        // Tint
        let rb_avg = (stats.r_mean + stats.b_mean) / 2.0;
        let g_ratio = stats.g_mean / rb_avg.max(0.001);
        let tint = ((g_ratio - 1.0) * 80.0).clamp(-50.0, 50.0);
        if (tint * strength).abs() > 2.0 {
            params.insert("tint".to_string(), round2(tint * strength));
        }
    }

    params
}

/// Technical NIMA: biases toward exposure, highlights, shadows, whites, blacks, clarity, dehaze.
fn nima_technical_params(stats: &ImageStats, strength: f64) -> HashMap<String, f64> {
    let mut params = HashMap::new();

    // Exposure
    let target_median = 0.45;
    let exposure_ev =
        ((target_median / stats.median_luminance.max(0.01)).ln()) / (2.0_f64.ln());
    let exposure = exposure_ev.clamp(-3.0, 3.0);
    if (exposure * strength).abs() > 0.05 {
        params.insert("exposure".to_string(), round2(exposure * strength));
    }

    // Highlights
    if stats.highlight_clip_pct > 0.5 {
        let highlights = (-stats.highlight_clip_pct * 10.0).clamp(-100.0, 0.0);
        params.insert("highlights".to_string(), round2(highlights * strength));
    }

    // Shadows
    if stats.shadow_clip_pct > 0.5 {
        let shadows = (stats.shadow_clip_pct * 10.0).clamp(0.0, 100.0);
        params.insert("shadows".to_string(), round2(shadows * strength));
    }

    // Whites
    let whites = ((0.90 - stats.percentile_95) * 200.0).clamp(-60.0, 60.0);
    if (whites * strength).abs() > 3.0 {
        params.insert("whites".to_string(), round2(whites * strength));
    }

    // Blacks
    let blacks = ((stats.percentile_5 - 0.10) * 200.0).clamp(-60.0, 60.0);
    if (blacks * strength).abs() > 3.0 {
        params.insert("blacks".to_string(), round2(blacks * strength));
    }

    // Clarity
    let clarity = ((0.20 - stats.stddev_luminance) / 0.20 * 30.0).clamp(0.0, 35.0);
    if clarity * strength > 3.0 {
        params.insert("clarity".to_string(), round2(clarity * strength));
    }

    // Dehaze
    let haze_indicator = (stats.percentile_5 - 0.05).max(0.0) * 5.0
        + (0.15 - stats.stddev_luminance).max(0.0) * 3.0;
    let dehaze = (haze_indicator * 40.0).clamp(0.0, 40.0);
    if dehaze * strength > 3.0 {
        params.insert("dehaze".to_string(), round2(dehaze * strength));
    }

    params
}
