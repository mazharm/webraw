use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use base64::Engine as _;
use image::{DynamicImage, ImageFormat, GenericImageView, imageops::FilterType};
use parking_lot::Mutex;
use tokio::sync::Semaphore;

use crate::models::config::AppConfig;
use crate::models::edit_state::EditState;
use crate::models::error::AppError;
use crate::services::cache::FileCache;

const MAX_CACHED_IMAGES: usize = 8;

pub struct RenderService {
    config: Arc<AppConfig>,
    render_semaphore: Semaphore,
    export_semaphore: Semaphore,
    base_image_cache: Mutex<HashMap<(String, u32), (Arc<DynamicImage>, std::time::Instant)>>,
}

impl RenderService {
    pub fn new(config: Arc<AppConfig>) -> Self {
        Self {
            render_semaphore: Semaphore::new(config.max_parallel_renders),
            export_semaphore: Semaphore::new(config.max_parallel_exports),
            config,
            base_image_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Check if base image is already cached (no file I/O needed)
    fn get_cached_base_image(&self, file_id: &str, max_edge: u32) -> Option<Arc<DynamicImage>> {
        let key = (file_id.to_string(), max_edge);
        let cache = self.base_image_cache.lock();
        cache.get(&key).map(|(img, _)| Arc::clone(img))
    }

    /// Get a decoded+resized base image, using cache when available.
    fn get_base_image(&self, file_id: &str, max_edge: u32, data: &[u8]) -> Result<Arc<DynamicImage>, AppError> {
        let key = (file_id.to_string(), max_edge);

        // Check cache
        {
            let cache = self.base_image_cache.lock();
            if let Some((img, _)) = cache.get(&key) {
                return Ok(Arc::clone(img));
            }
        }

        // Cache miss — decode and resize
        let img = load_image(data)?;
        let resized = resize_image(&img, max_edge);
        let arc_img = Arc::new(resized);

        // Insert into cache, evict oldest if needed
        {
            let mut cache = self.base_image_cache.lock();
            if cache.len() >= MAX_CACHED_IMAGES {
                if let Some(oldest_key) = cache
                    .iter()
                    .min_by_key(|(_, (_, ts))| *ts)
                    .map(|(k, _)| k.clone())
                {
                    cache.remove(&oldest_key);
                }
            }
            cache.insert(key, (Arc::clone(&arc_img), std::time::Instant::now()));
        }

        Ok(arc_img)
    }

    /// Resolve base image — try cache first, then fall back to file read + decode.
    async fn resolve_base_image(&self, file_cache: &FileCache, file_id: &str, max_edge: u32) -> Result<Arc<DynamicImage>, AppError> {
        // Fast path: return cached image without any file I/O
        if let Some(img) = self.get_cached_base_image(file_id, max_edge) {
            return Ok(img);
        }
        // Slow path: read file, decode + resize on blocking thread to avoid
        // holding a tokio worker hostage during CPU-heavy RAW decode
        let data = file_cache.read_file_bytes(file_id).await?;
        let arc_img = tokio::task::spawn_blocking(move || -> Result<Arc<DynamicImage>, AppError> {
            let img = load_image(&data)?;
            Ok(Arc::new(resize_image(&img, max_edge)))
        })
        .await
        .map_err(|e| AppError::Internal(format!("Decode task panicked: {}", e)))??;

        // Insert into cache
        let key = (file_id.to_string(), max_edge);
        {
            let mut cache = self.base_image_cache.lock();
            if cache.len() >= MAX_CACHED_IMAGES {
                if let Some(oldest_key) = cache
                    .iter()
                    .min_by_key(|(_, (_, ts))| *ts)
                    .map(|(k, _)| k.clone())
                {
                    cache.remove(&oldest_key);
                }
            }
            cache.insert(key, (Arc::clone(&arc_img), std::time::Instant::now()));
        }

        Ok(arc_img)
    }

    /// Public accessor for the proxy/base image. Used by auto-enhance to
    /// analyze pixels without rendering a full edit pipeline.
    pub async fn get_proxy_image(
        &self,
        cache: &FileCache,
        file_id: &str,
        max_edge: u32,
    ) -> Result<Arc<DynamicImage>, AppError> {
        self.resolve_base_image(cache, file_id, max_edge).await
    }

    pub async fn generate_thumbnail(
        &self,
        cache: &FileCache,
        file_id: &str,
        max_edge: u32,
    ) -> Result<Vec<u8>, AppError> {
        let _permit = self.render_semaphore.acquire().await.map_err(|_| {
            AppError::Internal("Render semaphore closed".to_string())
        })?;

        let base = self.resolve_base_image(cache, file_id, max_edge).await?;
        encode_jpeg(&base, 85)
    }

    pub async fn render_preview(
        &self,
        cache: &FileCache,
        file_id: &str,
        edit_state: &EditState,
        max_edge: u32,
        color_space: &str,
        _quality_hint: &str,
    ) -> Result<PreviewResult, AppError> {
        let _permit = self.render_semaphore.acquire().await.map_err(|_| {
            AppError::Internal("Render semaphore closed".to_string())
        })?;

        let base = self.resolve_base_image(cache, file_id, max_edge).await?;

        // apply_edit_state works on a clone — the cached base is untouched
        let rendered = apply_edit_state(&base, edit_state);
        let (width, height) = rendered.dimensions();

        let png_data = encode_png(&rendered)?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png_data);

        let histogram = compute_histogram(&rendered);

        Ok(PreviewResult {
            image_base64: b64,
            mime_type: "image/png".to_string(),
            bit_depth: 8,
            color_space: color_space.to_string(),
            width,
            height,
            histogram: Some(histogram),
        })
    }

    pub async fn render_base(
        &self,
        cache: &FileCache,
        file_id: &str,
        max_edge: u32,
    ) -> Result<(Vec<u8>, u32, u32), AppError> {
        let _permit = self.render_semaphore.acquire().await.map_err(|_| {
            AppError::Internal("Render semaphore closed".to_string())
        })?;

        let base = self.resolve_base_image(cache, file_id, max_edge).await?;
        let (w, h) = base.dimensions();
        let png_data = encode_png(&base)?;

        Ok((png_data, w, h))
    }

    pub async fn render_export(
        &self,
        cache: &FileCache,
        file_id: &str,
        edit_state: &EditState,
        format: &str,
        _bit_depth: u8,
        quality: u8,
        _color_space: &str,
    ) -> Result<ExportResult, AppError> {
        let _permit = self.export_semaphore.acquire().await.map_err(|_| {
            AppError::Internal("Export semaphore closed".to_string())
        })?;

        let data = cache.read_file_bytes(file_id).await?;
        // For export, use full-resolution (no max_edge limit — use original size)
        let img = load_image(&data)?;
        let rendered = apply_edit_state(&img, edit_state);

        let (encoded, mime) = match format {
            "JPG" => (encode_jpeg(&rendered, quality)?, "image/jpeg"),
            "PNG" => (encode_png(&rendered)?, "image/png"),
            "TIFF" => (encode_tiff(&rendered)?, "image/tiff"),
            _ => return Err(AppError::ValidationError(format!("Unsupported format: {}", format))),
        };

        let mut hasher = sha2::Sha256::new();
        use sha2::Digest;
        hasher.update(&encoded);
        let checksum = hex::encode(hasher.finalize());

        let artifact = cache.store_artifact(&encoded, mime, file_id).await?;

        Ok(ExportResult {
            download_url: format!("/api/v1/files/{}", artifact.file_id),
            size_bytes: encoded.len() as u64,
            checksum,
        })
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewResult {
    pub image_base64: String,
    pub mime_type: String,
    pub bit_depth: u8,
    pub color_space: String,
    pub width: u32,
    pub height: u32,
    pub histogram: Option<HistogramData>,
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub download_url: String,
    pub size_bytes: u64,
    pub checksum: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HistogramData {
    pub r: Vec<u32>,
    pub g: Vec<u32>,
    pub b: Vec<u32>,
    pub lum: Vec<u32>,
}

fn load_image(data: &[u8]) -> Result<DynamicImage, AppError> {
    // Try standard formats first (JPEG, PNG, TIFF)
    if let Ok(img) = image::load_from_memory(data) {
        return Ok(img);
    }
    // Fall back to RAW decode
    load_raw_image(data)
}

fn load_raw_image(data: &[u8]) -> Result<DynamicImage, AppError> {
    let raw = rawloader::decode(&mut Cursor::new(data)).map_err(|e| {
        let msg = e.to_string();
        if msg.contains("unknown") || msg.contains("Unsupported") {
            AppError::RawUnsupportedCamera(msg)
        } else {
            AppError::RawCorruptData(msg)
        }
    })?;

    let width = raw.width;
    let height = raw.height;

    let raw_data = match raw.data {
        rawloader::RawImageData::Integer(ref v) => v.clone(),
        rawloader::RawImageData::Float(ref v) => {
            v.iter().map(|&f| f as u16).collect::<Vec<u16>>()
        }
    };

    if raw_data.len() < width * height {
        return Err(AppError::RawCorruptData(
            "RAW data size mismatch".to_string(),
        ));
    }

    // Extract black/white levels and WB coefficients
    let blacks = &raw.blacklevels;
    let whites = &raw.whitelevels;
    let wb = &raw.wb_coeffs;

    // Normalize WB: scale so green multiplier = 1.0
    let wb_green = if wb[1] > 0.0 { wb[1] } else { 1.0 };
    let wb_norm = [wb[0] / wb_green, 1.0, wb[2] / wb_green];

    let cfa = &raw.cfa;

    // HDR: normalize raw pixels WITHOUT clamping to [0,1] — preserve values > 1.0
    // for highlight recovery. Values are in linear light space.
    let mut normalized = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let cfa_x = x % cfa.width;
            let cfa_y = y % cfa.height;
            let color_idx = cfa.color_at(cfa_x, cfa_y);
            let black = blacks[color_idx] as f32;
            let white = whites[color_idx] as f32;
            let range = (white - black).max(1.0);
            let val = ((raw_data[idx] as f32 - black) / range).max(0.0);
            // Apply white balance — don't clamp to preserve HDR headroom
            let wb_factor = match color_idx {
                0 => wb_norm[0],
                1 | 3 => wb_norm[1],
                2 => wb_norm[2],
                _ => 1.0,
            };
            normalized[idx] = val * wb_factor as f32;
        }
    }

    // Bilinear demosaic → filmic tone mapping → sRGB gamma
    let mut rgb = vec![0u8; width * height * 3];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let cfa_x = x % cfa.width;
            let cfa_y = y % cfa.height;
            let color_idx = cfa.color_at(cfa_x, cfa_y);
            let idx = y * width + x;

            let (r, g, b) = match color_idx {
                // Red pixel
                0 => {
                    let r = normalized[idx];
                    let g = (normalized[idx - 1] + normalized[idx + 1]
                        + normalized[idx - width] + normalized[idx + width]) / 4.0;
                    let b = (normalized[idx - width - 1] + normalized[idx - width + 1]
                        + normalized[idx + width - 1] + normalized[idx + width + 1]) / 4.0;
                    (r, g, b)
                }
                // Green pixel — use row parity to determine neighbor colors
                1 => {
                    let red_row = cfa.color_at(0, cfa_y) == 0;
                    if red_row {
                        let r = (normalized[idx - 1] + normalized[idx + 1]) / 2.0;
                        let g = normalized[idx];
                        let b = (normalized[idx - width] + normalized[idx + width]) / 2.0;
                        (r, g, b)
                    } else {
                        let r = (normalized[idx - width] + normalized[idx + width]) / 2.0;
                        let g = normalized[idx];
                        let b = (normalized[idx - 1] + normalized[idx + 1]) / 2.0;
                        (r, g, b)
                    }
                }
                // Blue pixel
                2 => {
                    let r = (normalized[idx - width - 1] + normalized[idx - width + 1]
                        + normalized[idx + width - 1] + normalized[idx + width + 1]) / 4.0;
                    let g = (normalized[idx - 1] + normalized[idx + 1]
                        + normalized[idx - width] + normalized[idx + width]) / 4.0;
                    let b = normalized[idx];
                    (r, g, b)
                }
                _ => (normalized[idx], normalized[idx], normalized[idx]),
            };

            // HDR filmic tone mapping — preserves highlight detail
            let out_idx = (y * width + x) * 3;
            rgb[out_idx] = srgb_gamma(filmic_tonemap(r));
            rgb[out_idx + 1] = srgb_gamma(filmic_tonemap(g));
            rgb[out_idx + 2] = srgb_gamma(filmic_tonemap(b));
        }
    }

    let img_buf = image::RgbImage::from_raw(width as u32, height as u32, rgb)
        .ok_or_else(|| AppError::RawCorruptData("Failed to create image buffer".to_string()))?;

    Ok(DynamicImage::ImageRgb8(img_buf))
}

/// Filmic tone mapping (Hable/Uncharted 2 style)
/// Maps linear HDR values [0, ∞) to [0, 1) with natural highlight roll-off
fn filmic_tonemap(x: f32) -> f32 {
    // Attempt to handle the typical exposure of camera RAWs where
    // 1.0 = sensor white point. Most well-exposed images sit around 0.1-0.5.
    // We want a tone curve that:
    // - Passes through ~0.18 (middle gray) mostly unchanged
    // - Gently rolls off highlights above 0.7
    // - Gracefully handles values > 1.0 from WB or hot highlights
    const A: f32 = 0.22;  // shoulder strength
    const B: f32 = 0.30;  // linear strength
    const C: f32 = 0.10;  // linear angle
    const D: f32 = 0.20;  // toe strength
    const E: f32 = 0.01;  // toe numerator
    const F: f32 = 0.30;  // toe denominator

    fn curve(x: f32) -> f32 {
        ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    }

    const W: f32 = 2.0; // linear white point
    let numerator = curve(x);
    let denominator = curve(W);
    (numerator / denominator).clamp(0.0, 1.0)
}

/// Apply sRGB gamma transfer function (linear → sRGB)
fn srgb_gamma(v: f32) -> u8 {
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).clamp(0.0, 255.0) as u8
}

fn resize_image(img: &DynamicImage, max_edge: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w <= max_edge && h <= max_edge {
        return img.clone();
    }
    let scale = max_edge as f64 / w.max(h) as f64;
    let nw = (w as f64 * scale) as u32;
    let nh = (h as f64 * scale) as u32;
    // Use fast bilinear for small targets (thumbnails); Lanczos3 for larger previews
    let filter = if max_edge <= 600 { FilterType::Triangle } else { FilterType::Lanczos3 };
    img.resize_exact(nw, nh, filter)
}

/// Buffer-level operations that can be inserted between pipeline phases.
/// When all fields are None, the pipeline runs identically to the original.
#[derive(Default)]
pub struct OptimizeBufferOps {
    /// Denoised image buffer (RGBA u8) to replace after Phase 1 (Exposure/WB)
    pub denoised_buffer: Option<Vec<u8>>,
    /// HDRNet bilateral grid coefficients to apply after Phase 2 (Tonal/Contrast/Curve)
    pub hdrnet_coeffs: Option<Vec<f32>>,
}

fn apply_edit_state(img: &DynamicImage, state: &EditState) -> DynamicImage {
    apply_edit_state_with_ops(img, state, &OptimizeBufferOps::default())
}

fn apply_edit_state_with_ops(img: &DynamicImage, state: &EditState, ops: &OptimizeBufferOps) -> DynamicImage {
    let rgba = img.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());
    let mut output = rgba.clone();
    let g = &state.global;

    // Pre-compute tone curve LUT (256 entries)
    let tone_lut = build_tone_curve_lut(&g.tone_curve);

    // Pre-compute film sim parameters (strength is already 0.0-2.0 from frontend)
    let film = state.film_sim.as_ref().map(|fs| {
        let look = get_film_look(fs);
        (look, fs.strength, fs.grain_amount, fs.grain_size, fs.bw_filter.clone())
    });

    // Pre-compute HSL adjustments
    let has_hsl = !g.hsl.is_empty() && g.hsl.values().any(|v| v.h != 0.0 || v.s != 0.0 || v.l != 0.0);

    // Pre-compute basic adjustments
    let exposure_factor = if g.exposure != 0.0 { 2.0_f64.powf(g.exposure) } else { 1.0 };
    let contrast_factor = if g.contrast != 0.0 { (100.0 + g.contrast) / 100.0 } else { 1.0 };
    let sat_factor = if g.saturation != 0.0 { 1.0 + g.saturation / 100.0 } else { 1.0 };
    let vib_factor = if g.vibrance != 0.0 { g.vibrance / 100.0 } else { 0.0 };
    let temp_shift = if g.temperature != 5500.0 { (g.temperature - 5500.0) / 5500.0 * 0.1 } else { 0.0 };
    let tint_shift = if g.tint != 0.0 { g.tint / 150.0 * 0.05 } else { 0.0 };
    let has_tonal = g.highlights != 0.0 || g.shadows != 0.0 || g.whites != 0.0 || g.blacks != 0.0;
    let hdr_strength = g.hdr / 100.0; // 0..1

    // Effects
    let effects = g.effects.as_ref();
    let vignette_amount = effects.map(|e| e.vignette_amount).unwrap_or(0.0);
    let grain_amount = effects.map(|e| e.grain_amount).unwrap_or(0.0);
    let grain_size = effects.map(|e| e.grain_size).unwrap_or(1.0);

    // Texture/Clarity (mid-frequency contrast)
    let clarity_strength = g.clarity / 100.0;
    let texture_strength = g.texture / 100.0;

    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;

    // ═══ Phase 1: Exposure + White Balance ═══════════════════════════
    for pixel in output.pixels_mut() {
        let mut r = pixel[0] as f64 / 255.0;
        let mut g_ch = pixel[1] as f64 / 255.0;
        let mut b = pixel[2] as f64 / 255.0;

        // --- Exposure ---
        if exposure_factor != 1.0 {
            r *= exposure_factor;
            g_ch *= exposure_factor;
            b *= exposure_factor;
        }

        // --- White balance ---
        if temp_shift != 0.0 {
            r *= 1.0 + temp_shift;
            b *= 1.0 - temp_shift;
        }
        if tint_shift != 0.0 {
            g_ch *= 1.0 - tint_shift;
            r *= 1.0 + tint_shift * 0.5;
            b *= 1.0 + tint_shift * 0.5;
        }

        pixel[0] = (r * 255.0).clamp(0.0, 255.0) as u8;
        pixel[1] = (g_ch * 255.0).clamp(0.0, 255.0) as u8;
        pixel[2] = (b * 255.0).clamp(0.0, 255.0) as u8;
    }

    // ═══ [BUFFER] Denoise insertion point ═══════════════════════════
    if let Some(ref denoised) = ops.denoised_buffer {
        let expected_len = (width * height * 4) as usize;
        if denoised.len() == expected_len {
            for (i, pixel) in output.pixels_mut().enumerate() {
                let base = i * 4;
                pixel[0] = denoised[base];
                pixel[1] = denoised[base + 1];
                pixel[2] = denoised[base + 2];
                pixel[3] = denoised[base + 3];
            }
        }
    }

    // ═══ Phase 2: Tonal range, Contrast, Tone Curve ════════════════
    for pixel in output.pixels_mut() {
        let mut r = pixel[0] as f64 / 255.0;
        let mut g_ch = pixel[1] as f64 / 255.0;
        let mut b = pixel[2] as f64 / 255.0;

        // --- Tonal range (highlights, shadows, whites, blacks) ---
        if has_tonal {
            let lum = 0.2126 * r + 0.7152 * g_ch + 0.0722 * b;
            if g.blacks != 0.0 {
                let w = smooth_step(0.15, 0.0, lum);
                let adj = g.blacks / 100.0 * 0.3 * w;
                r += adj; g_ch += adj; b += adj;
            }
            if g.shadows != 0.0 {
                let w = smooth_step(0.35, 0.0, lum);
                let adj = g.shadows / 100.0 * 0.4 * w;
                r += adj; g_ch += adj; b += adj;
            }
            if g.highlights != 0.0 {
                let w = smooth_step(0.65, 1.0, lum);
                let adj = g.highlights / 100.0 * 0.4 * w;
                r += adj; g_ch += adj; b += adj;
            }
            if g.whites != 0.0 {
                let w = smooth_step(0.85, 1.0, lum);
                let adj = g.whites / 100.0 * 0.3 * w;
                r += adj; g_ch += adj; b += adj;
            }
        }

        // --- Contrast ---
        if contrast_factor != 1.0 {
            r = (r - 0.5) * contrast_factor + 0.5;
            g_ch = (g_ch - 0.5) * contrast_factor + 0.5;
            b = (b - 0.5) * contrast_factor + 0.5;
        }

        // --- Tone curve ---
        if let Some(ref lut) = tone_lut {
            r = apply_curve_lut(lut, r);
            g_ch = apply_curve_lut(lut, g_ch);
            b = apply_curve_lut(lut, b);
        }

        pixel[0] = (r * 255.0).clamp(0.0, 255.0) as u8;
        pixel[1] = (g_ch * 255.0).clamp(0.0, 255.0) as u8;
        pixel[2] = (b * 255.0).clamp(0.0, 255.0) as u8;
    }

    // ═══ [BUFFER] HDRNet insertion point ════════════════════════════
    if let Some(ref coeffs) = ops.hdrnet_coeffs {
        // HDRNet bilateral grid: coeffs are [12, 16, 16, 8] flattened
        let grid_d = 16usize;
        let grid_l = 8usize;
        let num_coeffs = 12usize;

        if coeffs.len() >= num_coeffs * grid_d * grid_d * grid_l {
            for (idx, pixel) in output.pixels_mut().enumerate() {
                let px = (idx as u32) % width;
                let py = (idx as u32) / width;

                let r = pixel[0] as f32 / 255.0;
                let g_v = pixel[1] as f32 / 255.0;
                let b_v = pixel[2] as f32 / 255.0;
                let lum = 0.2126 * r + 0.7152 * g_v + 0.0722 * b_v;

                let gx = (px as f32 / width as f32) * (grid_d as f32 - 1.0);
                let gy = (py as f32 / height as f32) * (grid_d as f32 - 1.0);
                let gl = lum * (grid_l as f32 - 1.0);

                let c = trilinear_sample_grid(coeffs, num_coeffs, grid_d, grid_d, grid_l, gx, gy, gl);

                let out_r = (c[0] * r + c[1] * g_v + c[2] * b_v + c[3]).clamp(0.0, 1.0);
                let out_g = (c[4] * r + c[5] * g_v + c[6] * b_v + c[7]).clamp(0.0, 1.0);
                let out_b = (c[8] * r + c[9] * g_v + c[10] * b_v + c[11]).clamp(0.0, 1.0);

                pixel[0] = (out_r * 255.0) as u8;
                pixel[1] = (out_g * 255.0) as u8;
                pixel[2] = (out_b * 255.0) as u8;
            }
        }
    }

    // ═══ Phase 3: HSL, Sat/Vib, Dehaze, HDR, Clarity, Texture, Film, Vignette, Grain ═══
    for (idx, pixel) in output.pixels_mut().enumerate() {
        let px = (idx as u32) % width;
        let py = (idx as u32) / width;
        let mut r = pixel[0] as f64 / 255.0;
        let mut g_ch = pixel[1] as f64 / 255.0;
        let mut b = pixel[2] as f64 / 255.0;

        // --- HSL per-channel adjustments ---
        if has_hsl {
            let (h, s, l) = rgb_to_hsl(r, g_ch, b);
            let (dh, ds, dl) = get_hsl_adjustments(&g.hsl, h);
            let new_h = (h + dh).rem_euclid(360.0);
            let new_s = (s + ds / 100.0).clamp(0.0, 1.0);
            let new_l = (l + dl / 100.0 * 0.5).clamp(0.0, 1.0);
            let (nr, ng, nb) = hsl_to_rgb(new_h, new_s, new_l);
            r = nr; g_ch = ng; b = nb;
        }

        // --- Saturation + Vibrance ---
        if sat_factor != 1.0 || vib_factor != 0.0 {
            let lum = 0.2126 * r + 0.7152 * g_ch + 0.0722 * b;
            let mut factor = sat_factor;
            if vib_factor != 0.0 {
                let max_c = r.max(g_ch).max(b);
                let min_c = r.min(g_ch).min(b);
                let current_sat = if max_c > 0.0 { (max_c - min_c) / max_c } else { 0.0 };
                factor += vib_factor * (1.0 - current_sat);
            }
            r = lum + (r - lum) * factor;
            g_ch = lum + (g_ch - lum) * factor;
            b = lum + (b - lum) * factor;
        }

        // --- Dehaze ---
        if g.dehaze != 0.0 {
            let strength = g.dehaze / 100.0;
            let lum = 0.2126 * r + 0.7152 * g_ch + 0.0722 * b;
            let lift = -0.02 * strength;
            let gain = 1.0 + 0.15 * strength;
            r = (r * gain + lift).max(0.0);
            g_ch = (g_ch * gain + lift).max(0.0);
            b = (b * gain + lift).max(0.0);
            let sat_boost = 1.0 + 0.3 * strength * (1.0 - lum).max(0.0);
            let new_lum = 0.2126 * r + 0.7152 * g_ch + 0.0722 * b;
            r = new_lum + (r - new_lum) * sat_boost;
            g_ch = new_lum + (g_ch - new_lum) * sat_boost;
            b = new_lum + (b - new_lum) * sat_boost;
        }

        // --- HDR tone compression ---
        // Compresses dynamic range: lifts shadows, pulls down highlights,
        // then restores mid-tone contrast and boosts saturation.
        if hdr_strength > 0.0 {
            // Step 1: Pull all channels toward midtone (0.5)
            let pull = hdr_strength * 0.6;
            r = r + (0.5 - r) * pull;
            g_ch = g_ch + (0.5 - g_ch) * pull;
            b = b + (0.5 - b) * pull;

            // Step 2: Restore mid-tone contrast to prevent flat look
            let contrast_add = 1.0 + hdr_strength * 0.35;
            r = (r - 0.5) * contrast_add + 0.5;
            g_ch = (g_ch - 0.5) * contrast_add + 0.5;
            b = (b - 0.5) * contrast_add + 0.5;

            // Step 3: Boost saturation for HDR vibrancy
            let new_lum = 0.2126 * r + 0.7152 * g_ch + 0.0722 * b;
            let sat_boost = 1.0 + hdr_strength * 0.2;
            r = new_lum + (r - new_lum) * sat_boost;
            g_ch = new_lum + (g_ch - new_lum) * sat_boost;
            b = new_lum + (b - new_lum) * sat_boost;
        }

        // --- Clarity / Texture (approximate with local contrast boost) ---
        if clarity_strength != 0.0 || texture_strength != 0.0 {
            let lum = 0.2126 * r + 0.7152 * g_ch + 0.0722 * b;
            // Clarity: boost mid-tone contrast (centered around 0.5)
            if clarity_strength != 0.0 {
                let mid_weight = 1.0 - (2.0 * (lum - 0.5)).abs().min(1.0);
                let boost = clarity_strength * mid_weight * 0.3;
                let diff = lum - 0.5;
                let adj = diff * boost;
                r += adj; g_ch += adj; b += adj;
            }
            // Texture: boost fine detail (similar to clarity but affects all tones)
            if texture_strength != 0.0 {
                let diff = lum - 0.5;
                let adj = diff * texture_strength * 0.15;
                r += adj; g_ch += adj; b += adj;
            }
        }

        // --- Film simulation ---
        if let Some((ref look, strength, grain_amt, grain_sz, ref bw_filter)) = film {
            let orig_r = r;
            let orig_g = g_ch;
            let orig_b = b;

            // Color matrix transform
            let mr = look.matrix[0] * r + look.matrix[1] * g_ch + look.matrix[2] * b;
            let mg = look.matrix[3] * r + look.matrix[4] * g_ch + look.matrix[5] * b;
            let mb = look.matrix[6] * r + look.matrix[7] * g_ch + look.matrix[8] * b;
            r = mr; g_ch = mg; b = mb;

            // Base curve (contrast adjustment)
            let curve_factor = look.contrast_factor;
            if curve_factor != 1.0 {
                r = (r - 0.5) * curve_factor + 0.5;
                g_ch = (g_ch - 0.5) * curve_factor + 0.5;
                b = (b - 0.5) * curve_factor + 0.5;
            }

            // Film saturation
            if look.saturation != 1.0 {
                let lum = 0.2126 * r + 0.7152 * g_ch + 0.0722 * b;
                r = lum + (r - lum) * look.saturation;
                g_ch = lum + (g_ch - lum) * look.saturation;
                b = lum + (b - lum) * look.saturation;
            }

            // B&W conversion with color filter
            if look.is_bw {
                let (rw, gw, bw) = match bw_filter.as_deref() {
                    Some("R") => (0.6, 0.3, 0.1),   // Red filter: brightens reds, darkens blues
                    Some("Y") => (0.4, 0.45, 0.15),  // Yellow filter: warm tones brighter
                    Some("G") => (0.2, 0.6, 0.2),    // Green filter: brightens foliage
                    _ => (0.299, 0.587, 0.114),       // Standard luminance
                };
                let mono = rw * r + gw * g_ch + bw * b;
                r = mono; g_ch = mono; b = mono;
            }

            // Grain (deterministic noise based on pixel position)
            if grain_amt > 0.0 {
                let noise = pixel_noise(px, py, grain_sz as f64) * grain_amt as f64 / 100.0 * 0.15;
                r += noise; g_ch += noise; b += noise;
            }

            // Blend with original based on strength
            if strength < 1.0 {
                r = orig_r + (r - orig_r) * strength;
                g_ch = orig_g + (g_ch - orig_g) * strength;
                b = orig_b + (b - orig_b) * strength;
            }
        }

        // --- Effects: Post-crop vignette ---
        if vignette_amount != 0.0 {
            let dx = (px as f64 - cx) / cx;
            let dy = (py as f64 - cy) / cy;
            let dist = (dx * dx + dy * dy).sqrt();
            let falloff = (dist * dist * 0.5).min(1.0);
            let vig = 1.0 - falloff * vignette_amount / 100.0;
            r *= vig; g_ch *= vig; b *= vig;
        }

        // --- Effects: Grain (standalone, not film sim) ---
        if grain_amount > 0.0 && film.is_none() {
            let noise = pixel_noise(px, py, grain_size as f64) * grain_amount as f64 / 100.0 * 0.15;
            r += noise; g_ch += noise; b += noise;
        }

        pixel[0] = (r * 255.0).clamp(0.0, 255.0) as u8;
        pixel[1] = (g_ch * 255.0).clamp(0.0, 255.0) as u8;
        pixel[2] = (b * 255.0).clamp(0.0, 255.0) as u8;
    }

    DynamicImage::ImageRgba8(output)
}

// ─── Tone Curve ────────────────────────────────────────────────

fn build_tone_curve_lut(tc: &crate::models::edit_state::ToneCurve) -> Option<[f64; 256]> {
    if tc.mode == "POINT" && tc.points.len() <= 2 {
        // Default curve (0,0)→(1,1) — identity, skip
        if tc.points.len() == 2
            && (tc.points[0].x - 0.0).abs() < 0.001
            && (tc.points[0].y - 0.0).abs() < 0.001
            && (tc.points[1].x - 1.0).abs() < 0.001
            && (tc.points[1].y - 1.0).abs() < 0.001
        {
            return None;
        }
    }

    let mut lut = [0.0f64; 256];
    let points = &tc.points;
    if points.is_empty() { return None; }

    for i in 0..256 {
        let x = i as f64 / 255.0;
        lut[i] = interpolate_curve(points, x).clamp(0.0, 1.0);
    }
    Some(lut)
}

fn interpolate_curve(points: &[crate::models::edit_state::CurvePoint], x: f64) -> f64 {
    if points.is_empty() { return x; }
    if points.len() == 1 { return points[0].y; }
    if x <= points[0].x { return points[0].y; }
    if x >= points[points.len() - 1].x { return points[points.len() - 1].y; }

    // Find segment
    let mut i = 0;
    while i < points.len() - 1 && points[i + 1].x < x { i += 1; }

    let p0 = &points[i];
    let p1 = &points[(i + 1).min(points.len() - 1)];
    let range = p1.x - p0.x;
    if range < 0.0001 { return p0.y; }

    let t = (x - p0.x) / range;
    // Hermite spline for smooth interpolation
    let t2 = t * t;
    let t3 = t2 * t;
    let h1 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h2 = -2.0 * t3 + 3.0 * t2;
    h1 * p0.y + h2 * p1.y
}

fn apply_curve_lut(lut: &[f64; 256], v: f64) -> f64 {
    let idx = (v * 255.0).clamp(0.0, 255.0);
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(255);
    let frac = idx - lo as f64;
    lut[lo] * (1.0 - frac) + lut[hi] * frac
}

// ─── HSL Color Grading ────────────────────────────────────────

fn rgb_to_hsl(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let r = r.clamp(0.0, 1.0);
    let g = g.clamp(0.0, 1.0);
    let b = b.clamp(0.0, 1.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    if (max - min).abs() < 0.0001 {
        return (0.0, 0.0, l);
    }
    let d = max - min;
    let s = if l > 0.5 { d / (2.0 - max - min) } else { d / (max + min) };
    let h = if (max - r).abs() < 0.0001 {
        ((g - b) / d + if g < b { 6.0 } else { 0.0 }) * 60.0
    } else if (max - g).abs() < 0.0001 {
        ((b - r) / d + 2.0) * 60.0
    } else {
        ((r - g) / d + 4.0) * 60.0
    };
    (h, s, l)
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (f64, f64, f64) {
    if s < 0.0001 { return (l, l, l); }
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let hn = h / 360.0;
    (
        hue_to_rgb(p, q, hn + 1.0 / 3.0),
        hue_to_rgb(p, q, hn),
        hue_to_rgb(p, q, hn - 1.0 / 3.0),
    )
}

fn hue_to_rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 1.0 / 2.0 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    p
}

/// Get weighted HSL adjustments for a given hue angle
fn get_hsl_adjustments(hsl: &std::collections::HashMap<String, crate::models::edit_state::HslValues>, hue: f64) -> (f64, f64, f64) {
    // Hue ranges for each channel (center, width)
    let channels: &[(&str, f64)] = &[
        ("red", 0.0), ("orange", 30.0), ("yellow", 60.0), ("green", 120.0),
        ("aqua", 180.0), ("blue", 240.0), ("purple", 280.0), ("magenta", 320.0),
    ];
    let mut dh = 0.0;
    let mut ds = 0.0;
    let mut dl = 0.0;

    for (name, center) in channels {
        if let Some(vals) = hsl.get(*name) {
            if vals.h == 0.0 && vals.s == 0.0 && vals.l == 0.0 { continue; }
            let w = hue_weight(hue, *center, 30.0);
            if w > 0.001 {
                dh += vals.h * w;
                ds += vals.s * w;
                dl += vals.l * w;
            }
        }
    }
    (dh, ds, dl)
}

fn hue_weight(hue: f64, center: f64, width: f64) -> f64 {
    let mut diff = (hue - center).abs();
    if diff > 180.0 { diff = 360.0 - diff; }
    if diff > width { 0.0 } else { 1.0 - diff / width }
}

// ─── Film Simulation ──────────────────────────────────────────

struct FilmLook {
    matrix: [f64; 9],
    contrast_factor: f64,
    saturation: f64,
    is_bw: bool,
}

fn get_film_look(fs: &crate::models::edit_state::FilmSimState) -> FilmLook {
    match fs.id.as_str() {
        "chrome_like" => FilmLook {
            matrix: [1.1, -0.05, -0.05, -0.02, 1.05, -0.03, 0.0, -0.02, 1.02],
            contrast_factor: 1.15, saturation: 1.1, is_bw: false,
        },
        "velvia_like" => FilmLook {
            matrix: [1.15, -0.08, -0.07, -0.03, 1.08, -0.05, 0.01, -0.03, 1.02],
            contrast_factor: 1.3, saturation: 1.3, is_bw: false,
        },
        "provia_like" => FilmLook {
            matrix: [1.05, -0.03, -0.02, -0.01, 1.03, -0.02, 0.0, -0.01, 1.01],
            contrast_factor: 1.15, saturation: 1.05, is_bw: false,
        },
        "eterna_like" => FilmLook {
            matrix: [1.02, -0.01, -0.01, -0.01, 1.02, -0.01, 0.0, -0.01, 1.01],
            contrast_factor: 0.9, saturation: 0.9, is_bw: false,
        },
        "astia_like" => FilmLook {
            matrix: [1.05, -0.03, -0.02, -0.01, 1.04, -0.03, 0.0, -0.01, 1.01],
            contrast_factor: 1.05, saturation: 1.0, is_bw: false,
        },
        "acros_like" => FilmLook {
            matrix: [0.299, 0.587, 0.114, 0.299, 0.587, 0.114, 0.299, 0.587, 0.114],
            contrast_factor: 1.2, saturation: 0.0, is_bw: true,
        },
        _ => FilmLook {
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            contrast_factor: 1.0, saturation: 1.0, is_bw: false,
        },
    }
}

/// Deterministic pixel noise for film grain
fn pixel_noise(x: u32, y: u32, size: f64) -> f64 {
    // Hash-based noise (deterministic per-pixel)
    let sx = (x as f64 / size.max(0.5)) as u32;
    let sy = (y as f64 / size.max(0.5)) as u32;
    let mut h = sx.wrapping_mul(374761393).wrapping_add(sy.wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    h ^= h >> 16;
    (h as f64 / u32::MAX as f64) * 2.0 - 1.0 // range [-1, 1]
}

// ─── HDRNet Grid Sampling (for buffer ops) ───────────────────

/// Trilinear interpolation in a bilateral grid for HDRNet buffer insertion
fn trilinear_sample_grid(
    grid: &[f32],
    num_coeffs: usize,
    gw: usize,
    gh: usize,
    gd: usize,
    x: f32,
    y: f32,
    z: f32,
) -> Vec<f32> {
    let x0 = (x.floor() as usize).min(gw - 1);
    let x1 = (x0 + 1).min(gw - 1);
    let y0 = (y.floor() as usize).min(gh - 1);
    let y1 = (y0 + 1).min(gh - 1);
    let z0 = (z.floor() as usize).min(gd - 1);
    let z1 = (z0 + 1).min(gd - 1);

    let fx = x - x.floor();
    let fy = y - y.floor();
    let fz = z - z.floor();

    let mut result = vec![0.0f32; num_coeffs];

    for c in 0..num_coeffs {
        let idx = |cc: usize, yy: usize, xx: usize, zz: usize| -> usize {
            cc * gh * gw * gd + yy * gw * gd + xx * gd + zz
        };

        let v000 = grid.get(idx(c, y0, x0, z0)).copied().unwrap_or(0.0);
        let v001 = grid.get(idx(c, y0, x0, z1)).copied().unwrap_or(0.0);
        let v010 = grid.get(idx(c, y0, x1, z0)).copied().unwrap_or(0.0);
        let v011 = grid.get(idx(c, y0, x1, z1)).copied().unwrap_or(0.0);
        let v100 = grid.get(idx(c, y1, x0, z0)).copied().unwrap_or(0.0);
        let v101 = grid.get(idx(c, y1, x0, z1)).copied().unwrap_or(0.0);
        let v110 = grid.get(idx(c, y1, x1, z0)).copied().unwrap_or(0.0);
        let v111 = grid.get(idx(c, y1, x1, z1)).copied().unwrap_or(0.0);

        let c00 = v000 * (1.0 - fz) + v001 * fz;
        let c01 = v010 * (1.0 - fz) + v011 * fz;
        let c10 = v100 * (1.0 - fz) + v101 * fz;
        let c11 = v110 * (1.0 - fz) + v111 * fz;

        let c0 = c00 * (1.0 - fx) + c01 * fx;
        let c1 = c10 * (1.0 - fx) + c11 * fx;

        result[c] = c0 * (1.0 - fy) + c1 * fy;
    }

    result
}

// ─── Helpers ──────────────────────────────────────────────────

/// Smooth interpolation weight for tonal range selection
fn smooth_step(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = if edge0 < edge1 {
        ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    } else {
        ((edge0 - x) / (edge0 - edge1)).clamp(0.0, 1.0)
    };
    t * t * (3.0 - 2.0 * t)
}

fn compute_histogram(img: &DynamicImage) -> HistogramData {
    let rgba = img.to_rgba8();
    let mut r = vec![0u32; 256];
    let mut g = vec![0u32; 256];
    let mut b = vec![0u32; 256];
    let mut lum = vec![0u32; 256];

    for pixel in rgba.pixels() {
        r[pixel[0] as usize] += 1;
        g[pixel[1] as usize] += 1;
        b[pixel[2] as usize] += 1;
        let l = (0.2126 * pixel[0] as f64 + 0.7152 * pixel[1] as f64 + 0.0722 * pixel[2] as f64) as usize;
        lum[l.min(255)] += 1;
    }

    HistogramData { r, g, b, lum }
}

fn encode_jpeg(img: &DynamicImage, quality: u8) -> Result<Vec<u8>, AppError> {
    let mut buf = Cursor::new(Vec::new());
    let rgb = img.to_rgb8();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    encoder
        .encode(rgb.as_raw(), rgb.width(), rgb.height(), image::ExtendedColorType::Rgb8)
        .map_err(|e| AppError::Internal(format!("JPEG encode error: {}", e)))?;
    Ok(buf.into_inner())
}

fn encode_png(img: &DynamicImage) -> Result<Vec<u8>, AppError> {
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, ImageFormat::Png)
        .map_err(|e| AppError::Internal(format!("PNG encode error: {}", e)))?;
    Ok(buf.into_inner())
}

fn encode_tiff(img: &DynamicImage) -> Result<Vec<u8>, AppError> {
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, ImageFormat::Tiff)
        .map_err(|e| AppError::Internal(format!("TIFF encode error: {}", e)))?;
    Ok(buf.into_inner())
}
