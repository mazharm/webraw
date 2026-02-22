use std::io::Cursor;
use std::sync::Arc;
use base64::Engine as _;
use image::{DynamicImage, ImageFormat, GenericImageView, imageops::FilterType};
use tokio::sync::Semaphore;

use crate::models::config::AppConfig;
use crate::models::edit_state::EditState;
use crate::models::error::AppError;
use crate::services::cache::FileCache;

#[allow(dead_code)]
pub struct RenderService {
    config: Arc<AppConfig>,
    render_semaphore: Semaphore,
    export_semaphore: Semaphore,
}

impl RenderService {
    pub fn new(config: Arc<AppConfig>) -> Self {
        Self {
            render_semaphore: Semaphore::new(config.max_parallel_renders),
            export_semaphore: Semaphore::new(config.max_parallel_exports),
            config,
        }
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

        let data = cache.read_file_bytes(file_id).await?;
        let img = load_image(&data)?;

        let thumb = resize_image(&img, max_edge);
        encode_jpeg(&thumb, 85)
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

        let data = cache.read_file_bytes(file_id).await?;
        let img = load_image(&data)?;

        let preview = resize_image(&img, max_edge);
        let rendered = apply_edit_state(&preview, edit_state);
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

        let data = cache.read_file_bytes(file_id).await?;
        let img = load_image(&data)?;

        let base = resize_image(&img, max_edge);
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
    image::load_from_memory(data).map_err(|e| {
        if e.to_string().contains("unsupported") {
            AppError::RawUnsupportedCamera(e.to_string())
        } else {
            AppError::RawCorruptData(e.to_string())
        }
    })
}

fn resize_image(img: &DynamicImage, max_edge: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w <= max_edge && h <= max_edge {
        return img.clone();
    }
    let scale = max_edge as f64 / w.max(h) as f64;
    let nw = (w as f64 * scale) as u32;
    let nh = (h as f64 * scale) as u32;
    img.resize_exact(nw, nh, FilterType::Lanczos3)
}

fn apply_edit_state(img: &DynamicImage, state: &EditState) -> DynamicImage {
    let mut result = img.clone();

    // Apply exposure adjustment
    if state.global.exposure != 0.0 {
        let factor = 2.0_f64.powf(state.global.exposure);
        result = adjust_brightness(&result, factor);
    }

    // Apply contrast
    if state.global.contrast != 0.0 {
        result = adjust_contrast(&result, state.global.contrast);
    }

    // Apply saturation
    if state.global.saturation != 0.0 {
        result = adjust_saturation(&result, state.global.saturation);
    }

    // Apply vibrance
    if state.global.vibrance != 0.0 {
        result = adjust_saturation(&result, state.global.vibrance * 0.5);
    }

    // Apply temperature shift
    if state.global.temperature != 5500.0 {
        result = adjust_temperature(&result, state.global.temperature);
    }

    result
}

fn adjust_brightness(img: &DynamicImage, factor: f64) -> DynamicImage {
    let rgba = img.to_rgba8();
    let mut output = rgba.clone();
    for pixel in output.pixels_mut() {
        pixel[0] = (pixel[0] as f64 * factor).clamp(0.0, 255.0) as u8;
        pixel[1] = (pixel[1] as f64 * factor).clamp(0.0, 255.0) as u8;
        pixel[2] = (pixel[2] as f64 * factor).clamp(0.0, 255.0) as u8;
    }
    DynamicImage::ImageRgba8(output)
}

fn adjust_contrast(img: &DynamicImage, amount: f64) -> DynamicImage {
    let factor = (100.0 + amount) / 100.0;
    let rgba = img.to_rgba8();
    let mut output = rgba.clone();
    for pixel in output.pixels_mut() {
        for c in 0..3 {
            let v = pixel[c] as f64 / 255.0;
            let adjusted = ((v - 0.5) * factor + 0.5).clamp(0.0, 1.0);
            pixel[c] = (adjusted * 255.0) as u8;
        }
    }
    DynamicImage::ImageRgba8(output)
}

fn adjust_saturation(img: &DynamicImage, amount: f64) -> DynamicImage {
    let factor = 1.0 + amount / 100.0;
    let rgba = img.to_rgba8();
    let mut output = rgba.clone();
    for pixel in output.pixels_mut() {
        let r = pixel[0] as f64 / 255.0;
        let g = pixel[1] as f64 / 255.0;
        let b = pixel[2] as f64 / 255.0;
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        pixel[0] = ((lum + (r - lum) * factor) * 255.0).clamp(0.0, 255.0) as u8;
        pixel[1] = ((lum + (g - lum) * factor) * 255.0).clamp(0.0, 255.0) as u8;
        pixel[2] = ((lum + (b - lum) * factor) * 255.0).clamp(0.0, 255.0) as u8;
    }
    DynamicImage::ImageRgba8(output)
}

fn adjust_temperature(img: &DynamicImage, temp: f64) -> DynamicImage {
    let shift = (temp - 5500.0) / 5500.0;
    let rgba = img.to_rgba8();
    let mut output = rgba.clone();
    for pixel in output.pixels_mut() {
        // Warm: boost red, reduce blue. Cool: opposite.
        pixel[0] = (pixel[0] as f64 * (1.0 + shift * 0.1)).clamp(0.0, 255.0) as u8;
        pixel[2] = (pixel[2] as f64 * (1.0 - shift * 0.1)).clamp(0.0, 255.0) as u8;
    }
    DynamicImage::ImageRgba8(output)
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
