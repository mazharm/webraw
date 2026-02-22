use std::collections::HashMap;
use std::path::{Path, PathBuf};

use image::{DynamicImage, GenericImageView, imageops::FilterType};
use parking_lot::Mutex;

use crate::models::error::AppError;
use crate::models::optimize::*;

// ---------------------------------------------------------------------------
// Optimize ONNX model adapter names
// ---------------------------------------------------------------------------

const ADAPTER_NAFNET: &str = "nafnet";
const ADAPTER_RESTORMER: &str = "restormer";
const ADAPTER_HDRNET: &str = "hdrnet";
const ADAPTER_MOBILESAM_ENCODER: &str = "mobilesam_encoder";
const ADAPTER_MOBILESAM_DECODER: &str = "mobilesam_decoder";
const ADAPTER_BISENET: &str = "bisenet";

const OPTIMIZE_ADAPTERS: &[&str] = &[
    ADAPTER_NAFNET,
    ADAPTER_RESTORMER,
    ADAPTER_HDRNET,
    ADAPTER_MOBILESAM_ENCODER,
    ADAPTER_MOBILESAM_DECODER,
    ADAPTER_BISENET,
];

// ---------------------------------------------------------------------------
// Session holder (lazy-loaded)
// ---------------------------------------------------------------------------

struct OnnxSession {
    session: ort::session::Session,
    manifest: OptimizeModelManifest,
}

/// Pending model info for lazy loading — stores manifest + path until first use.
struct PendingModel {
    manifest: OptimizeModelManifest,
    onnx_path: PathBuf,
}

/// Holds either a lazily-loaded session or pending info for on-demand loading.
enum LazySession {
    Pending(PendingModel),
    Loaded(OnnxSession),
    Unavailable,
}

// ---------------------------------------------------------------------------
// OptimizeService
// ---------------------------------------------------------------------------

pub struct OptimizeService {
    nafnet: Mutex<LazySession>,
    restormer: Mutex<LazySession>,
    hdrnet: Mutex<LazySession>,
    mobilesam_encoder: Mutex<LazySession>,
    mobilesam_decoder: Mutex<LazySession>,
    bisenet: Mutex<LazySession>,
    model_statuses: Vec<OptimizeModelStatus>,
}

impl OptimizeService {
    pub fn new(models_dir: &str) -> Self {
        let models_path = Path::new(models_dir);
        let mut nafnet = LazySession::Unavailable;
        let mut restormer = LazySession::Unavailable;
        let mut hdrnet = LazySession::Unavailable;
        let mut mobilesam_encoder = LazySession::Unavailable;
        let mut mobilesam_decoder = LazySession::Unavailable;
        let mut bisenet = LazySession::Unavailable;
        let mut model_statuses = Vec::new();
        let mut found_adapters = Vec::new();

        // Scan for optimize model manifests (but don't load ONNX sessions yet)
        if let Ok(entries) = std::fs::read_dir(models_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "json") {
                    if let Ok(json_str) = std::fs::read_to_string(&path) {
                        if let Ok(manifest) = serde_json::from_str::<OptimizeModelManifest>(&json_str) {
                            if !OPTIMIZE_ADAPTERS.contains(&manifest.adapter.as_str()) {
                                continue;
                            }

                            let onnx_path = models_path.join(&manifest.file);
                            let file_size_mb = std::fs::metadata(&onnx_path)
                                .ok()
                                .map(|m| m.len() as f64 / 1_048_576.0);
                            let available = onnx_path.exists();

                            model_statuses.push(OptimizeModelStatus {
                                id: manifest.id.clone(),
                                name: manifest.name.clone(),
                                available,
                                file_size_mb,
                            });

                            found_adapters.push(manifest.adapter.clone());

                            if !available {
                                tracing::debug!(
                                    "Optimize model '{}' ONNX file not found: {:?}",
                                    manifest.id,
                                    onnx_path
                                );
                                continue;
                            }

                            // Register for lazy loading instead of loading now
                            let pending = LazySession::Pending(PendingModel {
                                manifest: manifest.clone(),
                                onnx_path: onnx_path.clone(),
                            });
                            match manifest.adapter.as_str() {
                                ADAPTER_NAFNET => { nafnet = pending; }
                                ADAPTER_RESTORMER => { restormer = pending; }
                                ADAPTER_HDRNET => { hdrnet = pending; }
                                ADAPTER_MOBILESAM_ENCODER => { mobilesam_encoder = pending; }
                                ADAPTER_MOBILESAM_DECODER => { mobilesam_decoder = pending; }
                                ADAPTER_BISENET => { bisenet = pending; }
                                _ => {}
                            }
                            tracing::info!(
                                "Registered optimize model '{}' for lazy loading from {:?}",
                                manifest.id,
                                onnx_path
                            );
                        }
                    }
                }
            }
        } else {
            tracing::debug!(
                "Models directory '{}' not found; optimize models unavailable",
                models_dir
            );
        }

        // Add default statuses for adapters not found in any manifest
        let known_models = [
            (ADAPTER_NAFNET, "NAFNet Denoise"),
            (ADAPTER_RESTORMER, "Restormer Denoise"),
            (ADAPTER_HDRNET, "HDRNet Enhance"),
            (ADAPTER_MOBILESAM_ENCODER, "MobileSAM Encoder"),
            (ADAPTER_MOBILESAM_DECODER, "MobileSAM Decoder"),
            (ADAPTER_BISENET, "BiSeNet Face Parse"),
        ];
        for (adapter, name) in known_models {
            if !found_adapters.iter().any(|a| a == adapter) {
                model_statuses.push(OptimizeModelStatus {
                    id: adapter.to_string(),
                    name: name.to_string(),
                    available: false,
                    file_size_mb: None,
                });
            }
        }

        let available_count = model_statuses.iter().filter(|s| s.available).count();
        tracing::info!(
            "OptimizeService registered {}/6 optimize models (lazy-loaded on first use)",
            available_count
        );

        Self {
            nafnet: Mutex::new(nafnet),
            restormer: Mutex::new(restormer),
            hdrnet: Mutex::new(hdrnet),
            mobilesam_encoder: Mutex::new(mobilesam_encoder),
            mobilesam_decoder: Mutex::new(mobilesam_decoder),
            bisenet: Mutex::new(bisenet),
            model_statuses,
        }
    }

    fn load_session(onnx_path: &Path) -> Result<ort::session::Session, String> {
        ort::session::Session::builder()
            .and_then(|b| b.with_intra_threads(2))
            .and_then(|b| b.commit_from_file(onnx_path))
            .map_err(|e| format!("ONNX session load error: {}", e))
    }

    /// Ensure a lazy session is loaded, returning a mutable ref to the OnnxSession if available.
    fn ensure_loaded(slot: &mut LazySession) -> Option<&mut OnnxSession> {
        // If pending, attempt to load now
        if let LazySession::Pending(_) = slot {
            let old = std::mem::replace(slot, LazySession::Unavailable);
            if let LazySession::Pending(pending) = old {
                match Self::load_session(&pending.onnx_path) {
                    Ok(session) => {
                        tracing::info!(
                            "Lazy-loaded optimize model '{}' from {:?}",
                            pending.manifest.id,
                            pending.onnx_path
                        );
                        *slot = LazySession::Loaded(OnnxSession {
                            session,
                            manifest: pending.manifest,
                        });
                    }
                    Err(e) => {
                        tracing::warn!("Failed to lazy-load optimize model: {}", e);
                        *slot = LazySession::Unavailable;
                    }
                }
            }
        }

        match slot {
            LazySession::Loaded(ref mut s) => Some(s),
            _ => None,
        }
    }

    /// Returns status of all optimize models
    pub fn available_models(&self) -> Vec<OptimizeModelStatus> {
        self.model_statuses.clone()
    }

    /// Run the full optimize pipeline
    pub fn run_optimize(
        &self,
        img: &DynamicImage,
        config: &OptimizeConfig,
        progress_cb: &dyn Fn(f64),
    ) -> Result<OptimizeResult, AppError> {
        let (_w, _h) = img.dimensions();
        let mut working = img.clone();

        // --- Phase 1: Compute masks (5%) ---
        let masks = if config.masks {
            progress_cb(0.05);
            self.compute_masks(&working)?
        } else {
            progress_cb(0.05);
            MaskSet::default()
        };

        // --- Phase 2: Tiled denoise (15-55%) ---
        if config.denoise {
            progress_cb(0.15);
            working = self.run_tiled_denoise(&working, &|p| {
                progress_cb(0.15 + p * 0.40);
            })?;
        } else {
            progress_cb(0.55);
        }

        // --- Phase 3: HDRNet enhance (55-85%) ---
        if config.enhance {
            progress_cb(0.55);
            working = self.run_hdrnet(&working, &|p| {
                progress_cb(0.55 + p * 0.30);
            })?;
        } else {
            progress_cb(0.85);
        }

        // --- Phase 4: Guardrails (85-95%) ---
        progress_cb(0.85);
        self.apply_guardrails(&mut working, img, &masks);
        progress_cb(0.95);

        // --- Phase 5: Blend with original based on strength ---
        if config.strength < 1.0 {
            working = blend_images(img, &working, config.strength);
        }

        // --- Phase 6: Decompose to params ---
        let applied_params = self.decompose_to_params(img, &working);

        // Encode result as PNG
        let mut buf = std::io::Cursor::new(Vec::new());
        working
            .write_to(&mut buf, image::ImageFormat::Png)
            .map_err(|e| AppError::OptimizeError(format!("Failed to encode result: {}", e)))?;

        progress_cb(1.0);

        Ok(OptimizeResult {
            image_data: buf.into_inner(),
            mime_type: "image/png".to_string(),
            masks,
            applied_params,
        })
    }

    /// Run mask computation only
    pub fn compute_masks_only(
        &self,
        img: &DynamicImage,
    ) -> Result<MaskSet, AppError> {
        self.compute_masks(img)
    }

    // ─── Internal: Mask Computation ──────────────────────────────────

    fn compute_masks(&self, img: &DynamicImage) -> Result<MaskSet, AppError> {
        let mut masks = MaskSet::default();

        // MobileSAM subject detection (1024px proxy)
        {
            let mut encoder_guard = self.mobilesam_encoder.lock();
            let mut decoder_guard = self.mobilesam_decoder.lock();
            if let (Some(enc), Some(dec)) = (
                Self::ensure_loaded(&mut encoder_guard),
                Self::ensure_loaded(&mut decoder_guard),
            ) {
                match self.run_mobilesam(img, enc, dec) {
                    Ok(mask) => { masks.subject = Some(mask); }
                    Err(e) => { tracing::warn!("MobileSAM failed: {}", e); }
                }
            }
        }

        // BiSeNet face parsing for sky + skin masks (512px proxy)
        {
            let mut bisenet_guard = self.bisenet.lock();
            if let Some(bise) = Self::ensure_loaded(&mut bisenet_guard) {
                match self.run_bisenet(img, bise) {
                    Ok((sky, skin)) => {
                        masks.sky = sky;
                        masks.skin = skin;
                    }
                    Err(e) => { tracing::warn!("BiSeNet failed: {}", e); }
                }
            }
        }

        Ok(masks)
    }

    fn run_mobilesam(
        &self,
        img: &DynamicImage,
        encoder: &mut OnnxSession,
        decoder: &mut OnnxSession,
    ) -> Result<MaskData, AppError> {
        let (ow, oh) = img.dimensions();
        let proxy_size = 1024u32;
        let proxy = img.resize(proxy_size, proxy_size, FilterType::Triangle);
        let (pw, ph) = proxy.dimensions();

        // Prepare NCHW tensor for encoder
        let rgb = proxy.to_rgb8();
        let mut input = ndarray::Array4::<f32>::zeros((1, 3, ph as usize, pw as usize));
        for y in 0..ph as usize {
            for x in 0..pw as usize {
                let p = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    input[[0, c, y, x]] = p[c] as f32 / 255.0;
                }
            }
        }

        // Run encoder
        let input_tensor = ort::value::Tensor::from_array(input)
            .map_err(|e| AppError::OptimizeError(format!("MobileSAM encoder tensor: {}", e)))?;
        let enc_outputs = encoder.session.run(ort::inputs![input_tensor])
            .map_err(|e| AppError::OptimizeError(format!("MobileSAM encoder: {}", e)))?;

        // Center point prompt (image center)
        let point_coords = ndarray::Array3::<f32>::from_shape_vec(
            (1, 1, 2),
            vec![pw as f32 / 2.0, ph as f32 / 2.0],
        ).map_err(|e| AppError::OptimizeError(format!("Point coords: {}", e)))?;
        let point_labels = ndarray::Array2::<f32>::from_shape_vec(
            (1, 1),
            vec![1.0],
        ).map_err(|e| AppError::OptimizeError(format!("Point labels: {}", e)))?;
        let mask_input = ndarray::Array4::<f32>::zeros((1, 1, 256, 256));
        let has_mask_input = ndarray::Array1::<f32>::from_vec(vec![0.0]);

        let coords_tensor = ort::value::Tensor::from_array(point_coords)
            .map_err(|e| AppError::OptimizeError(format!("Coords tensor: {}", e)))?;
        let labels_tensor = ort::value::Tensor::from_array(point_labels)
            .map_err(|e| AppError::OptimizeError(format!("Labels tensor: {}", e)))?;
        let mask_tensor = ort::value::Tensor::from_array(mask_input)
            .map_err(|e| AppError::OptimizeError(format!("Mask tensor: {}", e)))?;
        let has_mask_tensor = ort::value::Tensor::from_array(has_mask_input)
            .map_err(|e| AppError::OptimizeError(format!("HasMask tensor: {}", e)))?;

        // Run decoder with image embeddings + point prompt
        let dec_inputs = ort::inputs![
            "image_embeddings" => &enc_outputs[0],
            "point_coords" => coords_tensor,
            "point_labels" => labels_tensor,
            "mask_input" => mask_tensor,
            "has_mask_input" => has_mask_tensor,
        ];

        let dec_outputs = decoder.session.run(dec_inputs)
            .map_err(|e| AppError::OptimizeError(format!("MobileSAM decoder: {}", e)))?;

        let (_shape, mask_data) = dec_outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AppError::OptimizeError(format!("Extract mask: {}", e)))?;

        // Upscale mask to original size using bilinear interpolation
        let mask_vec = mask_data.to_vec();
        let upscaled = bilinear_upsample_mask(&mask_vec, 256, 256, ow, oh);

        Ok(MaskData {
            mask_type: "subject".to_string(),
            width: ow,
            height: oh,
            data: upscaled,
        })
    }

    fn run_bisenet(
        &self,
        img: &DynamicImage,
        bise: &mut OnnxSession,
    ) -> Result<(Option<MaskData>, Option<MaskData>), AppError> {
        let (ow, oh) = img.dimensions();
        let proxy = img.resize_exact(512, 512, FilterType::Triangle);
        let rgb = proxy.to_rgb8();

        let mut input = ndarray::Array4::<f32>::zeros((1, 3, 512, 512));
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        for y in 0..512usize {
            for x in 0..512usize {
                let p = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    input[[0, c, y, x]] = (p[c] as f32 / 255.0 - mean[c]) / std[c];
                }
            }
        }

        let input_tensor = ort::value::Tensor::from_array(input)
            .map_err(|e| AppError::OptimizeError(format!("BiSeNet tensor: {}", e)))?;
        let outputs = bise.session.run(ort::inputs![input_tensor])
            .map_err(|e| AppError::OptimizeError(format!("BiSeNet inference: {}", e)))?;

        let (_shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AppError::OptimizeError(format!("Extract BiSeNet output: {}", e)))?;
        let logits_vec = logits.to_vec();

        // BiSeNet face parsing labels: skin=1, sky=10 (typical CelebAMask-HQ labels)
        let num_classes = 19usize;
        let sky_class = 10usize;
        let skin_classes = [1usize]; // skin

        // Argmax to get class map
        let mut sky_mask = vec![0.0f32; 512 * 512];
        let mut skin_mask = vec![0.0f32; 512 * 512];

        for y in 0..512usize {
            for x in 0..512usize {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_class = 0usize;
                for c in 0..num_classes {
                    let idx = c * 512 * 512 + y * 512 + x;
                    if idx < logits_vec.len() && logits_vec[idx] > max_val {
                        max_val = logits_vec[idx];
                        max_class = c;
                    }
                }
                let pixel_idx = y * 512 + x;
                if max_class == sky_class {
                    sky_mask[pixel_idx] = 1.0;
                }
                if skin_classes.contains(&max_class) {
                    skin_mask[pixel_idx] = 1.0;
                }
            }
        }

        let has_sky = sky_mask.iter().any(|&v| v > 0.5);
        let has_skin = skin_mask.iter().any(|&v| v > 0.5);

        let sky = if has_sky {
            Some(MaskData {
                mask_type: "sky".to_string(),
                width: ow,
                height: oh,
                data: bilinear_upsample_mask(&sky_mask, 512, 512, ow, oh),
            })
        } else {
            None
        };

        let skin = if has_skin {
            Some(MaskData {
                mask_type: "skin".to_string(),
                width: ow,
                height: oh,
                data: bilinear_upsample_mask(&skin_mask, 512, 512, ow, oh),
            })
        } else {
            None
        };

        Ok((sky, skin))
    }

    // ─── Internal: Tiled Denoise ────────────────────────────────────

    fn run_tiled_denoise(
        &self,
        img: &DynamicImage,
        progress_cb: &dyn Fn(f64),
    ) -> Result<DynamicImage, AppError> {
        // Prefer NAFNet; fall back to Restormer if available
        let mut nafnet_guard = self.nafnet.lock();
        let mut restormer_guard = self.restormer.lock();

        let session = Self::ensure_loaded(&mut nafnet_guard)
            .or_else(|| Self::ensure_loaded(&mut restormer_guard));

        let session = match session {
            Some(s) => s,
            None => {
                // No denoise model available — return image unchanged
                progress_cb(1.0);
                return Ok(img.clone());
            }
        };

        let tile_cfg = TileConfig {
            size: session.manifest.tile_size.unwrap_or(512),
            overlap: session.manifest.tile_overlap.unwrap_or(32),
        };

        let (w, h) = img.dimensions();
        let tiles = compute_tiles(w, h, &tile_cfg);
        let total_tiles = tiles.len() as f64;
        let rgba = img.to_rgba8();
        let mut output_buf = vec![0u8; (w * h * 4) as usize];
        let mut weight_buf = vec![0.0f32; (w * h) as usize];

        // Copy original into output as base
        for (i, p) in rgba.pixels().enumerate() {
            let base = i * 4;
            output_buf[base] = p[0];
            output_buf[base + 1] = p[1];
            output_buf[base + 2] = p[2];
            output_buf[base + 3] = p[3];
        }

        for (tile_idx, (tx, ty, tw, th)) in tiles.iter().enumerate() {
            let tile_rgba = extract_tile(&rgba, *tx, *ty, *tw, *th);
            let nchw = tile_to_nchw(&tile_rgba, *tw, *th, session.manifest.normalize.as_ref());

            let input_tensor = ort::value::Tensor::from_array(nchw)
                .map_err(|e| AppError::OptimizeError(format!("Denoise tensor: {}", e)))?;

            let outputs = session.session.run(ort::inputs![input_tensor])
                .map_err(|e| AppError::OptimizeError(format!("Denoise inference: {}", e)))?;

            let (_shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| AppError::OptimizeError(format!("Extract denoise output: {}", e)))?;
            let out_data = data.to_vec();

            blend_tile_into(
                &mut output_buf,
                &mut weight_buf,
                &out_data,
                *tx,
                *ty,
                *tw,
                *th,
                w,
                h,
                &tile_cfg,
                session.manifest.normalize.as_ref(),
            );

            progress_cb((tile_idx + 1) as f64 / total_tiles);
        }

        // Normalize by accumulated weights
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let base = idx * 4;
                let wt = weight_buf[idx];
                if wt > 0.0 {
                    output_buf[base] = (output_buf[base] as f32 / wt).clamp(0.0, 255.0) as u8;
                    output_buf[base + 1] = (output_buf[base + 1] as f32 / wt).clamp(0.0, 255.0) as u8;
                    output_buf[base + 2] = (output_buf[base + 2] as f32 / wt).clamp(0.0, 255.0) as u8;
                }
            }
        }

        let result_buf = image::RgbaImage::from_raw(w, h, output_buf)
            .ok_or_else(|| AppError::OptimizeError("Failed to create denoised image".to_string()))?;

        Ok(DynamicImage::ImageRgba8(result_buf))
    }

    // ─── Internal: HDRNet Enhance ───────────────────────────────────

    fn run_hdrnet(
        &self,
        img: &DynamicImage,
        progress_cb: &dyn Fn(f64),
    ) -> Result<DynamicImage, AppError> {
        let mut hdrnet_guard = self.hdrnet.lock();
        let session = match Self::ensure_loaded(&mut hdrnet_guard) {
            Some(s) => s,
            None => {
                progress_cb(1.0);
                return Ok(img.clone());
            }
        };

        let (w, h) = img.dimensions();

        // Downsample to 256x256 for bilateral grid inference
        let proxy = img.resize_exact(256, 256, FilterType::Triangle);
        let rgb_proxy = proxy.to_rgb8();

        let mut input = ndarray::Array4::<f32>::zeros((1, 3, 256, 256));
        for y in 0..256usize {
            for x in 0..256usize {
                let p = rgb_proxy.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    input[[0, c, y, x]] = p[c] as f32 / 255.0;
                }
            }
        }

        progress_cb(0.2);

        let input_tensor = ort::value::Tensor::from_array(input)
            .map_err(|e| AppError::OptimizeError(format!("HDRNet tensor: {}", e)))?;
        let outputs = session.session.run(ort::inputs![input_tensor])
            .map_err(|e| AppError::OptimizeError(format!("HDRNet inference: {}", e)))?;

        // Output: bilateral grid [1, 12, 16, 16, 8]
        let (_shape, grid_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AppError::OptimizeError(format!("Extract HDRNet grid: {}", e)))?;
        let grid = grid_data.to_vec();

        progress_cb(0.5);

        // Apply bilateral grid via trilinear slicing
        let rgba = img.to_rgba8();
        let mut result = rgba.clone();

        let grid_d = 16usize; // spatial dimensions
        let grid_l = 8usize;  // luma dimension
        let num_coeffs = 12usize; // 3 output channels * (3 input + 1 bias)

        for (idx, pixel) in result.pixels_mut().enumerate() {
            let px = (idx as u32) % w;
            let py = (idx as u32) / w;

            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            // Grid coordinates
            let gx = (px as f32 / w as f32) * (grid_d as f32 - 1.0);
            let gy = (py as f32 / h as f32) * (grid_d as f32 - 1.0);
            let gl = lum * (grid_l as f32 - 1.0);

            // Trilinear interpolation of 12 affine coefficients
            let coeffs = trilinear_sample(&grid, num_coeffs, grid_d, grid_d, grid_l, gx, gy, gl);

            // Apply 3x4 affine: out_c = coeffs[c*4..c*4+3] . [r,g,b] + coeffs[c*4+3]
            let out_r = (coeffs[0] * r + coeffs[1] * g + coeffs[2] * b + coeffs[3]).clamp(0.0, 1.0);
            let out_g = (coeffs[4] * r + coeffs[5] * g + coeffs[6] * b + coeffs[7]).clamp(0.0, 1.0);
            let out_b = (coeffs[8] * r + coeffs[9] * g + coeffs[10] * b + coeffs[11]).clamp(0.0, 1.0);

            pixel[0] = (out_r * 255.0) as u8;
            pixel[1] = (out_g * 255.0) as u8;
            pixel[2] = (out_b * 255.0) as u8;
        }

        progress_cb(1.0);
        Ok(DynamicImage::ImageRgba8(result))
    }

    // ─── Internal: Guardrails ───────────────────────────────────────

    fn apply_guardrails(
        &self,
        optimized: &mut DynamicImage,
        original: &DynamicImage,
        masks: &MaskSet,
    ) {
        let orig_rgba = original.to_rgba8();
        let mut opt_rgba = optimized.to_rgba8();
        let (w, h) = opt_rgba.dimensions();

        for y in 0..h {
            for x in 0..w {
                let orig_p = orig_rgba.get_pixel(x, y);
                let opt_p = opt_rgba.get_pixel_mut(x, y);

                let or = orig_p[0] as f32 / 255.0;
                let og = orig_p[1] as f32 / 255.0;
                let ob = orig_p[2] as f32 / 255.0;
                let mut nr = opt_p[0] as f32 / 255.0;
                let mut ng = opt_p[1] as f32 / 255.0;
                let mut nb = opt_p[2] as f32 / 255.0;

                // Highlight preservation: cap max delta at 0.15
                let max_delta = 0.15f32;
                nr = nr.clamp(or - max_delta, or + max_delta);
                ng = ng.clamp(og - max_delta, og + max_delta);
                nb = nb.clamp(ob - max_delta, ob + max_delta);

                // Saturation limiter: cap at +30% saturation increase
                let orig_sat = channel_saturation(or, og, ob);
                let new_sat = channel_saturation(nr, ng, nb);
                if new_sat > orig_sat * 1.3 && new_sat > 0.01 {
                    let scale = (orig_sat * 1.3) / new_sat;
                    let lum = 0.2126 * nr + 0.7152 * ng + 0.0722 * nb;
                    nr = lum + (nr - lum) * scale;
                    ng = lum + (ng - lum) * scale;
                    nb = lum + (nb - lum) * scale;
                }

                // Skin hue stability within BiSeNet skin mask
                if let Some(ref skin_mask) = masks.skin {
                    let pixel_idx = (y * w + x) as usize;
                    if pixel_idx < skin_mask.data.len() && skin_mask.data[pixel_idx] > 0.5 {
                        let (oh, os, _) = rgb_to_hsl(or as f64, og as f64, ob as f64);
                        let (nh, ns, nl) = rgb_to_hsl(nr as f64, ng as f64, nb as f64);

                        // Constrain hue to ±5° and saturation to ±10%
                        let clamped_h = clamp_circular(nh, oh, 5.0);
                        let clamped_s = ns.clamp(os - 0.10, os + 0.10).clamp(0.0, 1.0);

                        let (sr, sg, sb) = hsl_to_rgb(clamped_h, clamped_s, nl);
                        nr = sr as f32;
                        ng = sg as f32;
                        nb = sb as f32;
                    }
                }

                opt_p[0] = (nr * 255.0).clamp(0.0, 255.0) as u8;
                opt_p[1] = (ng * 255.0).clamp(0.0, 255.0) as u8;
                opt_p[2] = (nb * 255.0).clamp(0.0, 255.0) as u8;
            }
        }

        *optimized = DynamicImage::ImageRgba8(opt_rgba);
    }

    // ─── Internal: Parameter Decomposition ──────────────────────────

    fn decompose_to_params(
        &self,
        original: &DynamicImage,
        optimized: &DynamicImage,
    ) -> HashMap<String, f64> {
        let mut params = HashMap::new();

        let orig_stats = compute_simple_stats(original);
        let opt_stats = compute_simple_stats(optimized);

        // Approximate exposure delta (EV)
        if orig_stats.mean_lum > 0.001 {
            let ev = (opt_stats.mean_lum / orig_stats.mean_lum).ln() / 2.0_f64.ln();
            if ev.abs() > 0.05 {
                params.insert("exposure".to_string(), round2(ev.clamp(-3.0, 3.0)));
            }
        }

        // Approximate contrast delta
        if orig_stats.stddev_lum > 0.001 {
            let contrast_ratio = opt_stats.stddev_lum / orig_stats.stddev_lum;
            let contrast = (contrast_ratio - 1.0) * 100.0;
            if contrast.abs() > 2.0 {
                params.insert("contrast".to_string(), round2(contrast.clamp(-50.0, 50.0)));
            }
        }

        // Approximate vibrance delta
        let sat_delta = opt_stats.mean_sat - orig_stats.mean_sat;
        let vibrance = sat_delta * 200.0;
        if vibrance.abs() > 2.0 {
            params.insert("vibrance".to_string(), round2(vibrance.clamp(-50.0, 50.0)));
        }

        // Approximate shadows delta
        let shadow_delta = opt_stats.shadow_mean - orig_stats.shadow_mean;
        let shadows = shadow_delta * 400.0;
        if shadows.abs() > 3.0 {
            params.insert("shadows".to_string(), round2(shadows.clamp(-100.0, 100.0)));
        }

        // Approximate highlights delta
        let highlight_delta = opt_stats.highlight_mean - orig_stats.highlight_mean;
        let highlights = highlight_delta * 400.0;
        if highlights.abs() > 3.0 {
            params.insert("highlights".to_string(), round2(highlights.clamp(-100.0, 100.0)));
        }

        params
    }
}

// ─── Tiling functions ──────────────────────────────────────────────

/// Compute tile positions with overlap
fn compute_tiles(w: u32, h: u32, cfg: &TileConfig) -> Vec<(u32, u32, u32, u32)> {
    let mut tiles = Vec::new();
    let step = cfg.size - cfg.overlap;

    let mut y = 0u32;
    while y < h {
        let th = cfg.size.min(h - y);
        let mut x = 0u32;
        while x < w {
            let tw = cfg.size.min(w - x);
            tiles.push((x, y, tw, th));
            if x + tw >= w { break; }
            x += step;
        }
        if y + th >= h { break; }
        y += step;
    }

    tiles
}

/// Extract a tile from an RGBA image
fn extract_tile(img: &image::RgbaImage, x: u32, y: u32, w: u32, h: u32) -> Vec<u8> {
    let mut tile = Vec::with_capacity((w * h * 4) as usize);
    let img_w = img.width();
    for ty in 0..h {
        for tx in 0..w {
            let px = (x + tx).min(img_w - 1);
            let py = (y + ty).min(img.height() - 1);
            let p = img.get_pixel(px, py);
            tile.extend_from_slice(&p.0);
        }
    }
    tile
}

/// Convert tile to NCHW float tensor
fn tile_to_nchw(
    tile: &[u8],
    w: u32,
    h: u32,
    normalize: Option<&OptimizeNormalizeConfig>,
) -> ndarray::Array4<f32> {
    let (ww, hh) = (w as usize, h as usize);
    let mut tensor = ndarray::Array4::<f32>::zeros((1, 3, hh, ww));

    for y in 0..hh {
        for x in 0..ww {
            let base = (y * ww + x) * 4;
            for c in 0..3 {
                let mut val = tile[base + c] as f32 / 255.0;
                if let Some(norm) = normalize {
                    val = (val - norm.mean[c]) / norm.std[c];
                }
                tensor[[0, c, y, x]] = val;
            }
        }
    }

    tensor
}

/// Blend denoised tile back into the output with linear feathering in overlap
fn blend_tile_into(
    output: &mut [u8],
    weights: &mut [f32],
    tile_data: &[f32],
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    img_w: u32,
    img_h: u32,
    cfg: &TileConfig,
    normalize: Option<&OptimizeNormalizeConfig>,
) {
    let overlap = cfg.overlap as f32;

    for y in 0..th {
        for x in 0..tw {
            let px = tx + x;
            let py = ty + y;
            if px >= img_w || py >= img_h {
                continue;
            }

            // Linear feathering weight based on distance from tile edge
            let wx = if overlap > 0.0 {
                let left_dist = x as f32;
                let right_dist = (tw - 1 - x) as f32;
                (left_dist / overlap).min(1.0).min((right_dist / overlap).min(1.0))
            } else {
                1.0
            };
            let wy = if overlap > 0.0 {
                let top_dist = y as f32;
                let bot_dist = (th - 1 - y) as f32;
                (top_dist / overlap).min(1.0).min((bot_dist / overlap).min(1.0))
            } else {
                1.0
            };
            let weight = wx * wy;

            let img_idx = (py * img_w + px) as usize;
            let base = img_idx * 4;

            // Tile output is NCHW: [1, 3, H, W]
            for c in 0..3 {
                let tile_idx = c * (th as usize * tw as usize) + y as usize * tw as usize + x as usize;
                let mut val = if tile_idx < tile_data.len() {
                    tile_data[tile_idx]
                } else {
                    0.0
                };

                // Denormalize
                if let Some(norm) = normalize {
                    val = val * norm.std[c] + norm.mean[c];
                }

                let pixel_val = (val * 255.0).clamp(0.0, 255.0);
                output[base + c] = ((output[base + c] as f32) + pixel_val * weight) as u8;
            }

            weights[img_idx] += weight;
        }
    }
}

// ─── HDRNet bilateral grid sampling ────────────────────────────────

/// Trilinear interpolation in the bilateral grid
fn trilinear_sample(
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
        // Grid layout: [1, num_coeffs, gh, gw, gd]
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

// ─── Mask upsampling ───────────────────────────────────────────────

/// Bilinear upsample a mask from (sw, sh) to (dw, dh)
fn bilinear_upsample_mask(
    data: &[f32],
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
) -> Vec<f32> {
    let mut result = vec![0.0f32; (dw * dh) as usize];
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;

    for y in 0..dh {
        for x in 0..dw {
            let src_x = x as f32 * sx;
            let src_y = y as f32 * sy;

            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let get = |xx: u32, yy: u32| -> f32 {
                let idx = (yy * sw + xx) as usize;
                data.get(idx).copied().unwrap_or(0.0)
            };

            let v = get(x0, y0) * (1.0 - fx) * (1.0 - fy)
                + get(x1, y0) * fx * (1.0 - fy)
                + get(x0, y1) * (1.0 - fx) * fy
                + get(x1, y1) * fx * fy;

            result[(y * dw + x) as usize] = v;
        }
    }

    result
}

// ─── Image blending ────────────────────────────────────────────────

fn blend_images(original: &DynamicImage, optimized: &DynamicImage, strength: f64) -> DynamicImage {
    let orig = original.to_rgba8();
    let opt = optimized.to_rgba8();
    let (w, h) = orig.dimensions();
    let mut result = orig.clone();

    for y in 0..h {
        for x in 0..w {
            let op = orig.get_pixel(x, y);
            let np = opt.get_pixel(x, y);
            let rp = result.get_pixel_mut(x, y);

            for c in 0..3 {
                let blended = op[c] as f64 * (1.0 - strength) + np[c] as f64 * strength;
                rp[c] = blended.clamp(0.0, 255.0) as u8;
            }
        }
    }

    DynamicImage::ImageRgba8(result)
}

// ─── Helper functions ──────────────────────────────────────────────

fn channel_saturation(r: f32, g: f32, b: f32) -> f32 {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    if max > 0.0 { (max - min) / max } else { 0.0 }
}

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

/// Clamp a hue value to within ±max_delta of a center, respecting circularity
fn clamp_circular(val: f64, center: f64, max_delta: f64) -> f64 {
    let mut diff = val - center;
    if diff > 180.0 { diff -= 360.0; }
    if diff < -180.0 { diff += 360.0; }
    let clamped_diff = diff.clamp(-max_delta, max_delta);
    (center + clamped_diff).rem_euclid(360.0)
}

struct SimpleStats {
    mean_lum: f64,
    stddev_lum: f64,
    mean_sat: f64,
    shadow_mean: f64,
    highlight_mean: f64,
}

fn compute_simple_stats(img: &DynamicImage) -> SimpleStats {
    let rgb = img.to_rgb8();
    let total = rgb.pixels().len() as f64;
    if total == 0.0 {
        return SimpleStats {
            mean_lum: 0.0, stddev_lum: 0.0, mean_sat: 0.0,
            shadow_mean: 0.0, highlight_mean: 0.0,
        };
    }

    let mut lum_sum = 0.0f64;
    let mut lum_sq_sum = 0.0f64;
    let mut sat_sum = 0.0f64;
    let mut shadow_sum = 0.0f64;
    let mut shadow_count = 0u64;
    let mut highlight_sum = 0.0f64;
    let mut highlight_count = 0u64;

    for p in rgb.pixels() {
        let r = p[0] as f64 / 255.0;
        let g = p[1] as f64 / 255.0;
        let b = p[2] as f64 / 255.0;
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        lum_sum += lum;
        lum_sq_sum += lum * lum;

        let max_c = r.max(g).max(b);
        let min_c = r.min(g).min(b);
        if max_c > 0.0 {
            sat_sum += (max_c - min_c) / max_c;
        }

        if lum < 0.25 {
            shadow_sum += lum;
            shadow_count += 1;
        }
        if lum > 0.75 {
            highlight_sum += lum;
            highlight_count += 1;
        }
    }

    let mean_lum = lum_sum / total;
    let variance = (lum_sq_sum / total) - (mean_lum * mean_lum);

    SimpleStats {
        mean_lum,
        stddev_lum: variance.max(0.0).sqrt(),
        mean_sat: sat_sum / total,
        shadow_mean: if shadow_count > 0 { shadow_sum / shadow_count as f64 } else { 0.0 },
        highlight_mean: if highlight_count > 0 { highlight_sum / highlight_count as f64 } else { 1.0 },
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
