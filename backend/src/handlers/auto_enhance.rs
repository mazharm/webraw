use axum::{extract::State, http::HeaderMap, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::models::auto_enhance::{EnhanceModelKind, EnhanceResult, ModelDescriptor};
use crate::models::error::AppError;
use crate::models::jobs::JobKind;
use crate::models::optimize::OptimizeConfig;
use crate::services::auto_enhance::{compute_image_stats, RunContext};
use crate::AppState;

// ---------------------------------------------------------------------------
// GET /api/v1/auto-enhance/models
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListModelsResponse {
    pub models: Vec<ModelDescriptor>,
}

pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<ListModelsResponse> {
    let mut models = state.auto_enhance.list_models();

    // Add optimize as a virtual model if at least one ONNX model is available
    let optimize_models = state.optimize.available_models();
    if optimize_models.iter().any(|m| m.available) {
        models.push(ModelDescriptor {
            id: "optimize".to_string(),
            name: "AI Optimize (Local)".to_string(),
            description: "Full pipeline: AI denoise, HDRNet enhance, smart masks \u{2014} runs locally, no API key".to_string(),
            kind: EnhanceModelKind::ParameterPredictor,
            version: "1.0".to_string(),
            builtin: true,
            requires_api_key: false,
            publisher: Some("Local ONNX".to_string()),
        });
    }

    Json(ListModelsResponse { models })
}

// ---------------------------------------------------------------------------
// POST /api/v1/auto-enhance/run
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRequest {
    pub file_id: String,
    pub model_id: String,
    pub strength: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunResponse {
    pub job_id: String,
}

pub async fn run_auto_enhance(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<RunRequest>,
) -> Result<Json<RunResponse>, AppError> {
    // Route "optimize" model to optimize service
    if req.model_id == "optimize" {
        return run_optimize_as_enhance(state, req).await;
    }

    let model = state
        .auto_enhance
        .get_model(&req.model_id)
        .ok_or_else(|| AppError::ModelNotFound(req.model_id.clone()))?;

    // Extract optional API keys from headers
    let api_key = headers
        .get("X-Gemini-Key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let anthropic_key = headers
        .get("X-Anthropic-Key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let openai_key = headers
        .get("X-OpenAI-Key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Validate: if the model requires an API key, ensure one was provided
    if model.requires_api_key() {
        let has_key = [&api_key, &anthropic_key, &openai_key]
            .iter()
            .any(|k| matches!(k, Some(v) if !v.trim().is_empty()));
        if !has_key {
            return Err(AppError::AiInvalidKey);
        }
    }

    let (job_id, job) = state.jobs.create_job(JobKind::AutoEnhance, None);

    let state_clone = state.clone();
    let file_id = req.file_id.clone();
    let jid = job_id.clone();

    tokio::spawn(async move {
        job.set_processing();

        // Get proxy image (1280px max edge)
        let img_result = state_clone
            .render
            .get_proxy_image(&state_clone.cache, &file_id, 1280)
            .await;

        match img_result {
            Ok(img) => {
                let stats = compute_image_stats(&img);
                let ctx = RunContext {
                    api_key,
                    anthropic_key,
                    openai_key,
                    config: state_clone.config.clone(),
                };
                match model.run(&img, &stats, &ctx).await {
                    Ok(result) => {
                        let result_json = serde_json::to_value(&result).unwrap_or_default();
                        job.set_complete(result_json);
                    }
                    Err(e) => {
                        job.set_failed(e.to_problem_detail(&jid));
                    }
                }
            }
            Err(e) => {
                job.set_failed(e.to_problem_detail(&jid));
            }
        }
    });

    Ok(Json(RunResponse { job_id }))
}

/// Route "optimize" model requests to the optimize service,
/// returning results in the same EnhanceResult::Parameters format.
async fn run_optimize_as_enhance(
    state: Arc<AppState>,
    req: RunRequest,
) -> Result<Json<RunResponse>, AppError> {
    let strength_pct = req.strength.unwrap_or(50.0);
    let config = OptimizeConfig {
        strength: strength_pct / 100.0,
        denoise: true,
        enhance: true,
        masks: true,
    };

    let (job_id, job) = state.jobs.create_job(JobKind::Optimize, None);
    let file_id = req.file_id.clone();
    let jid = job_id.clone();

    tokio::spawn(async move {
        job.set_processing();

        let img_result = state
            .render
            .get_proxy_image(&state.cache, &file_id, 2560)
            .await;

        match img_result {
            Ok(img) => {
                let job_ref = job.clone();
                let state_ref = state.clone();
                let result = tokio::task::spawn_blocking(move || {
                    state_ref.optimize.run_optimize(
                        &img,
                        &config,
                        &|progress| {
                            job_ref.set_progress(progress);
                        },
                    )
                })
                .await;

                match result {
                    Ok(Ok(opt_result)) => {
                        let enhance_result = EnhanceResult::Parameters {
                            values: opt_result.applied_params,
                        };
                        let json = serde_json::to_value(&enhance_result).unwrap_or_default();
                        job.set_complete(json);
                    }
                    Ok(Err(e)) => {
                        job.set_failed(e.to_problem_detail(&jid));
                    }
                    Err(e) => {
                        job.set_failed(
                            AppError::OptimizeError(format!("Task panicked: {}", e))
                                .to_problem_detail(&jid),
                        );
                    }
                }
            }
            Err(e) => {
                job.set_failed(e.to_problem_detail(&jid));
            }
        }
    });

    Ok(Json(RunResponse { job_id }))
}
