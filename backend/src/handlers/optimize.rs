use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

use crate::models::error::AppError;
use crate::models::jobs::JobKind;
use crate::models::optimize::*;
use crate::AppState;

// ---------------------------------------------------------------------------
// GET /api/v1/optimize/models
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListOptimizeModelsResponse {
    pub models: Vec<OptimizeModelStatus>,
}

pub async fn list_optimize_models(
    State(state): State<Arc<AppState>>,
) -> Json<ListOptimizeModelsResponse> {
    Json(ListOptimizeModelsResponse {
        models: state.optimize.available_models(),
    })
}

// ---------------------------------------------------------------------------
// POST /api/v1/optimize/run
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizeRunResponse {
    pub job_id: String,
}

pub async fn run_optimize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OptimizeRequest>,
) -> Result<Json<OptimizeRunResponse>, AppError> {
    let config = OptimizeConfig::from(&req);

    if config.strength < 0.0 || config.strength > 1.0 {
        return Err(AppError::ValidationError(
            "Strength must be between 0 and 100".to_string(),
        ));
    }

    let (job_id, job) = state.jobs.create_job(JobKind::Optimize, None);

    let state_clone = state.clone();
    let file_id = req.file_id.clone();
    let jid = job_id.clone();

    tokio::spawn(async move {
        job.set_processing();

        // Get proxy image (2560px max edge for optimize pipeline)
        let img_result = state_clone
            .render
            .get_proxy_image(&state_clone.cache, &file_id, 2560)
            .await;

        match img_result {
            Ok(img) => {
                let job_ref = job.clone();
                let blocking_state = state_clone.clone();
                let result = tokio::task::spawn_blocking(move || {
                    blocking_state.optimize.run_optimize(
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
                        // Store the result image in cache
                        match state_clone
                            .cache
                            .store_artifact(
                                &opt_result.image_data,
                                &opt_result.mime_type,
                                &file_id,
                            )
                            .await
                        {
                            Ok(cached) => {
                                let result_data = OptimizeResultData {
                                    result_file_id: cached.file_id,
                                    applied_params: opt_result.applied_params,
                                    masks: OptimizeMasksSummary::from(&opt_result.masks),
                                };
                                let json = serde_json::to_value(&result_data).unwrap_or_default();
                                job.set_complete(json);
                            }
                            Err(e) => {
                                job.set_failed(e.to_problem_detail(&jid));
                            }
                        }
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

    Ok(Json(OptimizeRunResponse { job_id }))
}

// ---------------------------------------------------------------------------
// POST /api/v1/optimize/masks
// ---------------------------------------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizeMasksResponse {
    pub job_id: String,
}

pub async fn compute_masks(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OptimizeMasksRequest>,
) -> Result<Json<OptimizeMasksResponse>, AppError> {
    let (job_id, job) = state.jobs.create_job(JobKind::Optimize, None);

    let state_clone = state.clone();
    let file_id = req.file_id.clone();
    let jid = job_id.clone();

    tokio::spawn(async move {
        job.set_processing();

        let img_result = state_clone
            .render
            .get_proxy_image(&state_clone.cache, &file_id, 2560)
            .await;

        match img_result {
            Ok(img) => {
                let result = tokio::task::spawn_blocking(move || {
                    state_clone.optimize.compute_masks_only(&img)
                })
                .await;

                match result {
                    Ok(Ok(masks)) => {
                        let summary = OptimizeMasksSummary::from(&masks);
                        let json = serde_json::to_value(&summary).unwrap_or_default();
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

    Ok(Json(OptimizeMasksResponse { job_id }))
}

/// Request body for POST /api/v1/optimize/masks
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OptimizeMasksRequest {
    pub file_id: String,
}
