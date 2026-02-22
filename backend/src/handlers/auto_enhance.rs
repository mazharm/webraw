use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::models::auto_enhance::ModelDescriptor;
use crate::models::error::AppError;
use crate::models::jobs::JobKind;
use crate::services::auto_enhance::compute_image_stats;
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
    Json(ListModelsResponse {
        models: state.auto_enhance.list_models(),
    })
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
    Json(req): Json<RunRequest>,
) -> Result<Json<RunResponse>, AppError> {
    let model = state
        .auto_enhance
        .get_model(&req.model_id)
        .ok_or_else(|| AppError::ModelNotFound(req.model_id.clone()))?;

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
                match model.run(&img, &stats) {
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
