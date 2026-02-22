use axum::{
    extract::{Path, State},
    Json,
};
use std::sync::Arc;

use crate::models::error::AppError;
use crate::models::jobs::JobInfo;
use crate::AppState;

pub async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> Result<Json<JobInfo>, AppError> {
    match state.jobs.get_job(&job_id) {
        Some(job) => Ok(Json(job.get_info())),
        None => Err(AppError::FileNotFound(format!("Job not found: {}", job_id))),
    }
}
