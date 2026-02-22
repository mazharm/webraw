use axum::{
    extract::{Query, State},
    response::sse::{Event, Sse},
    Json,
};
use base64::Engine as _;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;

use crate::models::edit_state::EditState;
use crate::models::error::AppError;
use crate::models::jobs::JobKind;
use crate::AppState;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThumbnailRequest {
    pub file_id: String,
    pub max_edge: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewRequest {
    pub file_id: String,
    pub edit_state: EditState,
    pub max_edge: Option<u32>,
    pub color_space: Option<String>,
    pub quality_hint: Option<String>,
}

#[allow(dead_code)]
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreviewResponse {
    pub image_base64: String,
    pub mime_type: String,
    pub bit_depth: u8,
    pub color_space: String,
    pub width: u32,
    pub height: u32,
    pub histogram: Option<crate::services::render::HistogramData>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobCreatedResponse {
    pub job_id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportRequest {
    pub file_id: String,
    pub edit_state: EditState,
    pub format: String,
    pub bit_depth: Option<u8>,
    pub quality: Option<u8>,
    pub color_space: Option<String>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BaseRenderStreamQuery {
    pub file_id: String,
    pub base_render_hash: Option<String>,
    pub max_edge: Option<u32>,
}

pub async fn create_thumbnail(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ThumbnailRequest>,
) -> Result<axum::response::Response, AppError> {
    let max_edge = req.max_edge.unwrap_or(300);
    let data = state
        .render
        .generate_thumbnail(&state.cache, &req.file_id, max_edge)
        .await?;

    Ok((
        axum::http::StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "image/jpeg")],
        data,
    )
        .into_response())
}

use axum::response::IntoResponse;

pub async fn create_preview(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<PreviewRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let session = headers.get("X-Session-Token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("anonymous");
    if let Err(retry_after) = state.render_limiter.check(session) {
        return Err(AppError::RateLimited { retry_after });
    }

    let max_edge = req.max_edge.unwrap_or(2560);
    let color_space = req.color_space.as_deref().unwrap_or("sRGB");
    let quality_hint = req.quality_hint.as_deref().unwrap_or("FAST");

    if quality_hint == "BEST" {
        let idempotency_key = None;
        let (job_id, job) = state.jobs.create_job(JobKind::Preview, idempotency_key);

        let state_clone = state.clone();
        let edit_state = req.edit_state.clone();
        let file_id = req.file_id.clone();
        let cs = color_space.to_string();
        let jid = job_id.clone();

        tokio::spawn(async move {
            job.set_processing();
            match state_clone.render.render_preview(
                &state_clone.cache, &file_id, &edit_state, max_edge, &cs, "BEST",
            ).await {
                Ok(result) => {
                    job.set_complete(serde_json::to_value(result).unwrap());
                }
                Err(e) => {
                    let pd = e.to_problem_detail(&jid);
                    job.set_failed(pd);
                }
            }
        });

        return Ok(Json(serde_json::json!({ "jobId": job_id })));
    }

    // FAST mode - synchronous
    let result = state.render.render_preview(
        &state.cache, &req.file_id, &req.edit_state, max_edge, color_space, quality_hint,
    ).await?;

    Ok(Json(serde_json::to_value(result).unwrap()))
}

pub async fn create_export(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExportRequest>,
) -> Result<Json<JobCreatedResponse>, AppError> {
    let (job_id, job) = state.jobs.create_job(JobKind::Export, None);

    let state_clone = state.clone();
    let file_id = req.file_id.clone();
    let edit_state = req.edit_state.clone();
    let format = req.format.clone();
    let bit_depth = req.bit_depth.unwrap_or(8);
    let quality = req.quality.unwrap_or(95);
    let color_space = req.color_space.clone().unwrap_or_else(|| "sRGB".to_string());
    let jid = job_id.clone();

    tokio::spawn(async move {
        job.set_processing();
        match state_clone.render.render_export(
            &state_clone.cache, &file_id, &edit_state, &format, bit_depth, quality, &color_space,
        ).await {
            Ok(result) => {
                job.set_complete(serde_json::to_value(result).unwrap());
            }
            Err(e) => {
                let pd = e.to_problem_detail(&jid);
                job.set_failed(pd);
            }
        }
    });

    Ok(Json(JobCreatedResponse { job_id }))
}

pub async fn base_render_stream(
    State(state): State<Arc<AppState>>,
    Query(query): Query<BaseRenderStreamQuery>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, AppError> {
    let file_id = query.file_id.clone();
    let max_edge_full = query.max_edge.unwrap_or(2560);
    let max_edge_fast = (max_edge_full / 2).max(640);

    let state_clone = state.clone();

    let stream = async_stream::stream! {
        // Fast render (low-res)
        match state_clone.render.render_base(&state_clone.cache, &file_id, max_edge_fast).await {
            Ok((png_data, width, height)) => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&png_data);
                let payload = serde_json::json!({
                    "imageBase64": b64,
                    "mimeType": "image/png",
                    "bitDepth": 16,
                    "colorSpace": "linear-rec2020",
                    "width": width,
                    "height": height,
                });
                yield Ok(Event::default().event("fast").json_data(payload).unwrap());
            }
            Err(e) => {
                let pd = e.to_problem_detail("sse");
                yield Ok(Event::default().event("error").json_data(pd).unwrap());
                return;
            }
        }

        // Full render
        match state_clone.render.render_base(&state_clone.cache, &file_id, max_edge_full).await {
            Ok((png_data, width, height)) => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(&png_data);
                let payload = serde_json::json!({
                    "imageBase64": b64,
                    "mimeType": "image/png",
                    "bitDepth": 16,
                    "colorSpace": "linear-rec2020",
                    "width": width,
                    "height": height,
                    "histogram": null,
                });
                yield Ok(Event::default().event("done").json_data(payload).unwrap());
            }
            Err(e) => {
                let pd = e.to_problem_detail("sse");
                yield Ok(Event::default().event("error").json_data(pd).unwrap());
            }
        }
    };

    Ok(Sse::new(stream))
}
