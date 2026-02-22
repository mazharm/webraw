use axum::{
    extract::{Multipart, Path, Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::models::error::AppError;
use crate::AppState;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UploadQuery {
    pub ttl_hint: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UploadResponse {
    pub file_id: String,
    pub exif: Option<serde_json::Value>,
    pub size_bytes: u64,
    pub expires_at: String,
}

pub async fn upload_file(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Query(query): Query<UploadQuery>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, AppError> {
    let session = headers.get("X-Session-Token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("anonymous");
    if let Err(retry_after) = state.upload_limiter.check(session) {
        return Err(AppError::RateLimited { retry_after });
    }

    let extended_ttl = query.ttl_hint.as_deref() == Some("extended");

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::ValidationError(format!("Multipart error: {}", e)))?
    {
        let filename = field.file_name().unwrap_or("upload").to_string();
        let content_type = field
            .content_type()
            .filter(|ct| !ct.is_empty())
            .unwrap_or("application/octet-stream")
            .to_string();

        // Read the body first — sending an error before consuming the upload
        // causes browsers to see a connection reset ("Failed to fetch").
        let data = field
            .bytes()
            .await
            .map_err(|e| AppError::Internal(format!("Failed to read upload: {}", e)))?;

        // Validate MIME type (case-insensitive, accept any image/* since
        // browsers assign non-standard types to RAW files like "image/CR2")
        let ct_lower = content_type.to_ascii_lowercase();
        let allowed = ct_lower.starts_with("image/")
            || ct_lower == "application/octet-stream";
        if !allowed {
            return Err(AppError::InvalidMimeType(content_type));
        }

        let cached = state
            .cache
            .store_file(&filename, &data, &content_type, extended_ttl)
            .await?;

        return Ok(Json(UploadResponse {
            file_id: cached.file_id,
            exif: cached.exif,
            size_bytes: cached.size_bytes,
            expires_at: cached.expires_at.to_rfc3339(),
        }));
    }

    Err(AppError::ValidationError("No file in upload".to_string()))
}

pub async fn get_file(
    State(state): State<Arc<AppState>>,
    Path(file_id): Path<String>,
) -> Result<Response, AppError> {
    let cached = state.cache.get_file(&file_id)?;
    let data = tokio::fs::read(&cached.path).await.map_err(|e| {
        AppError::Internal(format!("Failed to read cached file: {}", e))
    })?;

    let disposition = format!(
        "attachment; filename=\"{}\"",
        cached.original_filename.replace('"', "")
    );

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, cached.mime_type),
            (header::CONTENT_LENGTH, data.len().to_string()),
            (header::CONTENT_DISPOSITION, disposition),
        ],
        data,
    )
        .into_response())
}

pub async fn delete_file(
    State(state): State<Arc<AppState>>,
    Path(file_id): Path<String>,
) -> Result<StatusCode, AppError> {
    state.cache.delete_file(&file_id).await?;
    Ok(StatusCode::NO_CONTENT)
}
