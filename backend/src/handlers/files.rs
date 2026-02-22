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
            .unwrap_or("application/octet-stream")
            .to_string();

        // Validate MIME type
        let allowed_types = [
            "image/jpeg", "image/png", "image/tiff",
            "image/x-adobe-dng", "image/x-canon-cr2", "image/x-canon-cr3",
            "image/x-nikon-nef", "image/x-sony-arw", "image/x-fuji-raf",
            "application/octet-stream", // fallback for RAW files
        ];
        if !allowed_types.iter().any(|t| content_type.starts_with(t)) && content_type != "application/octet-stream" {
            return Err(AppError::InvalidMimeType(content_type));
        }

        let data = field
            .bytes()
            .await
            .map_err(|e| AppError::Internal(format!("Failed to read upload: {}", e)))?;

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
