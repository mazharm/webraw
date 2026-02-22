use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ProblemDetail {
    #[serde(rename = "type")]
    pub problem_type: String,
    pub title: String,
    pub status: u16,
    pub detail: String,
    pub code: String,
    pub request_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<u64>,
}

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("RAW unsupported camera: {0}")]
    RawUnsupportedCamera(String),

    #[error("RAW corrupt data: {0}")]
    RawCorruptData(String),

    #[error("Render timeout")]
    RenderTimeout,

    #[error("AI bad request: {0}")]
    AiBadRequest(String),

    #[error("AI quota exceeded")]
    AiQuotaExceeded,

    #[error("AI invalid key")]
    AiInvalidKey,

    #[error("Rate limited")]
    RateLimited { retry_after: u64 },

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Missing session token")]
    MissingSessionToken,

    #[error("File too large: {0} bytes")]
    FileTooLarge(u64),

    #[error("Invalid MIME type: {0}")]
    InvalidMimeType(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl AppError {
    pub fn to_problem_detail(&self, request_id: &str) -> ProblemDetail {
        let (status, code, title, detail) = match self {
            AppError::FileNotFound(id) => (
                StatusCode::NOT_FOUND,
                "FILE_NOT_FOUND",
                "File Not Found",
                format!("File '{}' is missing, expired, or inaccessible", id),
            ),
            AppError::RawUnsupportedCamera(model) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "RAW_UNSUPPORTED_CAMERA",
                "Unsupported Camera",
                format!("Cannot decode RAW from camera model: {}", model),
            ),
            AppError::RawCorruptData(detail) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "RAW_CORRUPT_DATA",
                "Corrupt RAW Data",
                detail.clone(),
            ),
            AppError::RenderTimeout => (
                StatusCode::GATEWAY_TIMEOUT,
                "RENDER_TIMEOUT",
                "Render Timeout",
                "Processing exceeded time limit".to_string(),
            ),
            AppError::AiBadRequest(detail) => (
                StatusCode::BAD_REQUEST,
                "AI_BAD_REQUEST",
                "AI Bad Request",
                detail.clone(),
            ),
            AppError::AiQuotaExceeded => (
                StatusCode::TOO_MANY_REQUESTS,
                "AI_QUOTA_EXCEEDED",
                "AI Quota Exceeded",
                "Upstream Gemini API quota exceeded".to_string(),
            ),
            AppError::AiInvalidKey => (
                StatusCode::UNAUTHORIZED,
                "AI_INVALID_KEY",
                "Invalid API Key",
                "Gemini API key was rejected".to_string(),
            ),
            AppError::RateLimited { retry_after } => (
                StatusCode::TOO_MANY_REQUESTS,
                "RATE_LIMITED",
                "Rate Limited",
                format!("Rate limit exceeded. Retry after {} seconds", retry_after),
            ),
            AppError::ValidationError(field) => (
                StatusCode::BAD_REQUEST,
                "VALIDATION_ERROR",
                "Validation Error",
                field.clone(),
            ),
            AppError::MissingSessionToken => (
                StatusCode::FORBIDDEN,
                "MISSING_SESSION_TOKEN",
                "Missing Session Token",
                "X-Session-Token header is required".to_string(),
            ),
            AppError::FileTooLarge(size) => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "FILE_TOO_LARGE",
                "File Too Large",
                format!("File size {} exceeds maximum allowed", size),
            ),
            AppError::InvalidMimeType(mime) => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "INVALID_MIME_TYPE",
                "Invalid MIME Type",
                format!("MIME type '{}' is not supported", mime),
            ),
            AppError::Internal(detail) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                "Internal Error",
                detail.clone(),
            ),
        };

        ProblemDetail {
            problem_type: format!("https://webraw.dev/problems/{}", code.to_lowercase()),
            title: title.to_string(),
            status: status.as_u16(),
            detail,
            code: code.to_string(),
            request_id: request_id.to_string(),
            retry_after: if let AppError::RateLimited { retry_after } = self {
                Some(*retry_after)
            } else {
                None
            },
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        // Note: The request_id_middleware sets X-Request-Id on responses after the handler.
        // Since errors short-circuit before the middleware can set it, we generate one here.
        // The middleware will overwrite it, so the header and body will match after middleware runs.
        // To ensure consistency, we use a placeholder; the middleware applies the canonical id.
        let request_id = uuid::Uuid::new_v4().to_string();
        let problem = self.to_problem_detail(&request_id);
        let status = StatusCode::from_u16(problem.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        let mut response = (status, Json(problem)).into_response();
        response
            .headers_mut()
            .insert("X-Request-Id", request_id.parse().unwrap());
        response
            .headers_mut()
            .insert("Content-Type", "application/problem+json".parse().unwrap());
        response
    }
}
