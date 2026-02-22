use axum::{
    extract::State,
    http::HeaderMap,
    Json,
};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::sync::Arc;

use crate::models::error::AppError;
use crate::models::jobs::JobKind;
use crate::AppState;

#[allow(dead_code)]
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiEditRequest {
    pub file_id: String,
    pub edit_state: crate::models::edit_state::EditState,
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub mode: String,
    pub provider: Option<String>,
    pub options: Option<serde_json::Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiEditResponse {
    pub job_id: String,
}

pub async fn create_ai_edit(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<AiEditRequest>,
) -> Result<Json<AiEditResponse>, AppError> {
    let session = headers.get("X-Session-Token")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("anonymous");
    if let Err(retry_after) = state.ai_limiter.check(session) {
        return Err(AppError::RateLimited { retry_after });
    }

    let provider = req.provider.as_deref().unwrap_or("gemini");

    // Route API key header based on provider
    let api_key = match provider {
        "openai" => headers
            .get("X-OpenAI-Key")
            .and_then(|v| v.to_str().ok()),
        "google-imagen" => headers
            .get("X-Gemini-Key")
            .and_then(|v| v.to_str().ok()),
        _ => headers
            .get("X-Gemini-Key")
            .and_then(|v| v.to_str().ok()),
    };

    let api_key = api_key.ok_or(AppError::AiInvalidKey)?;

    if api_key.is_empty() {
        return Err(AppError::AiInvalidKey);
    }

    // Validate mode
    let valid_modes = ["edit", "remove", "replace_bg", "relight", "expand"];
    if !valid_modes.contains(&req.mode.as_str()) {
        return Err(AppError::ValidationError(format!("Invalid mode: {}", req.mode)));
    }

    let idempotency_key = headers
        .get("Idempotency-Key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let (job_id, job) = state.jobs.create_job(
        JobKind::AiEdit,
        idempotency_key.as_deref(),
    );

    let state_clone = state.clone();
    let api_key = api_key.to_string();
    let file_id = req.file_id.clone();
    let edit_state = req.edit_state.clone();
    let prompt = req.prompt.clone();
    let mode = req.mode.clone();
    let options = req.options.clone();
    let provider_str = req.provider.clone();
    let jid = job_id.clone();

    tokio::spawn(async move {
        job.set_processing();

        // Render current state to get the image for AI
        let render_result = state_clone.render.render_preview(
            &state_clone.cache,
            &file_id,
            &edit_state,
            2048, // 4MP equivalent
            "sRGB",
            "FAST",
        ).await;

        match render_result {
            Ok(preview) => {
                match state_clone.ai_proxy.execute_edit(
                    &api_key,
                    &preview.image_base64,
                    &prompt,
                    &mode,
                    options.as_ref(),
                    provider_str.as_deref(),
                ).await {
                    Ok(ai_result) => {
                        // Store the result image in cache
                        match state_clone.cache.store_artifact(
                            &ai_result.image_data,
                            &ai_result.mime_type,
                            &file_id,
                        ).await {
                            Ok(cached) => {
                                let mut prompt_hash = Sha256::new();
                                prompt_hash.update(prompt.as_bytes());
                                let prompt_hash = hex::encode(prompt_hash.finalize());

                                let actual_provider = provider_str.as_deref().unwrap_or("gemini");
                                let meta = serde_json::json!({
                                    "provider": actual_provider,
                                    "model": ai_result.model,
                                    "prompt": prompt,
                                    "promptHash": prompt_hash,
                                    "createdAt": chrono::Utc::now().to_rfc3339(),
                                });

                                job.set_complete(serde_json::json!({
                                    "resultFileId": cached.file_id,
                                    "meta": meta,
                                }));
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
            }
            Err(e) => {
                job.set_failed(e.to_problem_detail(&jid));
            }
        }
    });

    Ok(Json(AiEditResponse { job_id }))
}
