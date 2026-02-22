use std::sync::Arc;
use base64::Engine as _;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::models::config::AppConfig;
use crate::models::error::AppError;

#[allow(dead_code)]
pub struct AiProxyService {
    config: Arc<AppConfig>,
    client: Client,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    generation_config: GeminiGenerationConfig,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text { text: String },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiInlineData,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    response_modalities: Vec<String>,
    response_mime_type: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    error: Option<GeminiError>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiResponseContent,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GeminiResponsePart {
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiResponseInlineData,
    },
    Text { text: String },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseInlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct GeminiError {
    code: u16,
    message: String,
}

#[derive(Debug)]
pub struct AiEditResult {
    pub image_data: Vec<u8>,
    pub mime_type: String,
    pub model: String,
}

impl AiProxyService {
    pub fn new(config: Arc<AppConfig>) -> Self {
        Self {
            config,
            client: Client::new(),
        }
    }

    pub async fn execute_edit(
        &self,
        api_key: &str,
        image_base64: &str,
        prompt: &str,
        mode: &str,
        _options: Option<&serde_json::Value>,
        provider: Option<&str>,
    ) -> Result<AiEditResult, AppError> {
        match provider.unwrap_or("gemini") {
            "openai" => return self.execute_openai_edit(api_key, image_base64, prompt, mode).await,
            "google-imagen" => return self.execute_imagen_edit(api_key, image_base64, prompt, mode).await,
            _ => {} // fall through to Gemini
        }

        let model = &self.config.gemini_model;
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            model
        );

        let system_prompt = match mode {
            "edit" => format!("Edit this image: {}", prompt),
            "remove" => format!("Remove the following from this image: {}", prompt),
            "replace_bg" => format!("Replace the background of this image with: {}", prompt),
            "relight" => format!("Relight this image: {}", prompt),
            "expand" => format!("Expand this image outward: {}", prompt),
            _ => prompt.to_string(),
        };

        let request = GeminiRequest {
            contents: vec![GeminiContent {
                parts: vec![
                    GeminiPart::Text { text: system_prompt },
                    GeminiPart::InlineData {
                        inline_data: GeminiInlineData {
                            mime_type: "image/jpeg".to_string(),
                            data: image_base64.to_string(),
                        },
                    },
                ],
            }],
            generation_config: GeminiGenerationConfig {
                response_modalities: vec!["IMAGE".to_string(), "TEXT".to_string()],
                response_mime_type: "image/png".to_string(),
            },
        };

        info!(model = %model, mode = %mode, "Sending AI edit request to Gemini");

        let response = self.client
            .post(&url)
            .header("x-goog-api-key", api_key)
            .json(&request)
            .send()
            .await
            .map_err(|e| AppError::Internal(format!("Gemini API request failed: {}", e)))?;

        let status = response.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(AppError::AiQuotaExceeded);
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(AppError::AiInvalidKey);
        }
        if status == reqwest::StatusCode::BAD_REQUEST {
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::AiBadRequest(body));
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::Internal(format!("Gemini API error {}: {}", status, body)));
        }

        let gemini_response: GeminiResponse = response.json().await.map_err(|e| {
            AppError::Internal(format!("Failed to parse Gemini response: {}", e))
        })?;

        if let Some(error) = gemini_response.error {
            return Err(AppError::AiBadRequest(error.message));
        }

        let candidates = gemini_response.candidates
            .ok_or_else(|| AppError::Internal("No candidates in Gemini response".to_string()))?;

        for candidate in &candidates {
            for part in &candidate.content.parts {
                if let GeminiResponsePart::InlineData { inline_data } = part {
                    let image_data = base64::engine::general_purpose::STANDARD
                        .decode(&inline_data.data)
                        .map_err(|e| AppError::Internal(format!("Failed to decode image: {}", e)))?;

                    return Ok(AiEditResult {
                        image_data,
                        mime_type: inline_data.mime_type.clone(),
                        model: model.clone(),
                    });
                }
            }
        }

        Err(AppError::Internal("No image data in Gemini response".to_string()))
    }

    async fn execute_openai_edit(
        &self,
        api_key: &str,
        image_base64: &str,
        prompt: &str,
        mode: &str,
    ) -> Result<AiEditResult, AppError> {
        let model = "gpt-image-1";
        let url = "https://api.openai.com/v1/images/edits";

        let system_prompt = match mode {
            "edit" => format!("Edit this image: {}", prompt),
            "remove" => format!("Remove the following from this image: {}", prompt),
            "replace_bg" => format!("Replace the background of this image with: {}", prompt),
            "relight" => format!("Relight this image: {}", prompt),
            "expand" => format!("Expand this image outward: {}", prompt),
            _ => prompt.to_string(),
        };

        // OpenAI images/edits expects multipart form with image file + prompt
        // Decode the base64 image to bytes for the multipart upload
        let image_bytes = base64::engine::general_purpose::STANDARD
            .decode(image_base64)
            .map_err(|e| AppError::Internal(format!("Failed to decode image base64: {}", e)))?;

        let image_part = reqwest::multipart::Part::bytes(image_bytes)
            .file_name("image.png")
            .mime_str("image/png")
            .map_err(|e| AppError::Internal(format!("Multipart error: {}", e)))?;

        let form = reqwest::multipart::Form::new()
            .text("model", model.to_string())
            .text("prompt", system_prompt)
            .part("image", image_part);

        info!(model = %model, mode = %mode, "Sending AI edit request to OpenAI");

        let response = self.client
            .post(url)
            .header("Authorization", format!("Bearer {}", api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| AppError::Internal(format!("OpenAI API request failed: {}", e)))?;

        let status = response.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(AppError::AiQuotaExceeded);
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(AppError::AiInvalidKey);
        }
        if status == reqwest::StatusCode::BAD_REQUEST {
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::AiBadRequest(body));
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::Internal(format!("OpenAI API error {}: {}", status, body)));
        }

        let resp_json: serde_json::Value = response.json().await.map_err(|e| {
            AppError::Internal(format!("Failed to parse OpenAI response: {}", e))
        })?;

        // OpenAI returns { data: [{ b64_json: "..." }] }
        let b64_data = resp_json
            .pointer("/data/0/b64_json")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AppError::Internal("No image data in OpenAI response".to_string()))?;

        let image_data = base64::engine::general_purpose::STANDARD
            .decode(b64_data)
            .map_err(|e| AppError::Internal(format!("Failed to decode OpenAI image: {}", e)))?;

        Ok(AiEditResult {
            image_data,
            mime_type: "image/png".to_string(),
            model: model.to_string(),
        })
    }

    async fn execute_imagen_edit(
        &self,
        api_key: &str,
        image_base64: &str,
        prompt: &str,
        mode: &str,
    ) -> Result<AiEditResult, AppError> {
        let model = "imagen-3.0-edit-001";
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:predict",
            model
        );

        let system_prompt = match mode {
            "edit" => format!("Edit this image: {}", prompt),
            "remove" => format!("Remove the following from this image: {}", prompt),
            "replace_bg" => format!("Replace the background of this image with: {}", prompt),
            "relight" => format!("Relight this image: {}", prompt),
            "expand" => format!("Expand this image outward: {}", prompt),
            _ => prompt.to_string(),
        };

        let body = serde_json::json!({
            "instances": [{
                "prompt": system_prompt,
                "image": {
                    "bytesBase64Encoded": image_base64
                }
            }],
            "parameters": {
                "sampleCount": 1
            }
        });

        info!(model = %model, mode = %mode, "Sending AI edit request to Imagen");

        let response = self.client
            .post(&url)
            .header("x-goog-api-key", api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| AppError::Internal(format!("Imagen API request failed: {}", e)))?;

        let status = response.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(AppError::AiQuotaExceeded);
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(AppError::AiInvalidKey);
        }
        if status == reqwest::StatusCode::BAD_REQUEST {
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::AiBadRequest(body));
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AppError::Internal(format!("Imagen API error {}: {}", status, body)));
        }

        let resp_json: serde_json::Value = response.json().await.map_err(|e| {
            AppError::Internal(format!("Failed to parse Imagen response: {}", e))
        })?;

        // Imagen returns { predictions: [{ bytesBase64Encoded: "..." }] }
        let b64_data = resp_json
            .pointer("/predictions/0/bytesBase64Encoded")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AppError::Internal("No image data in Imagen response".to_string()))?;

        let image_data = base64::engine::general_purpose::STANDARD
            .decode(b64_data)
            .map_err(|e| AppError::Internal(format!("Failed to decode Imagen image: {}", e)))?;

        Ok(AiEditResult {
            image_data,
            mime_type: "image/png".to_string(),
            model: model.to_string(),
        })
    }
}
