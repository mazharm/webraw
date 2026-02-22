use axum::Json;
use serde::Serialize;
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthResponse {
    pub status: String,
    pub uptime: f64,
    pub cache_usage: CacheUsage,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheUsage {
    pub used_bytes: u64,
    pub max_bytes: u64,
}

static START_TIME: OnceLock<Instant> = OnceLock::new();

pub fn init_start_time() {
    START_TIME.get_or_init(Instant::now);
}

pub async fn health_check() -> Json<HealthResponse> {
    let uptime = START_TIME
        .get()
        .map(|s| s.elapsed().as_secs_f64())
        .unwrap_or(0.0);

    Json(HealthResponse {
        status: "ok".to_string(),
        uptime,
        cache_usage: CacheUsage {
            used_bytes: 0,
            max_bytes: 0,
        },
    })
}

pub async fn version() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "apiVersion": "v1",
        "buildHash": env!("CARGO_PKG_VERSION")
    }))
}
