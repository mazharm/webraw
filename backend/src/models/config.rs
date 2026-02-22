use std::env;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub listen_addr: String,
    pub allowed_origins: Vec<String>,
    pub gemini_model: String,
    pub claude_model: String,
    pub openai_model: String,
    pub cache_dir: String,
    pub cache_ttl_secs: u64,
    pub cache_ttl_fallback_secs: u64,
    pub cache_max_bytes: u64,
    pub max_upload_bytes: u64,
    pub max_parallel_renders: usize,
    pub max_parallel_exports: usize,
    pub rate_limit_render: u32,
    pub rate_limit_ai: u32,
    pub rate_limit_upload: u32,
    pub log_level: String,
    pub sse_timeout_secs: u64,
    pub idempotency_ttl_secs: u64,
    pub models_dir: String,
    pub require_https: bool,
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            listen_addr: env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string()),
            allowed_origins: env::var("ALLOWED_ORIGINS")
                .unwrap_or_else(|_| "http://localhost:5173,http://localhost:5174,http://localhost:3000".to_string())
                .split(',')
                .map(|s| s.trim().to_string())
                .collect(),
            gemini_model: env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.5-pro".to_string()),
            claude_model: env::var("CLAUDE_MODEL").unwrap_or_else(|_| "claude-sonnet-4-6".to_string()),
            openai_model: env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1".to_string()),
            cache_dir: env::var("CACHE_DIR").unwrap_or_else(|_| "./cache".to_string()),
            cache_ttl_secs: env::var("CACHE_TTL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(86400),
            cache_ttl_fallback_secs: env::var("CACHE_TTL_FALLBACK_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(259200),
            cache_max_bytes: env::var("CACHE_MAX_BYTES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10_737_418_240),
            max_upload_bytes: env::var("MAX_UPLOAD_BYTES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(104_857_600),
            max_parallel_renders: env::var("MAX_PARALLEL_RENDERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            max_parallel_exports: env::var("MAX_PARALLEL_EXPORTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            rate_limit_render: env::var("RATE_LIMIT_RENDER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30),
            rate_limit_ai: env::var("RATE_LIMIT_AI")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            rate_limit_upload: env::var("RATE_LIMIT_UPLOAD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            sse_timeout_secs: env::var("SSE_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30),
            idempotency_ttl_secs: env::var("IDEMPOTENCY_TTL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600),
            models_dir: env::var("MODELS_DIR").unwrap_or_else(|_| "./models".to_string()),
            require_https: env::var("REQUIRE_HTTPS")
                .ok()
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
        }
    }
}
