mod handlers;
mod models;
mod services;

use std::sync::Arc;
use axum::{
    Router,
    middleware,
    extract::Request,
    http::{HeaderMap, Method, StatusCode, header::HeaderName},
    response::Response,
    body::Body,
};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::Span;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use models::config::AppConfig;
use models::error::AppError;
use services::cache::FileCache;
use services::render::RenderService;
use services::ai_proxy::AiProxyService;
use services::job_manager::JobManager;
use services::auto_enhance::AutoEnhanceService;
use services::optimize::OptimizeService;
use services::rate_limiter::RateLimiter;

pub struct AppState {
    pub config: Arc<AppConfig>,
    pub cache: FileCache,
    pub render: RenderService,
    pub ai_proxy: AiProxyService,
    pub jobs: JobManager,
    pub auto_enhance: AutoEnhanceService,
    pub optimize: OptimizeService,
    pub render_limiter: RateLimiter,
    pub ai_limiter: RateLimiter,
    pub upload_limiter: RateLimiter,
}

async fn session_token_middleware(
    headers: HeaderMap,
    request: Request<Body>,
    next: axum::middleware::Next,
) -> Result<Response, AppError> {
    let method = request.method().clone();
    let path = request.uri().path().to_string();

    // Skip session token check for health, version, OPTIONS, and GET requests
    if path == "/api/health" || path == "/api/version" || method == Method::OPTIONS || method == Method::GET {
        return Ok(next.run(request).await);
    }

    if method == Method::POST || method == Method::DELETE {
        let token = headers
            .get("X-Session-Token")
            .and_then(|v| v.to_str().ok());

        if token.map_or(true, |t| t.is_empty()) {
            return Err(AppError::MissingSessionToken);
        }
    }

    Ok(next.run(request).await)
}

/// Headers that must never appear in logs.
const SENSITIVE_HEADERS: &[&str] = &[
    "x-gemini-key",
    "x-anthropic-key",
    "x-openai-key",
    "authorization",
];

fn is_sensitive_header(name: &HeaderName) -> bool {
    let lower = name.as_str();
    SENSITIVE_HEADERS.iter().any(|&s| lower == s)
}

/// Reject plain-HTTP requests when the server is bound to a non-loopback address.
async fn require_https_middleware(
    request: Request<Body>,
    next: axum::middleware::Next,
) -> Result<Response, StatusCode> {
    // If the request came through a reverse proxy that set X-Forwarded-Proto, trust it.
    let proto = request
        .headers()
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok());

    let is_https = proto.map_or(false, |p| p.eq_ignore_ascii_case("https"));
    if !is_https {
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(request).await)
}

async fn request_id_middleware(
    request: Request<Body>,
    next: axum::middleware::Next,
) -> Response {
    let request_id = uuid::Uuid::new_v4().to_string();
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert("X-Request-Id", request_id.parse().unwrap());
    response
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = AppConfig::from_env();

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    handlers::health::init_start_time();

    let config = Arc::new(config);
    let cache = FileCache::new(config.clone()).await?;
    let render = RenderService::new(config.clone());
    let ai_proxy = AiProxyService::new(config.clone());
    let jobs = JobManager::new(config.idempotency_ttl_secs);
    let auto_enhance = AutoEnhanceService::new(&config.models_dir);
    let optimize = OptimizeService::new(&config.models_dir);
    let render_limiter = RateLimiter::new(config.rate_limit_render);
    let ai_limiter = RateLimiter::new(config.rate_limit_ai);
    let upload_limiter = RateLimiter::new(config.rate_limit_upload);

    let state = Arc::new(AppState {
        config: config.clone(),
        cache,
        render,
        ai_proxy,
        jobs,
        auto_enhance,
        optimize,
        render_limiter,
        ai_limiter,
        upload_limiter,
    });

    let cors_origins: Vec<_> = config.allowed_origins.iter()
        .filter_map(|o| o.parse().ok())
        .collect();

    let cors = CorsLayer::new()
        .allow_origin(cors_origins)
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers([
            "Content-Type".parse().unwrap(),
            "X-Gemini-Key".parse().unwrap(),
            "X-Anthropic-Key".parse().unwrap(),
            "X-OpenAI-Key".parse().unwrap(),
            "X-Request-Id".parse().unwrap(),
            "X-Session-Token".parse().unwrap(),
            "Idempotency-Key".parse().unwrap(),
        ])
        .expose_headers([
            "X-Request-Id".parse().unwrap(),
            "Retry-After".parse().unwrap(),
        ]);

    let api_v1 = Router::new()
        .route("/files/upload", axum::routing::post(handlers::files::upload_file))
        .route("/files/:file_id", axum::routing::get(handlers::files::get_file))
        .route("/files/:file_id", axum::routing::delete(handlers::files::delete_file))
        .route("/renders/thumbnail", axum::routing::post(handlers::renders::create_thumbnail))
        .route("/renders/preview", axum::routing::post(handlers::renders::create_preview))
        .route("/renders/export", axum::routing::post(handlers::renders::create_export))
        .route("/renders/base-render/stream", axum::routing::get(handlers::renders::base_render_stream))
        .route("/ai/edit", axum::routing::post(handlers::ai::create_ai_edit))
        .route("/auto-enhance/models", axum::routing::get(handlers::auto_enhance::list_models))
        .route("/auto-enhance/run", axum::routing::post(handlers::auto_enhance::run_auto_enhance))
        .route("/optimize/models", axum::routing::get(handlers::optimize::list_optimize_models))
        .route("/optimize/run", axum::routing::post(handlers::optimize::run_optimize))
        .route("/optimize/masks", axum::routing::post(handlers::optimize::compute_masks))
        .route("/jobs/:job_id", axum::routing::get(handlers::jobs::get_job));

    // Trace layer that redacts sensitive API-key headers from log output
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(|request: &Request<Body>| {
            // Build a sanitized header summary (redact key headers)
            let safe_headers: Vec<String> = request
                .headers()
                .keys()
                .map(|name| {
                    if is_sensitive_header(name) {
                        format!("{}=[REDACTED]", name)
                    } else {
                        let val = request
                            .headers()
                            .get(name)
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("");
                        format!("{}={}", name, val)
                    }
                })
                .collect();

            tracing::info_span!(
                "http_request",
                method = %request.method(),
                uri = %request.uri(),
                headers = %safe_headers.join(", "),
            )
        })
        .on_response(|response: &Response, latency: std::time::Duration, _span: &Span| {
            tracing::info!(
                status = response.status().as_u16(),
                latency_ms = latency.as_millis() as u64,
                "response",
            );
        });

    // Enforce HTTPS when explicitly configured or when bound to a remote address
    let is_remote = {
        let addr_str = config.listen_addr.as_str();
        let host = addr_str.rsplit_once(':').map(|(h, _)| h).unwrap_or(addr_str);
        !matches!(host, "127.0.0.1" | "::1" | "localhost" | "0.0.0.0")
    };
    let enforce_https = config.require_https || is_remote;

    let mut app = Router::new()
        .nest("/api/v1", api_v1)
        .route("/api/health", axum::routing::get(handlers::health::health_check))
        .route("/api/version", axum::routing::get(handlers::health::version))
        .layer(axum::extract::DefaultBodyLimit::max(config.max_upload_bytes as usize))
        .layer(middleware::from_fn(request_id_middleware))
        .layer(middleware::from_fn(session_token_middleware))
        .layer(trace_layer)
        .layer(cors)
        .with_state(state.clone());

    if enforce_https {
        tracing::warn!(
            "HTTPS enforced (require_https={}, listen_addr={}); non-HTTPS requests will be rejected",
            config.require_https, config.listen_addr
        );
        app = app.layer(middleware::from_fn(require_https_middleware));
    }

    let cleanup_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        loop {
            interval.tick().await;
            cleanup_state.cache.cleanup_expired().await;
            cleanup_state.jobs.cleanup_old_jobs();
            cleanup_state.render_limiter.cleanup();
            cleanup_state.ai_limiter.cleanup();
            cleanup_state.upload_limiter.cleanup();
        }
    });

    let addr = config.listen_addr.clone();
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    tracing::info!("Shutdown signal received, draining connections...");
}
