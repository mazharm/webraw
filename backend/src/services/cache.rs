use std::path::{Path, PathBuf};
use std::sync::Arc;
use chrono::{DateTime, Utc, Duration};
use dashmap::DashMap;
use sha2::{Sha256, Digest};
use tokio::fs;
use tracing::{info, warn};
use crate::models::config::AppConfig;
use crate::models::error::AppError;

#[derive(Debug, Clone)]
pub struct CachedFile {
    pub file_id: String,
    pub path: PathBuf,
    pub mime_type: String,
    pub size_bytes: u64,
    pub original_filename: String,
    pub hash: Option<String>,
    pub exif: Option<serde_json::Value>,
    pub expires_at: DateTime<Utc>,
}

pub struct FileCache {
    config: Arc<AppConfig>,
    files: DashMap<String, CachedFile>,
    cache_dir: PathBuf,
}

impl FileCache {
    pub async fn new(config: Arc<AppConfig>) -> anyhow::Result<Self> {
        let cache_dir = PathBuf::from(&config.cache_dir);
        fs::create_dir_all(&cache_dir).await?;

        Ok(Self {
            config,
            files: DashMap::new(),
            cache_dir,
        })
    }

    pub async fn store_file(
        &self,
        filename: &str,
        data: &[u8],
        mime_type: &str,
        extended_ttl: bool,
    ) -> Result<CachedFile, AppError> {
        if data.len() as u64 > self.config.max_upload_bytes {
            return Err(AppError::FileTooLarge(data.len() as u64));
        }

        let file_id = uuid::Uuid::new_v4().to_string();
        let ext = Path::new(filename)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("bin");
        let stored_name = format!("{}.{}", file_id, ext);
        let path = self.cache_dir.join(&stored_name);

        fs::write(&path, data).await.map_err(|e| {
            AppError::Internal(format!("Failed to write file: {}", e))
        })?;

        let mut hasher = Sha256::new();
        hasher.update(data);
        let hash = hex::encode(hasher.finalize());

        let ttl_secs = if extended_ttl {
            self.config.cache_ttl_fallback_secs
        } else {
            self.config.cache_ttl_secs
        };

        let exif = extract_basic_exif(data, mime_type);

        let cached = CachedFile {
            file_id: file_id.clone(),
            path,
            mime_type: mime_type.to_string(),
            size_bytes: data.len() as u64,
            original_filename: filename.to_string(),
            hash: Some(hash),
            exif,
            expires_at: Utc::now() + Duration::seconds(ttl_secs as i64),
        };

        self.files.insert(file_id.clone(), cached.clone());
        info!(file_id = %file_id, size = data.len(), "File cached");

        Ok(cached)
    }

    pub fn get_file(&self, file_id: &str) -> Result<CachedFile, AppError> {
        match self.files.get(file_id) {
            Some(entry) => {
                if entry.expires_at < Utc::now() {
                    drop(entry);
                    self.files.remove(file_id);
                    Err(AppError::FileNotFound(file_id.to_string()))
                } else {
                    Ok(entry.clone())
                }
            }
            None => Err(AppError::FileNotFound(file_id.to_string())),
        }
    }

    pub async fn read_file_bytes(&self, file_id: &str) -> Result<Vec<u8>, AppError> {
        let cached = self.get_file(file_id)?;
        fs::read(&cached.path).await.map_err(|e| {
            AppError::Internal(format!("Failed to read cached file: {}", e))
        })
    }

    pub async fn store_artifact(
        &self,
        data: &[u8],
        mime_type: &str,
        source_file_id: &str,
    ) -> Result<CachedFile, AppError> {
        let file_id = uuid::Uuid::new_v4().to_string();
        let ext = match mime_type {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/tiff" => "tiff",
            _ => "bin",
        };
        let stored_name = format!("{}.{}", file_id, ext);
        let path = self.cache_dir.join(&stored_name);

        fs::write(&path, data).await.map_err(|e| {
            AppError::Internal(format!("Failed to write artifact: {}", e))
        })?;

        let cached = CachedFile {
            file_id: file_id.clone(),
            path,
            mime_type: mime_type.to_string(),
            size_bytes: data.len() as u64,
            original_filename: format!("artifact_{}", source_file_id),
            hash: None,
            exif: None,
            expires_at: Utc::now() + Duration::seconds(self.config.cache_ttl_secs as i64),
        };

        self.files.insert(file_id, cached.clone());
        Ok(cached)
    }

    pub async fn delete_file(&self, file_id: &str) -> Result<(), AppError> {
        if let Some((_, cached)) = self.files.remove(file_id) {
            let _ = fs::remove_file(&cached.path).await;
            info!(file_id = %file_id, "File deleted from cache");
            Ok(())
        } else {
            Err(AppError::FileNotFound(file_id.to_string()))
        }
    }

    pub async fn cleanup_expired(&self) {
        let now = Utc::now();
        let mut to_remove = vec![];

        for entry in self.files.iter() {
            if entry.expires_at < now {
                to_remove.push(entry.key().clone());
            }
        }

        for file_id in to_remove {
            if let Some((_, cached)) = self.files.remove(&file_id) {
                let _ = fs::remove_file(&cached.path).await;
                warn!(file_id = %file_id, "Expired file removed from cache");
            }
        }
    }
}

fn extract_basic_exif(_data: &[u8], mime_type: &str) -> Option<serde_json::Value> {
    if mime_type.starts_with("image/") {
        Some(serde_json::json!({
            "width": null,
            "height": null,
            "make": null,
            "model": null,
            "iso": null,
            "focalLength": null,
            "exposureTime": null,
            "fNumber": null,
        }))
    } else {
        None
    }
}
