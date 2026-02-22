use serde::{Deserialize, Serialize};
use std::sync::Arc;
use chrono::Utc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobStatus {
    Pending,
    Processing,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobKind {
    Preview,
    Export,
    AiEdit,
    AutoEnhance,
    Optimize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobInfo {
    pub job_id: String,
    pub kind: JobKind,
    pub status: JobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<super::error::ProblemDetail>,
    pub created_at: String,
}

#[derive(Debug)]
pub struct Job {
    pub info: parking_lot::RwLock<JobInfo>,
}

impl Job {
    pub fn new(kind: JobKind) -> (String, Arc<Self>) {
        let job_id = uuid::Uuid::new_v4().to_string();
        let info = JobInfo {
            job_id: job_id.clone(),
            kind,
            status: JobStatus::Pending,
            progress: None,
            result: None,
            error: None,
            created_at: Utc::now().to_rfc3339(),
        };
        (job_id, Arc::new(Self {
            info: parking_lot::RwLock::new(info),
        }))
    }

    pub fn set_processing(&self) {
        let mut info = self.info.write();
        info.status = JobStatus::Processing;
    }

    pub fn set_progress(&self, progress: f64) {
        let mut info = self.info.write();
        info.progress = Some(progress);
    }

    pub fn set_complete(&self, result: serde_json::Value) {
        let mut info = self.info.write();
        info.status = JobStatus::Complete;
        info.progress = Some(1.0);
        info.result = Some(result);
    }

    pub fn set_failed(&self, error: super::error::ProblemDetail) {
        let mut info = self.info.write();
        info.status = JobStatus::Failed;
        info.error = Some(error);
    }

    pub fn get_info(&self) -> JobInfo {
        self.info.read().clone()
    }
}
