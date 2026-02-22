use std::sync::Arc;
use dashmap::DashMap;
use chrono::{Utc, Duration};
use crate::models::jobs::{Job, JobKind};

pub struct JobManager {
    jobs: DashMap<String, Arc<Job>>,
    idempotency_keys: DashMap<String, (String, chrono::DateTime<chrono::Utc>)>,
    idempotency_ttl_secs: u64,
}

impl JobManager {
    pub fn new(idempotency_ttl_secs: u64) -> Self {
        Self {
            jobs: DashMap::new(),
            idempotency_keys: DashMap::new(),
            idempotency_ttl_secs,
        }
    }

    pub fn create_job(&self, kind: JobKind, idempotency_key: Option<&str>) -> (String, Arc<Job>) {
        if let Some(key) = idempotency_key {
            if let Some(entry) = self.idempotency_keys.get(key) {
                let (job_id, expires) = entry.value();
                if *expires > Utc::now() {
                    if let Some(job) = self.jobs.get(job_id) {
                        return (job_id.clone(), job.value().clone());
                    }
                }
            }
        }

        let (job_id, job) = Job::new(kind);
        self.jobs.insert(job_id.clone(), job.clone());

        if let Some(key) = idempotency_key {
            let expires = Utc::now() + Duration::seconds(self.idempotency_ttl_secs as i64);
            self.idempotency_keys.insert(key.to_string(), (job_id.clone(), expires));
        }

        (job_id, job)
    }

    pub fn get_job(&self, job_id: &str) -> Option<Arc<Job>> {
        self.jobs.get(job_id).map(|j| j.value().clone())
    }

    pub fn cleanup_old_jobs(&self) {
        let cutoff = Utc::now() - Duration::hours(1);
        let mut to_remove = vec![];

        for entry in self.jobs.iter() {
            let info = entry.value().get_info();
            if let Ok(created) = chrono::DateTime::parse_from_rfc3339(&info.created_at) {
                if created < cutoff {
                    to_remove.push(entry.key().clone());
                }
            }
        }

        for id in to_remove {
            self.jobs.remove(&id);
        }

        // Also clean up expired idempotency keys
        let now = Utc::now();
        let mut keys_to_remove = vec![];
        for entry in self.idempotency_keys.iter() {
            if entry.value().1 < now {
                keys_to_remove.push(entry.key().clone());
            }
        }
        for key in keys_to_remove {
            self.idempotency_keys.remove(&key);
        }
    }
}
