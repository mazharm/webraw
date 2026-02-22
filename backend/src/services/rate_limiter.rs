use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};
use parking_lot::Mutex;

struct TokenBucket {
    tokens: f64,
    max_tokens: f64,
    last_refill: DateTime<Utc>,
    refill_rate: f64, // tokens per second
}

impl TokenBucket {
    fn new(max_per_minute: u32) -> Self {
        Self {
            tokens: max_per_minute as f64,
            max_tokens: max_per_minute as f64,
            last_refill: Utc::now(),
            refill_rate: max_per_minute as f64 / 60.0,
        }
    }

    fn try_consume(&mut self) -> Result<(), u64> {
        let now = Utc::now();
        let elapsed = (now - self.last_refill).num_milliseconds() as f64 / 1000.0;
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;

        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else {
            let wait_secs = ((1.0 - self.tokens) / self.refill_rate).ceil() as u64;
            Err(wait_secs.max(1))
        }
    }
}

pub struct RateLimiter {
    buckets: Mutex<HashMap<String, TokenBucket>>,
    default_limit: u32,
}

impl RateLimiter {
    pub fn new(default_limit: u32) -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
            default_limit,
        }
    }

    pub fn check(&self, session_token: &str) -> Result<(), u64> {
        let mut buckets = self.buckets.lock();
        let bucket = buckets
            .entry(session_token.to_string())
            .or_insert_with(|| TokenBucket::new(self.default_limit));
        bucket.try_consume()
    }

    pub fn cleanup(&self) {
        let mut buckets = self.buckets.lock();
        let cutoff = Utc::now() - Duration::minutes(5);
        buckets.retain(|_, b| b.last_refill > cutoff);
    }
}
