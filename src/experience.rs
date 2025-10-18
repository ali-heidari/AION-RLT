use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Experience {
    pub features: Vec<f32>,
    pub action: u8,
    pub latency_ms: f32,
    pub reward: f32,
    pub success: bool,
    pub timestamp_ms: u128,
}



