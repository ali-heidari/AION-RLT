use serde::Deserialize;

use crate::node::RunningMode;

/// Compute backend used for the neural network math.
/// `Cpu` (default) runs everything through `ndarray`;
/// `Gpu` dispatches to wgpu compute shaders (falls back to CPU when no
/// compatible adapter is found).
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    #[default]
    Cpu,
    Gpu,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Configurations {
    pub interval_secs: u64,
    pub batch_size: u32,
    pub total_batches: usize,
    pub input_number: usize,
    pub output_number: usize,
    pub hidden_layers: usize,
    pub reply_capacity: usize,
    pub model_name: String,
    pub log_interval: u64,
    pub mode: RunningMode,
    #[serde(default)]
    pub backend: ComputeBackend,
}
