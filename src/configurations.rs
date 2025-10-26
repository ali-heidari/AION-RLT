use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct Configurations {
    pub train_interval_secs: u64,
    pub batch_size: u32,
    pub total_batches: usize,
    pub input_number: usize,
    pub output_number: usize,
    pub hidden_layers: usize,
    pub reply_capacity: usize,
    pub model_name: String,
    pub log_interval: u64
}