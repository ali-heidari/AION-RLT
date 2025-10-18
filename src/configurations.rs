use std::sync::Arc;

use config::{Config, Environment, File};
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Configurations {
    pub train_interval_secs: u64,
    pub batch_size: u32,
    pub total_batches: usize,
    pub input_number: usize,
    pub output_number: usize,
    pub hidden_layers: usize,
    pub reply_capacity: usize,
    pub model_name: String,
}

fn load_config() -> Result<Arc<Configurations>, config::ConfigError> {
    let settings = Config::builder()
        .add_source(File::with_name("config.toml"))
        .add_source(Environment::with_prefix("APP").separator("__"))
        .build()?;
    let worker_config: Arc<Configurations> = Arc::new(settings.try_deserialize()?);
    Ok(worker_config)
}

lazy_static::lazy_static! {
    pub static ref CONFIG: Arc<Configurations> = load_config().expect("Failed to load configuration");
}
