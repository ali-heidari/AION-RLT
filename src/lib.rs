use std::sync::{Arc, OnceLock};

use crate::configurations::Configurations;

pub mod configurations;
mod experience;
mod infer_action;
mod model;
pub mod node;
mod reply_buffer;
mod worker;

pub static CONFIG: OnceLock<Arc<Configurations>> = OnceLock::new();

pub fn initialize(config: Arc<Configurations>) {
    CONFIG.set(config).unwrap()
}

pub fn get_config() -> Arc<Configurations> {
    let config = CONFIG.get().unwrap();
    config.clone()
}
