use crate::{configurations::CONFIG, model::Model, reply_buffer::ReplayBuffer};
use std::sync::{Arc, RwLock};

pub struct Node {
    pub model: Arc<RwLock<Model>>,
    pub buffer: Arc<RwLock<ReplayBuffer>>,
    pub loss: Arc<RwLock<f32>>,
    pub epsilon: RwLock<f32>,     // shared epsilon value
    pub temperature: RwLock<f32>, // shared temperature value
}

impl Node {
    pub fn new() -> Self {
        Self {
            model: Arc::new(RwLock::new(Model::new().load_model().unwrap())),
            buffer: Arc::new(RwLock::new(ReplayBuffer::new(CONFIG.reply_capacity))),
            loss: Arc::new(RwLock::new(0.0)),
            epsilon: RwLock::new(0.0),
            temperature: RwLock::new(0.5),
        }
    }
}
