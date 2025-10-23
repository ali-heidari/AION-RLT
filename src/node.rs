use log::{info};

use crate::{
    configurations::CONFIG, experience::Experience, infer_action::infer_action, model::Model,
    reply_buffer::ReplayBuffer, reward::compute_reward_with_success,
};
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Instant,
};

struct ActionDetails {
    total: u32,
    success_count: u32,
}

impl ActionDetails {
    fn add_successful(&mut self) {
        self.total += 1;
        self.success_count += 1;
    }
    fn add_failure(&mut self) {
        self.total += 1;
    }
    fn get_success_rate(&self) -> u32 {
        if self.total == 0 {
            0 // Avoid division by zero
        } else {
            self.success_count / self.total
        }
    }
}

pub struct Node {
    pub model: Arc<RwLock<Model>>,
    pub buffer: Arc<RwLock<ReplayBuffer>>,
    pub loss: Arc<RwLock<f32>>,
    pub epsilon: RwLock<f32>,     // shared epsilon value
    pub temperature: RwLock<f32>, // shared temperature value
    pub batch_history: RwLock<HashMap<u32, f32>>,
    pub action_details: RwLock<HashMap<u32, ActionDetails>>,
}

impl Node {
    pub fn new() -> Self {
        Self {
            model: Arc::new(RwLock::new(Model::new().load_model().unwrap())),
            buffer: Arc::new(RwLock::new(ReplayBuffer::new(CONFIG.reply_capacity))),
            loss: Arc::new(RwLock::new(0.0)),
            epsilon: RwLock::new(0.05),
            temperature: RwLock::new(1.0),
            batch_history: RwLock::new(HashMap::new()),
            action_details: RwLock::new(HashMap::new()),
        }
    }

    pub fn set_loss(&self, batch_number: u32, loss: f32) {
        {
            *self.loss.write().unwrap() = loss;
            self.batch_history
                .write()
                .unwrap()
                .insert(batch_number, loss);
        }
    }

    fn save_cycle(&mut self, predicted_action: u32, succeed: bool) {
        let mut action_details_guard = self.action_details.write().unwrap();
        let action_details = action_details_guard.get_mut(&predicted_action).unwrap();

        if succeed {
            action_details.add_successful();
        } else {
            action_details.add_failure();
        }
    }

    fn report(
        &self,
        inputs: &[f32; 6],
        logits: [f32; 3],
        probs: [f32; 3],
        action: usize,
        reward: f32,
        success: bool,
    ) {
        // TODO: Change the log using struct that provides inputs as vectors and custom log
        info!("Current features: CPU: {:.2}, MEM: {:.2}, SWAP: {:.2}, THROUGHPUT: {:.2}, LATENCY: {:.2}",
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);

        info!("logits: {:?}", logits);
        info!("probs: {:?}", probs);
        info!(
            "Action taken: {}, Reward: {:.3}, Success: {}",
            action, reward, success
        );

        info!(
            "epsilon: {}, temperatur: {}, loss: {}",
            self.epsilon.read().unwrap(),
            self.temperature.read().unwrap(),
            self.loss.read().unwrap()
        );
        let action_details_guard = self.action_details.read().unwrap();
        let ac0 = action_details_guard.get(&0).unwrap();
        let ac1 = action_details_guard.get(&1).unwrap();
        let ac2 = action_details_guard.get(&2).unwrap();
        info!(
            "Total action: [High pressure] {}\t[Normal] {}\t[Low pressure] {}",
            ac0.total, ac1.total, ac2.total
        );
        info!(
            "Success Rate: [High pressure] {:.2}\t[Normal] {:.2}\t[Low pressure] {:.2}",
            ac0.get_success_rate(),
            ac1.get_success_rate(),
            ac2.get_success_rate(),
        );
        println!("-----------------------------------------------------------------");
    }

    pub fn next(&mut self, inputs: Vec<f32>) {
        let (action, logits, probs) = infer_action(self, &inputs);
        let (reward, success) = compute_reward_with_success(&inputs, action as u8);

        self.save_cycle(action as u32, success);

        let ex = Experience {
            features: inputs.clone(),
            action: action as u8,
            latency_ms: *inputs.get(5).unwrap(),
            reward: reward,
            success: success,
            timestamp_ms: Instant::now().elapsed().as_millis(),
        };
        self.buffer.write().unwrap().push(ex);

        if self.batch_history.read().unwrap().iter().last().unwrap().0 % 10 == 0 {
            self.report(
                inputs.as_slice().try_into().unwrap(),
                logits.into_raw_vec_and_offset().0.try_into().unwrap(),
                probs.as_slice().try_into().unwrap(),
                action,
                reward,
                success,
            );
        }
    }
}
