use crate::experience::Experience;
use crate::get_config as CONFIG;

use super::model::Model;
use super::node::Node;
use anyhow::{Ok, Result};
use log::{info, warn};
use ndarray::{s, Array1, Array2};
use std::{sync::Arc, time::Duration};
use tokio::time::sleep;

pub struct Worker {
    id: u32,
    learning_rate: f32,
}

impl Worker {
    pub fn new(id: u32) -> Self {
        Self {
            id: id,
            learning_rate: 0.00001,
        }
    }

    fn train_on_batch(
        &self,
        batch: Vec<Experience>,
        mut model: std::sync::RwLockWriteGuard<'_, Model>,
    ) -> Result<f32> {
        let mut inputs = Array2::<f32>::zeros((batch.len(), CONFIG().input_number as usize));
        let mut predicted_action = Array1::<usize>::zeros(batch.len());
        let mut rewards = Array1::<f32>::zeros(batch.len());

        for (i, experience) in batch.iter().enumerate() {
            inputs.slice_mut(s![i, ..]).assign(&Array1::from(
                experience.features[..CONFIG().input_number].to_vec(),
            ));
            predicted_action[i] = experience.action as usize;
            rewards[i] = experience.reward.clamp(-1.0, 1.0);
        }

        //  let loss = model.backward(&x, &y, 0.01);
        let loss = model.reinforce(&inputs, &predicted_action, &rewards, self.learning_rate);


        Ok(loss)
    }

    pub fn start(node: Arc<Node>) {
        tokio::spawn(async move {
            let mut worker = Worker::new(1);
            if let Err(e) = worker.start_to_work(node).await {
                warn!("training worker error: {:?}", e);
            }
        });
    }

    async fn start_to_work(&mut self, node: Arc<Node>) -> Result<()> {
        info!("Worker {} is working!", self.id);

        let mut batch_counter = 0;

        loop {
            sleep(Duration::from_secs(CONFIG().train_interval_secs)).await;
            self.learning_rate = *node.lr.read().unwrap();

            let buffer = node.buffer.read().unwrap();
            if buffer.len() < CONFIG().batch_size as usize {
                drop(buffer);
                continue;
            }

            if batch_counter > CONFIG().total_batches as u32 {
                drop(buffer);
                info!(
                    "Training complete after {} batches.",
                    CONFIG().total_batches
                );
                break;
            }
            batch_counter += 1;

            let batch = buffer.sample(CONFIG().batch_size as usize);
            drop(buffer);

            let model = node.model.write().unwrap();
            let loss = self.train_on_batch(batch, model).unwrap();
            node.set_loss(batch_counter, loss);
        }
        Ok(())
    }
}
