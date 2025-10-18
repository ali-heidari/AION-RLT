mod configurations;
mod experience;
mod infer_action;
mod mock;
mod model;
mod node;
mod reply_buffer;
mod reward;
mod worker;

use log::{info, warn};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

use crate::{
    experience::Experience, infer_action::infer_action, mock::SyntheticState, node::Node,
    reward::compute_reward_with_success, worker::Worker,
};

#[tokio::main]
async fn main() {
    let mut state = SyntheticState::new();
    let node = Arc::new(Node::new());
    let n = node.clone();

    tokio::spawn(async move {
        let worker = Worker::new(1);
        if let Err(e) = worker.start(n).await {
            warn!("training worker error: {:?}", e);
        }
    });

    
}
