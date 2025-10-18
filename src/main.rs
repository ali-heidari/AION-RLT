mod configurations;
mod experience;
mod infer_action;
mod mock;
mod model;
mod node;
mod reply_buffer;
mod reward;
mod worker;

use color_eyre::Result;
use crossterm::event::{self, Event};
use log::{info, warn};
use ratatui::{DefaultTerminal, Frame};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

use crate::{
    experience::Experience, infer_action::infer_action, mock::SyntheticState, node::Node,
    reward::compute_reward_with_success, worker::Worker,
};

fn use_tui() {
    color_eyre::install()?;
    let terminal = ratatui::init();
    let result = run(terminal);
    ratatui::restore();
    result
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let mut state = SyntheticState::new();
    let node = Arc::new(Node::new());
    let n = node.clone();

    tokio::spawn(async move {
        let worker = Worker::new(1);
        if let Err(e) = worker.start(n).await {
            warn!("training worker error: {:?}", e);
        }
    });

    let mut ac: [i32; 3] = [0, 0, 0];
    let mut crc: [i32; 3] = [0, 0, 0];
    let mut sr_avg: [f32; 3] = [0.0, 0.0, 0.0];
    loop {
        let features = state.next();
        let (action, logits, probs) = infer_action(&node, &features);
        let (mut reward, success) = compute_reward_with_success(&features, action as u8);
        ac[action as usize] += 1;
        crc[action as usize] += if success { 1 } else { 0 };
        sr_avg[action as usize] = crc[action] as f32 / ac[action] as f32;

        if (ac[0] + ac[1] + ac[2]) % 50 == 0 {
            info!("Current features: CPU: {:.2}, MEM: {:.2}, SWAP: {:.2}, THROUGHPUT: {:.2}, LATENCY: {:.2}",
        features[0], features[1], features[2], features[3], features[4]);

            info!("logits: {:?}", logits);
            info!("probs: {:?}", probs);
            info!(
                "Action taken: {}, Reward: {:.3}, Success: {}",
                action, reward, success
            );

            info!(
                "Total action: [High pressure] {}\t[Normal] {}\t[Low pressure] {}",
                ac[0], ac[1], ac[2]
            );
            info!(
                "Success Rate: [High pressure] {:.2}\t[Normal] {:.2}\t[Low pressure] {:.2}",
                sr_avg[0], sr_avg[1], sr_avg[2],
            );

            let mut temprature = *node.temperature.write().unwrap();
            let mut epsilon = *node.epsilon.write().unwrap();
            if (ac[0] + ac[1] + ac[2]) % 64 * 100 == 0 {
                let mean = ac.iter().sum::<i32>() as f32 / ac.len() as f32;
                let variance =
                    ac.iter().map(|&x| (x as f32 - mean).powi(2)).sum::<f32>() / ac.len() as f32;
                let std_dev = variance.sqrt();
                let relative_std = std_dev / mean;
                if (relative_std > 0.05) {
                    epsilon = (epsilon * mean / *ac.iter().min().unwrap() as f32).min(0.5);
                    temprature = (temprature * mean / *ac.iter().min().unwrap() as f32).min(5.0);
                    if (success
                        && action
                            == ac
                                .iter()
                                .position(|x| *x == *ac.iter().min().unwrap())
                                .unwrap() as usize)
                    {
                        reward += 0.3;
                    }
                } else {
                    epsilon = (epsilon * *ac.iter().min().unwrap() as f32 / mean).max(0.1);
                    temprature = (temprature * *ac.iter().min().unwrap() as f32 / mean).max(1.0);
                }

                info!("relative_std: {}, variance: {}", relative_std, variance);

                *node.temperature.write().unwrap() = temprature;
                *node.epsilon.write().unwrap() = epsilon;
            }
            info!("epsilon: {}, temperatur: {}", epsilon, temprature);
            println!("-----------------------------------------------------------------");
            let ex = Experience {
                features: features.clone(),
                action: action as u8,
                latency_ms: *features.get(5).unwrap(),
                reward: reward,
                success: success,
                timestamp_ms: Instant::now().elapsed().as_millis(),
            };
            node.buffer.write().unwrap().push(ex);
        }
        sleep(Duration::from_millis(10)).await;
    }
}

fn run(mut terminal: DefaultTerminal) -> Result<()> {
    loop {
        terminal.draw(render)?;
        if matches!(event::read()?, Event::Key(_)) {
            break Ok(());
        }
    }
}

fn render(frame: &mut Frame) {
    frame.render_widget("hello world", frame.area());
}
