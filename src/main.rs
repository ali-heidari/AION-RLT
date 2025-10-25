mod configurations;
mod experience;
mod infer_action;
mod mock;
mod model;
mod node;
mod reply_buffer;
mod reward;
mod worker;

use aion_math::math::Math;
use color_eyre::Result;
use crossterm::event::{self, Event};
use ratatui::{DefaultTerminal, Frame};
use std::ops::Div;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use crate::mock::SyntheticState;
use crate::node::Node;
use crate::worker::Worker;

fn use_tui() -> Result<(), color_eyre::Report> {
    color_eyre::install()?;
    let terminal = ratatui::init();
    let result = run(terminal);
    ratatui::restore();
    result
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let mut state = SyntheticState::new();
    let node = Arc::new(Node::new(0.05, 0.5));

    let n = node.clone();
    Node::start_training(n);

    let mut math = Math::new();
    // println!(
    //     "(vec![cpu usage,memory usage, swap, disk usage, throughput, latency],ACTION= your answer)"
    // );

    // loop {
    //     // *node.batch_sampled.write().unwrap() = true;
    //     let features = state.next();
    //     // println!(
    //     //     "(vec![{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}],ACTION)",
    //     //     features[0], features[1], features[2], features[3], features[4], features[5]
    //     // );

    //     node.next(features, &mut math);

    //     sleep(Duration::from_millis(10)).await;
    // }
    let file_path = Path::new("vmCloud_data.csv");
    let mut rdr = csv::Reader::from_path(file_path)?;

        use rand::Rng;
        let mut rng = rand::thread_rng();

    // Iterate over the records
    for result in rdr.records() {
        let record = result?;
        if record.get(2).unwrap().is_empty() || record.get(3).unwrap().is_empty() {
            continue;
        }
        let max=1.0;//if rng.gen_bool(0.04) {1.0} else {0.4};
        let features = state.next(
            record.get(2).unwrap().parse::<f32>().unwrap().div(100.0).clamp(0.05, max),
            record.get(3).unwrap().parse::<f32>().unwrap().div(100.0).clamp(0.05, max),
        );
        node.next(features, &mut math);
         sleep(Duration::from_millis(10)).await;
    }
    Ok(())
}

fn run(mut terminal: DefaultTerminal) -> Result<()> {
    loop {
        terminal.draw(render)?;
        if matches!(event::read()?, Event::Key(_)) {
            break Result::Ok(());
        }
    }
}

fn render(frame: &mut Frame) {
    frame.render_widget("hello world", frame.area());
}
