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
use ratatui::{DefaultTerminal, Frame};
use std::ops::Div;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::mock::SyntheticState;
use crate::node::{Node, RunningMode};

fn use_tui() -> Result<(), color_eyre::Report> {
    color_eyre::install()?;
    let terminal = ratatui::init();
    let result = run(terminal);
    ratatui::restore();
    result
}

fn get_features(state: &mut SyntheticState) -> Vec<f32> {
    let features = state.next(mock::Mode::Generative, 0.0, 0.0);
    features
}

fn from_csv_dataset(state: &mut SyntheticState) -> Result<()> {
    let file_path = Path::new("vmCloud_data.csv");
    let mut rdr = csv::Reader::from_path(file_path)?;
    let max = 1.0; //if rng.gen_bool(0.04) {1.0} else {0.4};
    for result in rdr.records() {
        let record = result?;
        if record.get(2).unwrap().is_empty() || record.get(3).unwrap().is_empty() {
            continue;
        }
        let max = 1.0; //if rng.gen_bool(0.04) {1.0} else {0.4};
        let features = state.next(
            mock::Mode::Inputs,
            record
                .get(2)
                .unwrap()
                .parse::<f32>()
                .unwrap()
                .div(100.0)
                .clamp(0.05, max),
            record
                .get(3)
                .unwrap()
                .parse::<f32>()
                .unwrap()
                .div(100.0)
                .clamp(0.05, max),
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let state = Arc::new(Mutex::new(SyntheticState::new()));
    let cloned_state = Arc::clone(&state);

    Node::start(
        move || get_features(&mut cloned_state.lock().unwrap()),
        RunningMode::Training,
    )
    .await;

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
