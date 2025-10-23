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
async fn main() {
    env_logger::init();
    let mut state = SyntheticState::new();
    let node = Arc::new(Node::new());

    let n = node.clone();
    Worker::start(n);

    loop {
        let features = state.next();
        node.next(features);

        sleep(Duration::from_millis(10)).await;
    }
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
