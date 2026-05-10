# AION-RLT Architecture

AION-RLT is a lightweight reinforcement learning trainer designed for embeddable applications.

## Core modules

- `src/lib.rs` — library entrypoint and configuration singleton
- `src/node.rs` — main orchestrator, model lifecycle, and training state
- `src/model.rs` — neural network implementation and policy gradient updates
- `src/worker.rs` — async training loop and batch processing
- `src/reply_buffer.rs` — experience replay buffer
- `src/infer_action.rs` — inference and action selection logic
