# AION-RLT - AI Coding Instructions

## Architecture Overview

**AION-RLT** (AION Reinforcement Learning Trainer) is a **general-purpose, lightweight RL framework** designed to be embedded in applications like trading bots, adaptive load balancers, or protocol agents. It's NOT specific to trading — it's a reusable RL engine.

**Core responsibility**: Provide modular, production-ready RL components that applications configure and plug in.

### Design Philosophy
- **Minimal & modular**: Single responsibility per component
- **Configurable**: Input/output dimensions, batch size, hyperparameters all tunable
- **Embeddable**: No opinions on what application does — just trains & infers
- **Async-ready**: Tokio integration for background training loops

---

## Key Components

### Node (`node.rs`)
Central orchestrator managing RL lifecycle for a single agent/application.

**Responsibilities:**
- Holds `Model` (neural network weights), `ReplayBuffer` (experience storage)
- Manages `epsilon` (exploration) and `temperature` (softmax control) parameters
- Tracks training metrics: `batch_history`, `success_rate_mean`, convergence flags
- Supports `RunningMode`: Infer, Training, TrainingWithInterval

**Key Methods:**
- `Node::run(mode, model_name)`: Creates Node with loaded/new model
- `node.infer_action(features)`: Returns (action_idx, logits, probabilities)
- `node.push_experience(experience)`: Adds to replay buffer for training

**Thread-Safe Pattern:**
```rust
pub struct Node {
    pub model: Arc<RwLock<Model>>,
    pub buffer: Arc<RwLock<ReplayBuffer>>,
    pub epsilon: RwLock<f32>,
    pub temperature: RwLock<f32>,
    // ... metrics
}
```
All shared state via Arc<RwLock<>> for safe concurrent access.

### Model (`model.rs`)
Two-layer neural network with ReLU hidden activation.

**Architecture:**
```
input (f32) → w1 (input_number × hidden_layers) + b1
           → ReLU activation
           → w2 (hidden_layers × output_number) + b2
           → Softmax (temperature-controlled)
           → action probabilities
```

**Training Algorithm:**
- REINFORCE policy gradient: ∇J = Σ ∇log(π(a|s)) * R(a)
- Batch updates: samples 128 experiences, computes loss, updates weights
- Learning rate: 0.00001 (tunable in Worker)

**Serialization:**
- Model saved as JSON after each batch
- Format: {w1, b1, w2, b2} serializable via serde

### ReplayBuffer (`reply_buffer.rs`)
Experience storage with uniform random sampling.

**Responsibilities:**
- Store experiences: (features, action, reward, success, timestamp)
- Sample random batches for training (no prioritization yet)
- Enforce capacity limit: drop oldest when full

**Why separate buffer?**
- Decorrelates experience order (improves learning stability)
- Allows batch training independent of experience generation rate

### Worker (`worker.rs`)
Async training loop pulling batches and updating model.

**Flow:**
```
loop {
  sleep(interval_secs)
  if buffer has ≥ batch_size experiences:
    sample batch → train_on_batch() → model.reinforce()
    save model checkpoint
    update batch_history
    check for convergence/divergence
}
```

**Key Decision:**
- Worker runs in background indefinitely
- Application decides when to stop (counter reaches total_batches in config)

### Experience (`experience.rs`)
Data struct for single interaction:
```rust
pub struct Experience {
    pub features: Vec<f32>,      // 8 dimensions from God
    pub action: u8,               // 0/1/2 (hold/buy/sell)
    pub latency_ms: f32,
    pub reward: f32,              // Clamped to [-1.0, 1.0]
    pub success: bool,            // Reward > 0?
    pub timestamp_ms: u128,
}
```

### Configuration (`configurations.rs`)
Loads hyperparameters from config.toml:

```toml
interval_secs = 0.01           # Worker training loop frequency
batch_size = 128               # Experiences per training iteration
total_batches = 113000         # Training termination point
input_number = 8               # Feature vector dimension
output_number = 3              # Action space (hold/buy/sell)
hidden_layers = 16             # ReLU layer neurons
reply_capacity = 4096          # Max buffer size
model_name = "model-128.json"  # Checkpoint file name
log_interval = 64              # Logging frequency
mode = "Training"              # or "Infer"
```

**Critical contract:**
- `input_number` MUST match feature vector size (God's 8 features in Dealer)
- `output_number` MUST match action space (typically 3)
- Mismatch → runtime panic during model initialization

### Inference Engine (`infer_action.rs`)
Deterministic action selection using softmax + epsilon-greedy.

**Flow:**
```
1. Forward pass: logits = model.forward(features)
2. Softmax with temperature: probs = softmax(logits / temperature)
3. Epsilon-greedy: 
   - Random exploration (eps): pick random action
   - Exploitation (1-eps): pick action with highest probability
4. Return: (action_idx, logits, probabilities)
```

**Why epsilon-greedy?**
- Pure greedy → gets stuck in local optima
- Pure random → never exploits learned knowledge
- Epsilon balances both (dynamically adjusted by Node)

---

## Critical Workflows

### Initialization (Application's responsibility)

```rust
// 1. Load config
let config = load_config()?;
aion_rlt::initialize(config);

// 2. Create Node (wraps model + buffer + worker)
let node = Node::run(
    RunningMode::Training,
    "model-128.json.0"  // checkpoint name
).await;

// 3. Worker starts in background automatically
```

### Training Loop (Dealer's responsibility)

```rust
for candle in candles {
    // Get features from God processor
    let features = god.process_candle(candle);
    
    // Infer action (deterministic action selection)
    let (action, logits, probs) = infer_action(&node, &features);
    
    // Calculate reward (Dealer's logic)
    let reward = calculate_reward(action, candle_index);
    
    // Push experience to buffer
    let experience = Experience {
        features,
        action: action as u8,
        reward,
        success: reward > 0.0,
        ..default()
    };
    node.buffer.write().unwrap().push(experience);
    
    // Worker trains in background automatically
    // No explicit training call needed!
}
```

### Inference Mode (Same API, different config)

```rust
// Set mode = "Infer" in config.toml
let node = Node::run(RunningMode::Infer, "model-128.json").await;

for candle in live_candles {
    let features = god.process_candle(candle);
    let (action, _logits, probs) = infer_action(&node, &features);
    
    // Use action, no training happens
    println!("Predicted action: {}, confidence: {}", action, probs[action]);
}
```

---

## Design Patterns & Conventions

### Arc<RwLock<T>> Everywhere
Why shared mutable state via read-write locks?
- Multiple threads access same model/buffer
- Reader threads (inference) don't block each other
- Writer thread (training) exclusive access when updating

**Common pattern:**
```rust
let model = node.model.read().unwrap();   // Shared read
let logits = model.forward(&features);
drop(model);                              // Release lock

let mut model = node.model.write().unwrap();  // Exclusive write
model.reinforce(&batch);
```

### Temperature & Epsilon Tuning
Node dynamically adjusts exploration via RwLock:
```rust
// Application can modify at runtime
*node.temperature.write().unwrap() = 0.5;  // Lower = more greedy
*node.epsilon.write().unwrap() = 0.1;      // Lower = less random
```

### Batch Training Decoupling
- Experience generation (inference) independent from training
- Worker samples when buffer has enough data
- Tolerates rate mismatches (fast inference, slower training)

### Model Convergence Tracking
Node tracks:
- `batch_history`: HashMap<batch_num, loss>
- `diverging` flag: loss increasing → instability
- `success_rate_mean`: % of profitable actions
- Use for diagnostics, not automated control (yet)

---

## Common Pitfalls

- **Feature dimension mismatch**: `input_number ≠ feature_vector.len()` → panic
- **Reward range**: Model expects [-1.0, 1.0] clamped rewards (see Worker for clamp logic)
- **Buffer underflow**: Training starts before buffer has batch_size experiences (gracefully skipped in Worker)
- **Model file conflicts**: Multiple agents writing to same checkpoint name → corruption (use agent ID suffix)
- **Temperature too low**: Temperature = 0.1 → nearly deterministic (less exploration)
- **Epsilon too high**: Epsilon = 0.9 → 90% random actions (learns slowly)

---

## Dependencies & External Integration

- **ndarray + ndarray-rand**: Numerical arrays, matrix operations, random initialization
- **tokio**: Async runtime for Worker background task
- **serde/serde_json**: Model checkpoint serialization
- **aion-math**: External utility crate (ContinuousMath, Math traits)
- **log + env_logger**: Structured logging (initialized by application)

---

## File Organization

```
AION-RLT/src/
├── lib.rs                 # Module declarations, CONFIG singleton, initialize()
├── node.rs                # Node struct, RunningMode, training state
├── model.rs               # Neural network, forward pass, REINFORCE
├── worker.rs              # Async training loop orchestration
├── experience.rs          # Experience struct definition
├── infer_action.rs        # Action sampling (softmax + epsilon-greedy)
├── reply_buffer.rs        # Replay buffer (experience storage + sampling)
├── configurations.rs      # Configurations struct, config loading
└── footstep/              # (Internal diagnostics module)
    └── mod.rs
```

---

## Extending AION-RLT

### Adding New Training Algorithms
1. Modify `Model::reinforce()` to implement new gradient logic
2. Update loss calculation
3. Document reward expectations (magnitude, range)

### Supporting Different Input/Output Sizes
- Already supported: `input_number`, `output_number` in config
- Model initializes with these dimensions at startup
- No code changes needed, just config tuning

### Custom Logging/Metrics
- Node exposes `batch_history`, `success_rate_mean`
- Application reads these via RwLock and exports to Prometheus/Grafana
- No changes to RLT needed

### Moving to Production
1. Persist Node state: model + buffer to disk
2. Load on startup: `Node::run()` with existing checkpoint
3. Scale: Run multiple Node instances per agent (already supports)
4. Monitor: Track diverging flag, success_rate_mean

---

## Integration Points with Applications

**Contract: What AION-RLT Provides**
- Takes: Features (Vec<f32>), Actions (u8), Rewards (f32)
- Returns: Action probabilities, model predictions
- Handles: Training, model persistence, exploration/exploitation tradeoff

**Contract: What Application Must Provide**
- Feature extraction (God processor in Dealer)
- Reward calculation (domain-specific, trading logic in Dealer)
- Experience formatting (Experience struct)
- Config tuning (hyperparameters in config.toml)

