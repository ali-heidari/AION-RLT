pub fn compute_reward(features: &Vec<f32>) -> f32 {
    let cpu = features[0].clamp(0.0, 1.0);
    let mem = features[1].clamp(0.0, 1.0);
    let swap = features[2].clamp(0.0, 1.0);
    let disk = features[3].clamp(0.0, 1.0);
    let throughput = features[4].clamp(0.0, 1.0);
    let latency = features[5].clamp(0.0, 1.0);

    // Nonlinear penalty: heavy penalty when close to full utilization
    let load_penalty = 0.7 * cpu + 0.5 * mem;
    let io_penalty = 0.05 * swap + 0.1 * disk;

    // Performance term — prefer high throughput & low latency
    let perf_score = 1.5 * throughput * (1.0 - latency * 0.2).clamp(0.0, 1.0);

    // Combine into base reward
    let reward = perf_score - (load_penalty + io_penalty);

    reward.clamp(-1.0, 1.0)
}

pub fn compute_reward_with_success(features: &Vec<f32>, action: u8) -> (f32, bool) {
    let raw_reward = compute_reward(&features);

    let scaled_reward = ((raw_reward + 1.0) / (2.0)).clamp(0.0, 1.0);

    let state=    // Discretize the continuous value into states
    if scaled_reward < 0.3 {
        0 // High pressure (bad)
    } else if scaled_reward < 0.6 {
        1 // Normal
    } else {
        2 // Low pressure (good)
    };

    let target_center = [0.15, 0.45, 0.8][action as usize];
    let mut reward = 1.0 - f32::abs(target_center - scaled_reward);
    reward = (reward * 2.0) - 1.0; // scale to [-1, 1]
                                   // Penalize totally wrong actions (like doing opposite)

    if (action).abs_diff(state) == 1 {
        reward -= 0.5;
    }

    if (state == 0 && action == 2) || (state == 2 && action == 0) {
        reward = -1.0;
    }

    (reward.clamp(-1.0, 1.0), reward > 0.0)
}
