// reinforce.wgsl — REINFORCE gradient w.r.t. logits:
// grad_logits[i, a_i] = -reward_i * (1 - probs[i, a_i]), 0 elsewhere.

struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> actions: array<u32>;
@group(0) @binding(3) var<storage, read> rewards: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_logits: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.rows) {
        return;
    }

    let a = actions[i];
    for (var j: u32 = 0u; j < p.cols; j++) {
        var g: f32 = 0.0;
        if (j == a) {
            g = -rewards[i] * (1.0 - probs[i * p.cols + j]);
        }
        grad_logits[i * p.cols + j] = g;
    }
}
