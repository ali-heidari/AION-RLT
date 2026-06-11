// grad-hidden.wgsl — gradient through the hidden layer with ReLU mask:
// grad_h[i, k] = (h[i, k] > 0) ? Σ_j grad_logits[i, j] * w2[k, j] : 0
// (h is post-ReLU, so h > 0 is equivalent to h_pre > 0.)

struct Params {
    m: u32, // batch
    k: u32, // hidden size
    n: u32, // output size
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> grad_logits: array<f32>;
@group(0) @binding(2) var<storage, read> w2: array<f32>;
@group(0) @binding(3) var<storage, read> h: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_h: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;  // batch row
    let kk = id.y; // hidden unit
    if (i >= p.m || kk >= p.k) {
        return;
    }

    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < p.n; j++) {
        sum += grad_logits[i * p.n + j] * w2[kk * p.n + j];
    }

    if (h[i * p.k + kk] <= 0.0) {
        sum = 0.0;
    }
    grad_h[i * p.k + kk] = sum;
}
