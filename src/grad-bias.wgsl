// grad-bias.wgsl — bias gradient: column-wise sum over the batch.
// grad_b[j] = Σ_i grad[i, j]

struct Params {
    m: u32, // batch (rows)
    n: u32, // cols
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let j = id.x;
    if (j >= p.n) {
        return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < p.m; i++) {
        sum += grad[i * p.n + j];
    }
    out[j] = sum;
}
