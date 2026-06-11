// grad-weights.wgsl — weight gradient as C = Aᵀ·B.
// A: (m × k) activations, B: (m × n) upstream gradient, C: (k × n) weight grad.
// Used for grad_w2 = hᵀ·grad_logits and grad_w1 = xᵀ·grad_h.

struct Params {
    m: u32, // batch (summed dimension)
    k: u32, // rows of C
    n: u32, // cols of C
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x; // row of C (column of A)
    let cc = id.y; // col of C (column of B)
    if (r >= p.k || cc >= p.n) {
        return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < p.m; i++) {
        sum += a[i * p.k + r] * b[i * p.n + cc];
    }
    c[r * p.n + cc] = sum;
}
