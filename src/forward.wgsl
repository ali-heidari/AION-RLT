// forward.wgsl — generic linear layer: y = x·W + b, optional ReLU.
// Dispatched once per layer (layer 1 with relu = 1, layer 2 with relu = 0)
// so layer 2 never races against layer 1 writes.

struct Params {
    rows: u32,    // batch size
    in_dim: u32,  // input columns
    out_dim: u32, // output columns
    relu: u32,    // 1 = apply ReLU
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x; // row
    let j = id.y; // output column
    if (i >= p.rows || j >= p.out_dim) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < p.in_dim; k++) {
        sum += x[i * p.in_dim + k] * w[k * p.out_dim + j];
    }
    sum += b[j];

    if (p.relu == 1u && sum < 0.0) {
        sum = 0.0;
    }

    y[i * p.out_dim + j] = sum;
}
