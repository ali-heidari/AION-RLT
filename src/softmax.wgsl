// softmax.wgsl — row-wise softmax with temperature, numerically stabilized.

struct Params {
    rows: u32,
    cols: u32,
    temperature: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row >= p.rows) {
        return;
    }

    var max_v: f32 = -3.4e38;
    for (var i: u32 = 0u; i < p.cols; i++) {
        let v = logits[row * p.cols + i];
        if (v > max_v) {
            max_v = v;
        }
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < p.cols; i++) {
        let e = exp((logits[row * p.cols + i] - max_v) / p.temperature);
        probs[row * p.cols + i] = e;
        sum += e;
    }

    for (var i: u32 = 0u; i < p.cols; i++) {
        probs[row * p.cols + i] /= sum;
    }
}
