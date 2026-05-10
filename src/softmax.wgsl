@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;

const OUT: u32 = 3;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;

    var max: f32 = -1e9;
    for (var i: u32 = 0; i < OUT; i++) {
        let v = logits[row * OUT + i];
        if (v > max) { max = v; }
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < OUT; i++) {
        let e = exp(logits[row * OUT + i] - max);
        probs[row * OUT + i] = e;
        sum += e;
    }

    for (var i: u32 = 0; i < OUT; i++) {
        probs[row * OUT + i] /= sum;
    }
}