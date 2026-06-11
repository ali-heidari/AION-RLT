// weight-update.wgsl — SGD step: w -= lr * grad

struct Params {
    len: u32,
    lr: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.len) {
        return;
    }
    w[i] = w[i] - p.lr * grad[i];
}
