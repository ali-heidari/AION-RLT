@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> actions: array<u32>;
@group(0) @binding(2) var<storage, read> rewards: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_logits: array<f32>;

const OUT: u32 = 3;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let a = actions[i];

    for (var j: u32 = 0; j < OUT; j++) {
        var g: f32 = 0.0;

        if (j == a) {
            g = -rewards[i] * (1.0 - probs[i * OUT + j]);
        }

        grad_logits[i * OUT + j] = g;
    }
}