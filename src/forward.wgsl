// forward.wgsl
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w1: array<f32>;
@group(0) @binding(2) var<storage, read> b1: array<f32>;
@group(0) @binding(3) var<storage, read> w2: array<f32>;
@group(0) @binding(4) var<storage, read> b2: array<f32>;

@group(0) @binding(5) var<storage, read_write> h: array<f32>;
@group(0) @binding(6) var<storage, read_write> logits: array<f32>;

const IN: u32 = 64;
const H: u32 = 128;
const OUT: u32 = 3;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let j = id.y;

    // --- layer 1 ---
    if (j < H) {
        var sum: f32 = 0.0;
        for (var k: u32 = 0; k < IN; k++) {
            sum += x[i * IN + k] * w1[k * H + j];
        }
        sum += b1[j];

        // relu
        if (sum < 0.0) { sum = 0.0; }

        h[i * H + j] = sum;
    }

    // --- layer 2 ---
    if (j < OUT) {
        var sum2: f32 = 0.0;
        for (var k: u32 = 0; k < H; k++) {
            sum2 += h[i * H + k] * w2[k * OUT + j];
        }
        sum2 += b2[j];

        logits[i * OUT + j] = sum2;
    }
}