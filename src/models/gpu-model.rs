struct GpuModel {
    device: wgpu::Device,
    queue: wgpu::Queue,

    w1: Buffer,
    b1: Buffer,
    w2: Buffer,
    b2: Buffer,

    forward_pipeline: ComputePipeline,
    softmax_pipeline: ComputePipeline,
    reinforce_pipeline: ComputePipeline,
}