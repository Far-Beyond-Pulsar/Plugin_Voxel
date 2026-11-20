#![allow(dead_code)]
//! Headless GPU helpers for voxel planet tests.

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// The test process's device, as the engine requests one (none without an
/// adapter): made once and shared, so tests build their worlds on it in
/// parallel. A device each, created and dropped on many threads at once,
/// froze the whole test process now and then (the harness stopped printing
/// too: 3 of 11 in-process runs of tests/chunks.rs, 2026-10-19).
pub fn gpu() -> Option<&'static Gpu> {
    static GPU: std::sync::OnceLock<Option<Gpu>> = std::sync::OnceLock::new();
    GPU.get_or_init(|| {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            // 64-bit integers as the engine's devices request them
            // (`helio::required_wgpu_features`): the hardware wide products.
            required_features: adapter.features()
                & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_INT64),
            required_limits: wgpu::Limits {
                max_buffer_size: limits.max_buffer_size.min(u32::MAX as u64),
                ..limits
            },
            ..Default::default()
        }))
        .ok()?;
        Some(Gpu { device, queue })
    })
    .as_ref()
}

pub fn read_buffer(gpu: &Gpu, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let data = staging.slice(..).get_mapped_range().unwrap().to_vec();
    data
}

/// The terrain's tracer number in the tests' ray streams.
pub const VOXEL_TRACER: u32 = 1;

/// A ray from world point `from` along unit `dir` up to `t_max`, in the
/// space of frame origin `frame` (every caster, a footprint `spread` wide a
/// metre out).
pub fn scene_ray(
    frame: glam::DVec3,
    from: glam::DVec3,
    dir: glam::DVec3,
    t_max: f32,
    spread: f32,
) -> helio_frame_api::rays::stream::SceneRay {
    helio_frame_api::rays::stream::SceneRay::new(
        (from - frame).as_vec3().to_array(),
        t_max,
        dir.as_vec3().to_array(),
        helio_frame_api::rays::stream::SceneRay::cone(
            0.0,
            spread,
            helio_frame_api::rays::stream::MASK_ALL,
        ),
    )
}

/// Stream `rays` (of `class`, all but the last `past` rows in its header,
/// relative to frame origin `frame`) traced by the terrain through the cut
/// `world` drew last, its hits' rows `before` (none: cleared) beforehand:
/// its hits and surfaces after (none for an occlusion stream), the
/// surfaces at `SurfaceDetail::Diffuse` (GI's probes').
pub fn trace_stream(
    gpu: &Gpu,
    world: &helio_pass_voxel_planet::chunks::ChunkWorld,
    class: helio_frame_api::rays::stream::RayClass,
    frame: glam::DVec3,
    rays: &[helio_frame_api::rays::stream::SceneRay],
    before: Option<&[helio_frame_api::rays::stream::SceneHit]>,
    past: usize,
) -> (
    Vec<helio_frame_api::rays::stream::SceneHit>,
    Vec<helio_frame_api::rays::stream::SceneSurface>,
) {
    trace_stream_in_runs(gpu, world, class, frame, rays, before, past, 1)
}

/// [`trace_stream`] of a nearest-hit stream asking its hits' surfaces in
/// full (`SurfaceDetail::Full`).
pub fn trace_stream_full(
    gpu: &Gpu,
    world: &helio_pass_voxel_planet::chunks::ChunkWorld,
    frame: glam::DVec3,
    rays: &[helio_frame_api::rays::stream::SceneRay],
) -> (
    Vec<helio_frame_api::rays::stream::SceneHit>,
    Vec<helio_frame_api::rays::stream::SceneSurface>,
) {
    trace_stream_detailed(
        gpu,
        world,
        helio_frame_api::rays::stream::RayClass::Nearest,
        helio_frame_api::rays::stream::SurfaceDetail::Full,
        frame,
        rays,
        None,
        0,
        1,
    )
}

/// [`trace_stream`] of a stream whose rays come in runs of `run` from one
/// origin (`Stream::run`).
#[allow(clippy::too_many_arguments)]
pub fn trace_stream_in_runs(
    gpu: &Gpu,
    world: &helio_pass_voxel_planet::chunks::ChunkWorld,
    class: helio_frame_api::rays::stream::RayClass,
    frame: glam::DVec3,
    rays: &[helio_frame_api::rays::stream::SceneRay],
    before: Option<&[helio_frame_api::rays::stream::SceneHit]>,
    past: usize,
    run: u32,
) -> (
    Vec<helio_frame_api::rays::stream::SceneHit>,
    Vec<helio_frame_api::rays::stream::SceneSurface>,
) {
    trace_stream_detailed(
        gpu,
        world,
        class,
        helio_frame_api::rays::stream::SurfaceDetail::Diffuse,
        frame,
        rays,
        before,
        past,
        run,
    )
}

/// [`trace_stream_in_runs`] with the surfaces at `detail`.
#[allow(clippy::too_many_arguments)]
pub fn trace_stream_detailed(
    gpu: &Gpu,
    world: &helio_pass_voxel_planet::chunks::ChunkWorld,
    class: helio_frame_api::rays::stream::RayClass,
    detail: helio_frame_api::rays::stream::SurfaceDetail,
    frame: glam::DVec3,
    rays: &[helio_frame_api::rays::stream::SceneRay],
    before: Option<&[helio_frame_api::rays::stream::SceneHit]>,
    past: usize,
    run: u32,
) -> (
    Vec<helio_frame_api::rays::stream::SceneHit>,
    Vec<helio_frame_api::rays::stream::SceneSurface>,
) {
    use helio_frame_api::rays::stream::{SceneHit, SceneSurface, Stream, StreamBuffers, TracerId};
    use helio_pass_voxel_planet::chunks::rays::{StreamBindings, StreamTracer};
    let stream = Stream::new("voxel_test", class, rays.len() as u32);
    let buffers = StreamBuffers::new(&gpu.device, &stream);
    gpu.queue
        .write_buffer(&buffers.rays, 0, bytemuck::cast_slice(rays));
    buffers.set_count(&gpu.queue, (rays.len() - past) as u32);
    let cleared = vec![SceneHit::default(); rays.len()];
    gpu.queue.write_buffer(
        &buffers.hits,
        0,
        bytemuck::cast_slice(before.unwrap_or(&cleared)),
    );
    let tracer = StreamTracer::new(&gpu.device, class, TracerId(VOXEL_TRACER))
        .with_run(run)
        .with_detail(detail);
    tracer.write_frame(frame, |buffer, bytes| {
        gpu.queue.write_buffer(buffer, 0, bytes)
    });
    let cut = world.ray_cut().expect("a cut drawn for ray streams");
    assert!(cut.ready(), "the ray kernels compiled");
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut recorder = helio_core::CommandRecorder::from_encoder(&mut encoder);
        let bindings = StreamBindings {
            header: &buffers.header,
            rays: &buffers.rays,
            hits: &buffers.hits,
            surfaces: buffers.surfaces.as_ref(),
        };
        tracer.trace(&gpu.device, &mut recorder, &cut, bindings, None);
        tracer.surface(&gpu.device, &mut recorder, &cut, bindings, None);
    }
    gpu.queue.submit([encoder.finish()]);
    let hits: Vec<SceneHit> =
        bytemuck::cast_slice(&read_buffer(gpu, &buffers.hits, buffers.hits.size())).to_vec();
    let surfaces: Vec<SceneSurface> = buffers
        .surfaces
        .as_ref()
        .map(|s| bytemuck::cast_slice(&read_buffer(gpu, s, s.size())).to_vec())
        .unwrap_or_default();
    (hits, surfaces)
}
