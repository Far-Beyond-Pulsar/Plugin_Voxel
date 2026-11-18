//! The voxel terrain as a tracer of the frame's ray streams
//! ([`helio_frame_api::rays::stream`]): any consumer's rays (GI's probes,
//! shadow rays from any kind's hits, reflections, a radiance cache, a path
//! tracer) walk every volume's drawn cut as the camera's do, from anywhere,
//! with no CPU work per ray and no hardware ray tracing.
//!
//! - **Where the cut comes from.** Each camera draw ([`super::ChunkWorld`])
//!   writes, in command order with the draw itself, each drawn volume's ray
//!   parameters (its draw parameters for the camera's eye) and that eye
//!   ([`RayCut`]), and publishes the cut into the graph's [`RayLink`]. A
//!   tracer pass recorded before this frame's draw walks the last frame's
//!   cut, one after it this frame's: always one whole cut, the chunk storage
//!   and table as the draw that wrote the parameters left them (the update
//!   after it changes them in command order too, and a pool's growth makes
//!   new buffers, leaving the old ones to the passes that bound them).
//! - **Where a ray starts.** A ray carries a frame-space origin only. On the
//!   GPU it comes into each volume's frame by the frame's origin less the
//!   draw's eye (f64 parts), enters the volume where it starts or reaches it
//!   and is placed in the frame of a drawn chunk near its origin
//!   (`enter_volume`, `placing_chunk` in chunk_draw.wgsl: the anchor near the
//!   eye, else the drawn node a sphere's exact levels find, then the chunk it
//!   starts in), so it walks from its exact cell.
//! - **Two passes a stream.** The trace keeps the walk's kernel small: for
//!   a nearest-hit stream a volume's entry (`ray_enter_nearest`) writes each
//!   ray's start (a 64-byte row in the tracer's own buffer), then the walk
//!   (`ray_nearest`) reads it, so the walk keeps the old GI probe walk's registers (an
//!   occlusion stream's short walks keep their entry: `ray_occlusion`);
//!   the surfaces (`ray_surface`) are written once every tracer traced, only
//!   for the hits that stayed the terrain's. Two dispatches a volume over
//!   the stream's indirect groups.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use glam::DVec3;
use helio_core::compile::Compiling;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_frame_api::rays::stream::{Answers, RayClass, Stream, TracerId};

use super::volumes::DrawLayouts;
use super::{entry, split};

/// A form's ray stream kernels (chunk_draw.wgsl 8): each volume's ray
/// parameters in group 0 (the draw layout), the stream in group 1.
pub(super) struct RayPipelines {
    /// A nearest-hit stream's entry (the rays' starts) and walk; an
    /// occlusion stream's both in one.
    enter_nearest: wgpu::ComputePipeline,
    nearest: wgpu::ComputePipeline,
    occlusion: wgpu::ComputePipeline,
    surface: wgpu::ComputePipeline,
}

impl RayPipelines {
    /// The kernels of a form's draw source `module`: built once a stream
    /// is traced ([`super::ChunkWorld::want_rays`]), as walks are long
    /// kernels to compile.
    pub fn new(device: &wgpu::Device, layouts: &DrawLayouts, module: &wgpu::ShaderModule) -> Self {
        let compute = |label: &str, second: &wgpu::BindGroupLayout, entry: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layouts.draw), Some(second)],
                immediate_size: 0,
            });
            helio_core::shader::create_compute_pipeline(
                device,
                wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };
        Self {
            enter_nearest: compute(
                "voxel nearest ray starts",
                &layouts.rays,
                "ray_enter_nearest",
            ),
            nearest: compute("voxel nearest rays", &layouts.rays, "ray_nearest"),
            occlusion: compute("voxel occlusion rays", &layouts.rays, "ray_occlusion"),
            surface: compute("voxel ray surfaces", &layouts.ray_surfaces, "ray_surface"),
        }
    }
}

/// A volume of a [`RayCut`]: its form and draw bind group over its ray
/// parameters (its index is in them).
#[derive(Clone)]
pub(super) struct RayVolume {
    pub plane: bool,
    pub group: wgpu::BindGroup,
}

/// The cut ray streams walk: the volumes the last camera draw had drawn
/// chunks of, each with its ray parameters' bind group, the eye those are
/// relative to, and the kernels of each form (none until compiled).
#[derive(Clone)]
pub struct RayCut {
    pub(super) volumes: Vec<RayVolume>,
    pub(super) pipelines: [Option<Arc<Compiling<RayPipelines>>>; 2],
    /// The stream's layouts: the traces' and the surfaces'.
    pub(super) layouts: [wgpu::BindGroupLayout; 2],
    pub(super) eye: wgpu::Buffer,
}

impl RayCut {
    /// The volumes a stream's rays are traced against.
    pub fn volumes(&self) -> usize {
        self.volumes.len()
    }

    /// Whether every volume's form has its kernels.
    pub fn ready(&self) -> bool {
        self.volumes.iter().all(|v| self.kernels(v.plane).is_some())
    }

    fn kernels(&self, plane: bool) -> Option<&RayPipelines> {
        self.pipelines[usize::from(plane)]
            .as_ref()
            .and_then(|c| c.ready())
    }
}

/// Where a graph's chunk pass publishes the cut and the terrain's tracer
/// passes find it: shared by the graph's build ([`SharedRayLink`]), so
/// passes built before the chunk pass's node (a consumer before the
/// G-buffer phase) find the same link.
#[derive(Default)]
pub struct RayLink {
    cut: Mutex<Option<Arc<RayCut>>>,
    wanted: AtomicBool,
    /// The frame holds volumes (the chunk pass, each frame), and a cut
    /// with drawn chunks of them was published since they came.
    volumes: AtomicBool,
    drawn: AtomicBool,
}

impl RayLink {
    /// The cut drawn last (none: nothing drawn yet).
    pub fn cut(&self) -> Option<Arc<RayCut>> {
        self.cut
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The cut the world drew (its chunk pass, after each draw).
    pub fn publish(&self, cut: Option<Arc<RayCut>>) {
        if cut.as_ref().is_some_and(|c| c.volumes() > 0) {
            self.drawn.store(true, Ordering::Relaxed);
        }
        *self
            .cut
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = cut;
    }

    /// A stream of the graph is traced by the terrain: the world keeps a cut
    /// for it ([`super::ChunkWorld::want_rays`]).
    pub fn want(&self) {
        self.wanted.store(true, Ordering::Relaxed);
    }

    pub fn wanted(&self) -> bool {
        self.wanted.load(Ordering::Relaxed)
    }

    /// Whether this frame holds volumes (the chunk pass, as it takes the
    /// frame's).
    pub fn set_volumes(&self, held: bool) {
        self.volumes.store(held, Ordering::Relaxed);
        if !held {
            self.drawn.store(false, Ordering::Relaxed);
        }
    }

    /// Whether the cut answers rays for every volume the frame holds: none
    /// held, or chunks of them drawn once and the cut's kernels compiled.
    /// Short (`stream::Answers::short`) until the volumes' first chunks are
    /// drawn and while their ray kernels compile. Later frames that draw
    /// none of them (an eye looking away) and chunks still generating are
    /// what the cut holds of the volumes, no shortfall.
    pub fn complete(&self) -> bool {
        !self.volumes.load(Ordering::Relaxed)
            || (self.drawn.load(Ordering::Relaxed) && self.cut().is_some_and(|cut| cut.ready()))
    }
}

/// The graph's [`RayLink`] (`BuildEnv::shared`), made by whichever of the
/// chunk pass and a tracer pass builds first.
#[derive(Default)]
pub struct SharedRayLink(pub Arc<RayLink>);

/// What a stream's passes read and write.
#[derive(Clone, Copy)]
pub struct StreamBindings<'a> {
    pub header: &'a wgpu::Buffer,
    pub rays: &'a wgpu::Buffer,
    pub hits: &'a wgpu::Buffer,
    /// A nearest-hit stream's; none for an occlusion stream.
    pub surfaces: Option<&'a wgpu::Buffer>,
}

/// `RayFrame` of chunk_draw.wgsl: the frame's origin (world, high and low
/// parts) and the tracer's number.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuRayFrame {
    origin_hi: [f32; 4],
    origin_lo: [f32; 4],
    tracer: [u32; 4],
}

/// One stream traced by the terrain: its class, the terrain's tracer
/// number, the frame uniform its passes read and their bind groups (made
/// when the stream's buffers or the cut's change).
pub struct StreamTracer {
    class: RayClass,
    tracer: TracerId,
    frame: wgpu::Buffer,
    /// The rays' starts in the volume being traced (`RayStart`,
    /// [`START_BYTES`] a ray), as large as the stream's rays need.
    starts: Mutex<Option<wgpu::Buffer>>,
    cache: helio_core::gpu_cache::GpuCache,
}

/// `RayStart`'s size (chunk_draw.wgsl 8).
const START_BYTES: u64 = 64;

impl StreamTracer {
    pub fn new(device: &wgpu::Device, class: RayClass, tracer: TracerId) -> Self {
        Self {
            class,
            tracer,
            frame: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel ray frame"),
                size: std::mem::size_of::<GpuRayFrame>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            starts: Mutex::new(None),
            cache: helio_core::gpu_cache::GpuCache::new(),
        }
    }

    /// The starts' buffer for `rays` (made, or remade larger, when short).
    fn starts(&self, device: &wgpu::Device, rays: &wgpu::Buffer) -> wgpu::Buffer {
        let rows = (rays.size() / helio_frame_api::rays::stream::SceneRay::BYTES).max(1);
        let mut starts = self
            .starts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if starts
            .as_ref()
            .is_none_or(|b| b.size() < rows * START_BYTES)
        {
            *starts = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxel ray starts"),
                size: rows * START_BYTES,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
        }
        starts.as_ref().unwrap().clone()
    }

    /// The frame's origin `origin` (world) the stream's rays are relative
    /// to, written by `write` (the graph's `write_buffer`, a queue's)
    /// before the passes run.
    pub fn write_frame(&self, origin: DVec3, write: impl FnOnce(&wgpu::Buffer, &[u8])) {
        let (origin_hi, origin_lo) = split(origin);
        let frame = GpuRayFrame {
            origin_hi,
            origin_lo,
            tracer: [self.tracer.0, 0, 0, 0],
        };
        write(&self.frame, bytemuck::bytes_of(&frame));
    }

    fn group(
        &self,
        device: &wgpu::Device,
        cut: &RayCut,
        stream: StreamBindings<'_>,
        surfaces: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        let starts = surfaces.is_none().then(|| self.starts(device, stream.rays));
        let mut entries = vec![
            entry(67, stream.header),
            entry(68, stream.rays),
            entry(69, stream.hits),
            entry(71, &self.frame),
            entry(72, &cut.eye),
        ];
        entries.extend(surfaces.map(|s| entry(70, s)));
        entries.extend(starts.as_ref().map(|s| entry(73, s)));
        self.cache.bind_group(
            device,
            &wgpu::BindGroupDescriptor {
                label: Some("voxel rays"),
                layout: &cut.layouts[usize::from(surfaces.is_some())],
                entries: &entries,
            },
        )
    }

    /// The trace of `stream` through every volume of `cut`: a dispatch a
    /// volume over the stream's indirect groups (volumes whose kernels are
    /// still compiling trace nothing), its pass timed as `timing` says.
    pub fn trace(
        &self,
        device: &wgpu::Device,
        encoder: &mut helio_core::CommandRecorder<'_>,
        cut: &RayCut,
        stream: StreamBindings<'_>,
        timing: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let group = self.group(device, cut, stream, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel rays"),
            timestamp_writes: timing,
        });
        pass.set_bind_group(1, &group, &[]);
        for v in &cut.volumes {
            let Some(kernels) = cut.kernels(v.plane) else {
                continue;
            };
            pass.set_bind_group(0, &v.group, &[]);
            if self.class == RayClass::Nearest {
                pass.set_pipeline(&kernels.enter_nearest);
                pass.dispatch_workgroups_indirect(stream.header, 0);
            }
            pass.set_pipeline(match self.class {
                RayClass::Nearest => &kernels.nearest,
                RayClass::Occlusion => &kernels.occlusion,
            });
            pass.dispatch_workgroups_indirect(stream.header, 0);
        }
    }

    /// The surfaces of `stream`'s hits that are the terrain's, once every
    /// tracer traced it (a nearest-hit stream's; none for an occlusion
    /// stream).
    pub fn surface(
        &self,
        device: &wgpu::Device,
        encoder: &mut helio_core::CommandRecorder<'_>,
        cut: &RayCut,
        stream: StreamBindings<'_>,
        timing: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let (RayClass::Nearest, Some(surfaces)) = (self.class, stream.surfaces) else {
            return;
        };
        let group = self.group(device, cut, stream, Some(surfaces));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel ray surfaces"),
            timestamp_writes: timing,
        });
        pass.set_bind_group(1, &group, &[]);
        for v in &cut.volumes {
            let Some(kernels) = cut.kernels(v.plane) else {
                continue;
            };
            pass.set_pipeline(&kernels.surface);
            pass.set_bind_group(0, &v.group, &[]);
            pass.dispatch_workgroups_indirect(stream.header, 0);
        }
    }
}

/// A stream's terrain pass in the consumer's node: its trace, or (`surface`)
/// its surfaces, through the cut its graph's chunk pass drew last.
pub struct VoxelRayPass {
    link: Arc<RayLink>,
    /// The stream's answers this frame: short while the cut does not
    /// answer for every volume ([`RayLink::complete`]).
    answers: Arc<Answers>,
    stream: Stream,
    tracer: StreamTracer,
    surface: bool,
    name: &'static str,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl VoxelRayPass {
    /// `stream`'s trace (or `surface`s) as tracer `id` of the graph, through
    /// the cut published to `link`, telling the stream's `answers` when the
    /// cut is short.
    pub fn new(
        device: &wgpu::Device,
        link: Arc<RayLink>,
        answers: Arc<Answers>,
        stream: Stream,
        id: TracerId,
        surface: bool,
    ) -> Self {
        link.want();
        let (name, writes) = if surface {
            (
                format!("VoxelRaySurfaces {}", stream.name),
                stream.surfaces.into_iter().collect::<Vec<_>>(),
            )
        } else {
            (format!("VoxelRays {}", stream.name), vec![stream.hits])
        };
        Self {
            link,
            answers,
            tracer: StreamTracer::new(device, stream.class, id),
            surface,
            name: helio_frame_api::static_name(&name),
            reads: helio_frame_api::static_names(&[stream.header, stream.rays, stream.hits]),
            writes: helio_frame_api::static_names(&writes),
            stream,
        }
    }
}

impl RenderPass for VoxelRayPass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.tracer.write_frame(ctx.world_origin, |buffer, bytes| {
            ctx.write_buffer(buffer, 0, bytes)
        });
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.surface && !self.link.complete() {
            self.answers.short();
        }
        let Some(cut) = self.link.cut() else {
            return Ok(());
        };
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let (Some(header), Some(rays), Some(hits)) = (
            binding(self.stream.header),
            binding(self.stream.rays),
            binding(self.stream.hits),
        ) else {
            return Ok(());
        };
        let stream = StreamBindings {
            header,
            rays,
            hits,
            surfaces: self.stream.surfaces.and_then(binding),
        };
        let mut cmds = ctx.graphics_cmds();
        if self.surface {
            self.tracer
                .surface(ctx.device, &mut cmds, &cut, stream, None);
        } else {
            self.tracer.trace(ctx.device, &mut cmds, &cut, stream, None);
        }
        Ok(())
    }
}
