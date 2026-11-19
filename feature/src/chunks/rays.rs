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
//! - **Entry apart from the walk.** The trace keeps the walk's kernel
//!   small: for a nearest-hit stream a volume's entry writes each ray's
//!   start (`RayStart`, a 64-byte row in the tracer's own buffer), then the
//!   walk (`ray_nearest`) reads it. A stream whose rays come in runs from
//!   one origin (`Stream::run`: GI's probes, 64 rays each) places each run's
//!   origin once (`ray_enter_runs`, a thread a run) and its rays read that
//!   row; rays whose origin is outside the volume, or a stream of single
//!   rays, enter one by one (`ray_enter_nearest`). An occlusion stream's
//!   short walks keep their entry in their kernel (`ray_occlusion`). The
//!   surfaces (`ray_surface`) are written once
//!   every tracer traced, only for the hits that stayed the terrain's.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use glam::DVec3;
use helio_core::compile::Compiling;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_frame_api::rays::stream::{Answers, RayClass, Stream, SurfaceDetail, TracerId};

use super::volumes::DrawLayouts;
use super::{entry, split};

/// A form's ray stream kernels (chunk_draw.wgsl 8): each volume's ray
/// parameters in group 0 (the draw layout), the stream in group 1.
pub(super) struct RayPipelines {
    /// A nearest-hit stream's entry (the rays' starts) and walk; an
    /// occlusion stream's both in one.
    enter_nearest: wgpu::ComputePipeline,
    nearest: wgpu::ComputePipeline,
    /// A stream in runs from one origin: each run's start.
    enter_runs: wgpu::ComputePipeline,
    occlusion: wgpu::ComputePipeline,
    /// The surfaces of a stream's hits, by the detail it asks
    /// (`SurfaceDetail`): `Diffuse`'s, then `Full`'s.
    surface: [wgpu::ComputePipeline; 2],
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
            occlusion: compute(
                "voxel occlusion rays",
                &layouts.ray_occlusion,
                "ray_occlusion",
            ),
            enter_runs: compute("voxel ray run starts", &layouts.rays, "ray_enter_runs"),
            surface: [
                compute(
                    "voxel ray surfaces (diffuse)",
                    &layouts.ray_surfaces,
                    "ray_surface",
                ),
                compute(
                    "voxel ray surfaces (full)",
                    &layouts.ray_surfaces,
                    "ray_surface_full",
                ),
            ],
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
    /// The stream's layouts: a nearest-hit stream's trace, an occlusion
    /// stream's, the surfaces'.
    pub(super) layouts: [wgpu::BindGroupLayout; 3],
    pub(super) eye: wgpu::Buffer,
    /// The sun's occlusion pages, which answer occlusion rays along the sun
    /// past their first texels (chunk_draw.wgsl `paged_occlusion`).
    pub(super) sun: SunPages,
}

/// What occlusion rays read of the sun's occlusion pages
/// (`helio_light_api::sun_occlusion`) and where they ask for those they
/// lack.
#[derive(Clone)]
pub(super) struct SunPages {
    pub frame: wgpu::Buffer,
    pub atlas: wgpu::TextureView,
    pub pages: wgpu::Buffer,
    pub requests: wgpu::Buffer,
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
    /// The surface detail a nearest-hit stream asks (`Stream::detail`):
    /// which surface kernel its hits take.
    detail: SurfaceDetail,
    tracer: TracerId,
    /// The stream's rays a run from one origin (`Stream::run`; 1: none).
    run: u32,
    frame: wgpu::Buffer,
    /// The rays' starts in the volume being traced (`RayStart`,
    /// [`START_BYTES`] a ray), and the runs' (one a run), as large as the
    /// stream's rays need.
    starts: Mutex<Option<[wgpu::Buffer; 2]>>,
    cache: helio_core::gpu_cache::GpuCache,
}

/// `RayStart`'s size (chunk_draw.wgsl 8).
const START_BYTES: u64 = 64;

impl StreamTracer {
    pub fn new(device: &wgpu::Device, class: RayClass, tracer: TracerId) -> Self {
        Self {
            class,
            detail: SurfaceDetail::Full,
            tracer,
            run: 1,
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

    /// A stream whose rays come in runs of `run` from one origin
    /// (`Stream::run`): a nearest-hit stream's entry is then a run's, not a
    /// ray's (chunk_draw.wgsl `ray_enter_runs`). An occlusion stream's rays
    /// enter in their walk's kernel and ignore it.
    pub fn with_run(mut self, run: u32) -> Self {
        self.run = run.max(1);
        self
    }

    /// A stream whose hits need `detail` of their surface
    /// (`Stream::detail`): `Diffuse` takes the cheaper kernel, `Full` shades
    /// a hit as the G-buffer's publish does (chunk_draw.wgsl 8).
    pub fn with_detail(mut self, detail: SurfaceDetail) -> Self {
        self.detail = detail;
        self
    }

    /// The starts' buffers for `rays`, a ray's and a run's (made, or remade
    /// larger, when short).
    fn starts(&self, device: &wgpu::Device, rays: &wgpu::Buffer) -> [wgpu::Buffer; 2] {
        let rows = (rays.size() / helio_frame_api::rays::stream::SceneRay::BYTES).max(1);
        let sizes = [rows, rows.div_ceil(u64::from(self.run))].map(|n| n * START_BYTES);
        let mut starts = self
            .starts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if starts
            .as_ref()
            .is_none_or(|b| b[0].size() < sizes[0] || b[1].size() < sizes[1])
        {
            *starts = Some(sizes.map(|size| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("voxel ray starts"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
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
            tracer: [self.tracer.0, self.run, 0, 0],
        };
        write(&self.frame, bytemuck::bytes_of(&frame));
    }

    /// The bind group of `stream`'s trace (`surface` false) or its surfaces'
    /// pass, through `cut`.
    fn group(
        &self,
        device: &wgpu::Device,
        cut: &RayCut,
        stream: StreamBindings<'_>,
        surface: bool,
    ) -> wgpu::BindGroup {
        let starts =
            (self.class == RayClass::Nearest && !surface).then(|| self.starts(device, stream.rays));
        let mut entries = vec![
            entry(67, stream.header),
            entry(68, stream.rays),
            entry(69, stream.hits),
            entry(71, &self.frame),
            entry(72, &cut.eye),
        ];
        let layout = match (self.class, stream.surfaces) {
            (RayClass::Nearest, Some(surfaces)) if surface => {
                entries.push(entry(70, surfaces));
                2
            }
            (RayClass::Nearest, _) => {
                let [starts, runs] = starts.as_ref().expect("a trace's starts");
                entries.extend([entry(73, starts), entry(74, runs)]);
                0
            }
            _ => {
                entries.extend([
                    entry(62, &cut.sun.frame),
                    wgpu::BindGroupEntry {
                        binding: 63,
                        resource: wgpu::BindingResource::TextureView(&cut.sun.atlas),
                    },
                    entry(65, &cut.sun.pages),
                    entry(66, &cut.sun.requests),
                ]);
                1
            }
        };
        self.cache.bind_group(
            device,
            &wgpu::BindGroupDescriptor {
                label: Some("voxel rays"),
                layout: &cut.layouts[layout],
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
        let group = self.group(device, cut, stream, false);
        // The runs' starts: a thread a run, over as many as the rays' rows
        // hold (those past the header's count return at once).
        let rows = stream.rays.size() / helio_frame_api::rays::stream::SceneRay::BYTES;
        let run_groups = rows.div_ceil(u64::from(self.run)).div_ceil(64) as u32;
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
                if self.run > 1 {
                    pass.set_pipeline(&kernels.enter_runs);
                    pass.dispatch_workgroups(run_groups, 1, 1);
                }
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
        if self.class != RayClass::Nearest || stream.surfaces.is_none() {
            return;
        }
        let group = self.group(device, cut, stream, true);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("voxel ray surfaces"),
            timestamp_writes: timing,
        });
        pass.set_bind_group(1, &group, &[]);
        for v in &cut.volumes {
            let Some(kernels) = cut.kernels(v.plane) else {
                continue;
            };
            pass.set_pipeline(&kernels.surface[usize::from(self.detail == SurfaceDetail::Full)]);
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
            tracer: StreamTracer::new(device, stream.class, id)
                .with_run(stream.run)
                .with_detail(stream.detail),
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
