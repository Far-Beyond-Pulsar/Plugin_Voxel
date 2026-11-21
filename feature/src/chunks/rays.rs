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
//!   short walks keep their entry in their kernel (`ray_occlusion`); one
//!   along the sun as wide as a page texel walks only its first two texels
//!   and takes the sun's occlusion pages past that (`paged_occlusion`). The
//!   surfaces are written once every tracer traced, only for the hits that
//!   stayed the terrain's, by the stream's detail: `Full` shades a hit as
//!   the G-buffer's publish shades the face (`ray_surface_full`,
//!   `voxel_surface`), `Diffuse` takes the cheaper kernel (`ray_surface`).
//! - **Compiled in the order the streams need them.** Each kernel compiles
//!   as a job of its own ([`RayKernel`], `helio_core::compile`), only those
//!   a scheduled stream needs ([`RayKernels::of`]): first every stream's
//!   trace and the `Diffuse` surfaces, then `Full`'s (the longest, which only
//!   a stream asking `Full` waits for). A stream is answered once its own
//!   kernels are ready for every volume ([`RayLink::complete`]); until then
//!   it is short and traces nothing.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use glam::DVec3;
use helio_core::compile::Compiling;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_frame_api::rays::stream::{Answers, RayClass, Stream, SurfaceDetail, TracerId};

use super::volumes::DrawLayouts;
use super::{entry, split};

/// A ray stream kernel of a form's draw source (chunk_draw.wgsl 8): each
/// volume's ray parameters in group 0 (the draw layout), the stream in
/// group 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RayKernel {
    /// A nearest-hit stream's entry (the rays' starts).
    EnterNearest,
    /// A nearest-hit stream's walk.
    Nearest,
    /// A stream in runs from one origin: each run's start.
    EnterRuns,
    /// An occlusion stream's entry and walk, in one.
    Occlusion,
    /// The surfaces of a `SurfaceDetail::Diffuse` stream's hits.
    SurfaceDiffuse,
    /// The surfaces of a `SurfaceDetail::Full` stream's hits.
    SurfaceFull,
}

impl RayKernel {
    pub const ALL: [RayKernel; 6] = [
        RayKernel::EnterNearest,
        RayKernel::Nearest,
        RayKernel::EnterRuns,
        RayKernel::Occlusion,
        RayKernel::SurfaceDiffuse,
        RayKernel::SurfaceFull,
    ];

    fn bit(self) -> u8 {
        1 << self as u8
    }

    /// The order it compiles in: every trace kernel and `Diffuse`'s surfaces
    /// first (0), then `Full`'s (1), the longest kernel, which only a stream
    /// asking `Full` waits for. Kernels of one order compile at once.
    pub(super) fn order(self) -> u8 {
        u8::from(self == RayKernel::SurfaceFull)
    }

    /// Its entry point, label and layout (of `layouts`).
    fn entry(self) -> (&'static str, &'static str) {
        match self {
            RayKernel::EnterNearest => ("ray_enter_nearest", "voxel nearest ray starts"),
            RayKernel::Nearest => ("ray_nearest", "voxel nearest rays"),
            RayKernel::EnterRuns => ("ray_enter_runs", "voxel ray run starts"),
            RayKernel::Occlusion => ("ray_occlusion", "voxel occlusion rays"),
            RayKernel::SurfaceDiffuse => ("ray_surface", "voxel ray surfaces (diffuse)"),
            RayKernel::SurfaceFull => ("ray_surface_full", "voxel ray surfaces (full)"),
        }
    }

    /// The kernel's pipeline over the form's draw source `module`.
    pub(super) fn build(
        self,
        device: &wgpu::Device,
        layouts: &DrawLayouts,
        module: &wgpu::ShaderModule,
    ) -> wgpu::ComputePipeline {
        let (entry, label) = self.entry();
        let second = match self {
            RayKernel::Occlusion => &layouts.ray_occlusion,
            RayKernel::SurfaceDiffuse | RayKernel::SurfaceFull => &layouts.ray_surfaces,
            _ => &layouts.rays,
        };
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
    }
}

/// A set of [`RayKernel`]s: what a stream needs, or what a world's
/// scheduled streams need.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RayKernels(u8);

impl RayKernels {
    pub const NONE: RayKernels = RayKernels(0);
    /// Every kernel (tests and tools tracing streams of every kind).
    pub const ALL: RayKernels = RayKernels(0x3f);

    /// What a stream of `class` in runs of `run` rays, asking `detail` of
    /// its hits' surfaces, needs traced and shaded.
    pub fn of(class: RayClass, detail: SurfaceDetail, run: u32) -> Self {
        let kernels: &[RayKernel] = match (class, detail) {
            (RayClass::Occlusion, _) => &[RayKernel::Occlusion],
            (RayClass::Nearest, SurfaceDetail::Diffuse) => &[
                RayKernel::EnterNearest,
                RayKernel::Nearest,
                RayKernel::SurfaceDiffuse,
            ],
            (RayClass::Nearest, SurfaceDetail::Full) => &[
                RayKernel::EnterNearest,
                RayKernel::Nearest,
                RayKernel::SurfaceFull,
            ],
        };
        let mut out = kernels.iter().fold(Self::NONE, |s, k| s.with(*k));
        if class == RayClass::Nearest && run > 1 {
            out = out.with(RayKernel::EnterRuns);
        }
        out
    }

    pub fn with(self, kernel: RayKernel) -> Self {
        Self(self.0 | kernel.bit())
    }

    pub fn union(self, other: RayKernels) -> Self {
        Self(self.0 | other.0)
    }

    pub fn contains(self, kernel: RayKernel) -> bool {
        self.0 & kernel.bit() != 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn iter(self) -> impl Iterator<Item = RayKernel> {
        RayKernel::ALL
            .into_iter()
            .filter(move |k| self.contains(*k))
    }
}

/// A form's ray kernels, each compiling off the frame on its own
/// ([`RayKernel`]): those asked for so far.
#[derive(Clone, Default)]
pub struct FormRayKernels {
    kernels: [Option<Arc<Compiling<wgpu::ComputePipeline>>>; 6],
}

impl FormRayKernels {
    /// Starts compiling the kernels of `wanted` not started yet over the
    /// form's draw `module`, in their order (`RayKernel::order`): a later
    /// order once every kernel of the earlier ones it started has finished.
    pub(super) fn start(
        &mut self,
        device: &wgpu::Device,
        layouts: &DrawLayouts,
        module: &wgpu::ShaderModule,
        wanted: RayKernels,
    ) {
        for order in 0..2 {
            let mut started = false;
            for kernel in wanted.iter().filter(|k| k.order() == order) {
                let slot = &mut self.kernels[kernel as usize];
                if slot.is_none() {
                    let (layouts, module) = (layouts.clone(), module.clone());
                    *slot = Some(Arc::new(Compiling::start(
                        device,
                        kernel.entry().1,
                        move |device| kernel.build(device, &layouts, &module),
                    )));
                }
                started = true;
            }
            let pending = wanted.iter().filter(|k| k.order() == order).any(|k| {
                self.kernels[k as usize]
                    .as_ref()
                    .is_some_and(|c| c.is_pending())
            });
            if started && pending {
                return;
            }
        }
    }

    /// Kernel `kernel`'s pipeline, once compiled.
    pub fn ready(&self, kernel: RayKernel) -> Option<&wgpu::ComputePipeline> {
        self.kernels[kernel as usize]
            .as_ref()
            .and_then(|c| c.ready())
    }

    /// Kernel `kernel`'s pipeline, of a form whose kernels a stream checked
    /// ready ([`RayCut::kernels`]).
    fn get(&self, kernel: RayKernel) -> &wgpu::ComputePipeline {
        self.ready(kernel).expect("a stream's kernels are ready")
    }

    /// Whether every kernel of `kernels` is compiled.
    pub fn all_ready(&self, kernels: RayKernels) -> bool {
        kernels.iter().all(|k| self.ready(k).is_some())
    }

    /// Blocks until every kernel started has finished compiling, or one
    /// failed: for tests and offline tools.
    pub(super) fn wait(&self) -> Result<(), String> {
        for c in self.kernels.iter().flatten() {
            c.wait().map_err(str::to_owned)?;
        }
        Ok(())
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
    /// Each form's kernels (spheres, planes).
    pub(super) pipelines: [FormRayKernels; 2],
    /// The stream's layouts: a nearest-hit stream's trace, an occlusion
    /// stream's, the surfaces'.
    pub(super) layouts: [wgpu::BindGroupLayout; 3],
    pub(super) eye: wgpu::Buffer,
    /// The sun's occlusion pages, which answer occlusion rays along the sun
    /// past their first texels (chunk_draw.wgsl `paged_occlusion`).
    pub(super) sun: SunPages,
}

/// What occlusion rays read of the sun's occlusion pages
/// (`helio_light_api::sun_occlusion`).
#[derive(Clone)]
pub(super) struct SunPages {
    pub frame: wgpu::Buffer,
    pub atlas: wgpu::TextureView,
    pub pages: wgpu::Buffer,
}

impl RayCut {
    /// The volumes a stream's rays are traced against.
    pub fn volumes(&self) -> usize {
        self.volumes.len()
    }

    /// Whether every volume's form has the kernels of `kernels` (what a
    /// stream needs, [`RayKernels::of`]).
    pub fn ready(&self, kernels: RayKernels) -> bool {
        self.volumes
            .iter()
            .all(|v| self.pipelines[usize::from(v.plane)].all_ready(kernels))
    }

    /// The kernels of form `plane`, if every one of `kernels` is compiled.
    fn kernels(&self, plane: bool, kernels: RayKernels) -> Option<&FormRayKernels> {
        let form = &self.pipelines[usize::from(plane)];
        form.all_ready(kernels).then_some(form)
    }
}

/// Where a graph's chunk pass publishes the cut and the terrain's tracer
/// passes find it: shared by the graph's build ([`SharedRayLink`]), so
/// passes built before the chunk pass's node (a consumer before the
/// G-buffer phase) find the same link.
#[derive(Default)]
pub struct RayLink {
    cut: Mutex<Option<Arc<RayCut>>>,
    /// The kernels the graph's streams need ([`RayKernels`] bits).
    wanted: AtomicU8,
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

    /// A stream of the graph needing `kernels` is traced by the terrain: the
    /// world keeps a cut for it and compiles them
    /// ([`super::ChunkWorld::want_rays`]).
    pub fn want(&self, kernels: RayKernels) {
        self.wanted.fetch_or(kernels.0, Ordering::Relaxed);
    }

    /// The kernels the graph's streams need (none: no stream is traced).
    pub fn wanted(&self) -> RayKernels {
        RayKernels(self.wanted.load(Ordering::Relaxed))
    }

    /// Whether this frame holds volumes (the chunk pass, as it takes the
    /// frame's).
    pub fn set_volumes(&self, held: bool) {
        self.volumes.store(held, Ordering::Relaxed);
        if !held {
            self.drawn.store(false, Ordering::Relaxed);
        }
    }

    /// Whether the cut answers a stream needing `kernels` for every volume
    /// the frame holds: none held, or chunks of them drawn once and those
    /// kernels compiled. Short (`stream::Answers::short`) until the volumes'
    /// first chunks are drawn and while the stream's kernels compile. Later
    /// frames that draw none of them (an eye looking away) and chunks still
    /// generating are what the cut holds of the volumes, no shortfall.
    pub fn complete(&self, kernels: RayKernels) -> bool {
        !self.volumes.load(Ordering::Relaxed)
            || (self.drawn.load(Ordering::Relaxed)
                && self.cut().is_some_and(|cut| cut.ready(kernels)))
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

    /// The kernels the stream needs: a volume is traced only once every one
    /// of them is compiled, so a hit it writes always gets its surface.
    pub fn kernels(&self) -> RayKernels {
        RayKernels::of(self.class, self.detail, self.run)
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
    /// volume over the stream's indirect groups (volumes whose form has not
    /// compiled every kernel the stream needs trace nothing), its pass timed
    /// as `timing` says.
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
        let needs = self.kernels();
        for v in &cut.volumes {
            let Some(kernels) = cut.kernels(v.plane, needs) else {
                continue;
            };
            pass.set_bind_group(0, &v.group, &[]);
            if self.class == RayClass::Nearest {
                if self.run > 1 {
                    pass.set_pipeline(kernels.get(RayKernel::EnterRuns));
                    pass.dispatch_workgroups(run_groups, 1, 1);
                }
                pass.set_pipeline(kernels.get(RayKernel::EnterNearest));
                pass.dispatch_workgroups_indirect(stream.header, 0);
            }
            pass.set_pipeline(kernels.get(match self.class {
                RayClass::Nearest => RayKernel::Nearest,
                RayClass::Occlusion => RayKernel::Occlusion,
            }));
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
        let surface = match self.detail {
            SurfaceDetail::Diffuse => RayKernel::SurfaceDiffuse,
            SurfaceDetail::Full => RayKernel::SurfaceFull,
        };
        for v in &cut.volumes {
            // Only volumes the trace walked (every kernel of the stream).
            let Some(kernels) = cut.kernels(v.plane, self.kernels()) else {
                continue;
            };
            pass.set_pipeline(kernels.get(surface));
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
        let tracer = StreamTracer::new(device, stream.class, id)
            .with_run(stream.run)
            .with_detail(stream.detail);
        link.want(tracer.kernels());
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
            tracer,
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
        if !self.surface && !self.link.complete(self.tracer.kernels()) {
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
