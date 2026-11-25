//! Chunk trees against the canonical CPU planet: the WGSL terrain programs
//! bit for bit, and drawn hits near the eye (level 0) against exact CPU ray
//! casts, over natural terrain, caves, overhangs, planes, features and
//! edits.
mod common;
use common::{gpu, Gpu};
use glam::{DVec3, Mat4, Vec3};
use helio_pass_voxel_planet::chunks::{
    residency, Capacity, ChunkWorld, KeptChunks, ResidencyConfig, SceneLights, VolumeId,
};
use helio_pass_voxel_planet::engine::{PickHit, PickResult, VolumeAnswer, VolumeAnswers};
use helio_pass_voxel_planet::features::{FeatureSet, Spawn};
use helio_pass_voxel_planet::grid::Shape;
use helio_pass_voxel_planet::layers::{Layer, LayerKind, TerrainLayers};
use helio_pass_voxel_planet::stamp::{Part, Shape as Form, Stamp};
use helio_pass_voxel_planet::terrain::material;
use helio_pass_voxel_planet::{
    Brush, BrushOp, BrushShape, Cell, Planet, PlanetRecipe, TerrainSource,
};

const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;

/// The GI probes' finest spacing the tests place probes at: open ground's
/// (`GiSettings::probe_spacing` over terrain), whatever GI's default is.
const PROBE_SPACING: f64 = 1.0;

/// A camera at the eye (camera-relative) looking along `forward`.
struct View {
    eye: DVec3,
    forward: Vec3,
    size: [u32; 2],
    /// The near plane (m).
    near: f32,
}

impl View {
    fn new(eye: DVec3, forward: Vec3, size: [u32; 2]) -> Self {
        Self {
            eye,
            forward: forward.normalize(),
            size,
            near: 0.05,
        }
    }
    fn view_proj(&self, planet: &Planet) -> Mat4 {
        let up = planet.grid().up(self.eye).as_vec3();
        let up = if self.forward.dot(up).abs() > 0.99 {
            up.any_orthonormal_vector()
        } else {
            up
        };
        let view = glam::camera::rh::view::look_to_mat4(Vec3::ZERO, self.forward, up);
        let proj = helio_core::depth::perspective(
            FOV_Y,
            self.size[0] as f32 / self.size[1] as f32,
            self.near,
        );
        proj * view
    }
    /// The ray through pixel (x, y)'s centre.
    fn pixel_dir(&self, view_proj: Mat4, x: u32, y: u32) -> DVec3 {
        let ndc = glam::Vec4::new(
            (x as f32 + 0.5) / self.size[0] as f32 * 2.0 - 1.0,
            1.0 - (y as f32 + 0.5) / self.size[1] as f32 * 2.0,
            0.5,
            1.0,
        );
        let w = view_proj.inverse() * ndc;
        (w.truncate() / w.w).as_dvec3().normalize()
    }
    /// A pixel's angle.
    fn pixel_angle(&self) -> f64 {
        2.0 * f64::from((FOV_Y * 0.5).tan()) / f64::from(self.size[1])
    }
    fn config(&self) -> ResidencyConfig {
        let mut config = ResidencyConfig {
            footprints_per_frame: 512,
            generate_ms: None,
            ..Default::default()
        };
        config.plan.pixel_angle = self.pixel_angle();
        config
    }
}

/// Chunks of one planet and their frame counter.
/// A test world's capacity: past the chunks the largest view settles with
/// (12901, the hollowed core; the others 8683 at most) and the detail pages
/// natural per-voxel materials take (10076 at most), so tests running in
/// parallel fit in a 12 GB GPU (1.2 GB a test; a game's
/// default world, 2.9 GB, paged eight parallel tests out of device memory
/// and stalled them).
const TEST_CAPACITY: Capacity = Capacity {
    chunks: 16384,
    detail_pages: 16384,
    footprints: 1024,
    bricks: 65536,
    kept_bytes: 256 << 20,
    counters: false,
};

/// A tool's picks waiting to be copied (request id, view point 0..1 from
/// the top left) and the inbox they are answered into.
#[derive(Default)]
struct Picks {
    waiting: Vec<(u64, [f32; 2])>,
    answers: std::sync::Arc<VolumeAnswers>,
}

impl Picks {
    /// The answers so far.
    fn results(&self) -> Vec<PickResult> {
        self.answers
            .take()
            .into_iter()
            .filter_map(|a| match a {
                VolumeAnswer::Pick(r) => Some(r),
                _ => None,
            })
            .collect()
    }
}

struct World {
    chunks: ChunkWorld,
    frame: u64,
    /// The direction to the sun (eye-relative).
    sun: Vec3,
    /// The meshes' depth the draws shade (none: no meshes).
    scene_depth: Option<wgpu::TextureView>,
    /// Scene lights the draws shadow (`SceneLights`, eye-relative: lights,
    /// tile lists, tile counts).
    lights: Option<[wgpu::Buffer; 3]>,
    /// Whether each frame draws the sun's occlusion pages it lacks.
    sun_occlusion: bool,
    /// The projection jittered each frame as Helio's (`r1_r2_jitter`): the
    /// next frame's place in the jitter's sequence.
    jitter: Option<u64>,
    /// The last frame's view-projection and jitter (NDC).
    last: Option<(Mat4, glam::Vec2)>,
}

impl World {
    fn new(gpu: &Gpu, planet: &Planet) -> Self {
        Self::with_capacity(gpu, planet, TEST_CAPACITY)
    }

    fn with_capacity(gpu: &Gpu, planet: &Planet, capacity: Capacity) -> Self {
        Self {
            chunks: ChunkWorld::of_planet(&gpu.device, &gpu.queue, planet, capacity),
            frame: 0,
            sun: Vec3::new(0.3, 0.8, 0.4).normalize(),
            scene_depth: None,
            lights: None,
            sun_occlusion: false,
            jitter: None,
            last: None,
        }
    }

    /// The planet's edits now.
    fn sync(&mut self, _gpu: &Gpu, planet: &Planet) {
        self.chunks
            .sync_edits(VolumeId(0), planet)
            .expect("the world's volume");
    }

    /// One frame: residency, the draw, `picks` copied; its residency.
    fn frame(
        &mut self,
        gpu: &Gpu,
        planet: &Planet,
        view: &View,
        picks: Option<&mut Picks>,
    ) -> helio_pass_voxel_planet::chunks::UpdateStats {
        self.frame_picked(gpu, planet, view, &view.config(), picks)
    }

    /// One frame with residency `config`.
    fn frame_with(
        &mut self,
        gpu: &Gpu,
        planet: &Planet,
        view: &View,
        config: &ResidencyConfig,
    ) -> helio_pass_voxel_planet::chunks::UpdateStats {
        self.frame_picked(gpu, planet, view, config, None)
    }

    fn frame_picked(
        &mut self,
        gpu: &Gpu,
        planet: &Planet,
        view: &View,
        config: &ResidencyConfig,
        picks: Option<&mut Picks>,
    ) -> helio_pass_voxel_planet::chunks::UpdateStats {
        let mut view_proj = view.view_proj(planet);
        let mut jitter = glam::Vec2::ZERO;
        if let Some(n) = self.jitter.as_mut() {
            jitter = glam::Vec2::from(helio_core::temporal::r1_r2_jitter(*n)) * 2.0
                / glam::Vec2::new(view.size[0] as f32, view.size[1] as f32);
            *n += 1;
            view_proj = Mat4::from_translation(jitter.extend(0.0)) * view_proj;
        }
        let (prev_view_proj, prev_jitter) = self
            .last
            .replace((view_proj, jitter))
            .unwrap_or((view_proj, jitter));
        let tan_y = f64::from((FOV_Y * 0.5).tan());
        let tan_x = tan_y * f64::from(view.size[0]) / f64::from(view.size[1]);
        let cone = residency::View {
            forward: view.forward.as_dvec3(),
            cos_half: (tan_x * tan_x + tan_y * tan_y).sqrt().atan().cos(),
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut recorder = helio_core::CommandRecorder::from_encoder(&mut encoder);
        let stats = self.chunks.update(
            &mut recorder,
            view.eye,
            DVec3::ZERO,
            Some(cone),
            self.frame,
            config,
        );
        let lights = self
            .lights
            .as_ref()
            .map(|[lights, tile_lists, tile_counts]| SceneLights {
                lights,
                tile_lists,
                tile_counts,
                eye: Vec3::ZERO,
            });
        // As the pass: the sun's occlusion before the draw.
        if self.sun_occlusion {
            self.chunks
                .draw_sun_occlusion(&mut recorder, self.frame, view.eye, self.sun);
        }
        self.chunks.draw(
            &mut recorder,
            self.frame,
            view.eye,
            view_proj,
            prev_view_proj,
            [jitter, prev_jitter],
            view.near,
            self.sun,
            view.size,
            self.scene_depth.as_ref(),
            lights.as_ref(),
        );
        if let Some(picks) = picks {
            let taken = self
                .chunks
                .copy_picks(&mut recorder, &picks.waiting, &picks.answers);
            picks.waiting.drain(..taken);
        }
        drop(recorder);
        gpu.queue.submit([encoder.finish()]);
        self.chunks.submitted();
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        self.frame += 1;
        stats
    }

    /// Frames until the sun's occlusion pages the pixels asked for are all
    /// traced (and current).
    fn settle_sun_pages(&mut self, gpu: &Gpu, planet: &Planet, view: &View) {
        for n in 0.. {
            self.frame(gpu, planet, view, None);
            let s = self.chunks.sun_page_stats().expect("drawn");
            if n >= 3 && s.missing == 0 && s.stale == 0 && s.coarse == 0 {
                eprintln!("sun pages settled in {n} frames: {s:?}");
                return;
            }
            assert!(n < 160, "sun pages did not settle: {s:?}");
        }
    }

    /// Frames until residency settles; every pixel's hit then.
    fn settle(&mut self, gpu: &Gpu, planet: &Planet, view: &View) -> Vec<Option<PickHit>> {
        let started = self.frame;
        let clock = std::time::Instant::now();
        let stats = loop {
            let stats = self.frame(gpu, planet, view, None);
            if self.chunks.settled() {
                break stats;
            }
            if (self.frame - started) % 500 == 0 {
                eprintln!(
                    "settling: {} frames, {:.1} s",
                    self.frame - started,
                    clock.elapsed().as_secs_f64()
                );
            }
            assert!(
                self.frame - started < 4000,
                "residency did not settle: {stats:?}"
            );
        };
        eprintln!(
            "settled in {} frames: {} chunks drawn; {stats:?}",
            self.frame - started,
            self.chunks.chunk_count()
        );
        self.chunks.read_hits()
    }
}

/// Drawn hits against CPU ray casts: every second pixel whose CPU hit lies
/// within the level-0 range (cells of a pixel or more) must be drawn at
/// level 0 at the same cell (or its neighbour across an edge the ray runs
/// along) and distance. A chunk is drawn in an affine frame, off the curved
/// grid by up to 512 voxel^2 / radius (a millionth of a millimetre on Earth,
/// 2 mm on a 3 km world): within that a ray grazing a step's corner may meet
/// its neighbour. (compared, mismatched).
fn compare(gpu: &Gpu, world: &mut World, planet: &Planet, view: &View) -> (usize, usize) {
    let hits = world.settle(gpu, planet, view);
    let view_proj = view.view_proj(planet);
    let near = 0.7 * planet.grid().voxel_size() / view.pixel_angle();
    let voxel = planet.grid().voxel_size();
    let frame = if planet.grid().is_plane() {
        0.0
    } else {
        4.0 * 512.0 * voxel * voxel / planet.grid().radius()
    };
    let (mut compared, mut mismatched) = (0, 0);
    for y in (0..view.size[1]).step_by(2) {
        for x in (0..view.size[0]).step_by(2) {
            let Some(cpu) = planet
                .raycast(view.eye, view.pixel_dir(view_proj, x, y), near)
                .filter(|c| c.distance < near)
            else {
                continue;
            };
            compared += 1;
            let drawn = hits[(y * view.size[0] + x) as usize];
            // A ray along the edge between two solid cells meets both at
            // once: either is exact.
            let tie = |c: Cell| {
                c.face == cpu.cell.face
                    && (c.i - cpu.cell.i).abs()
                        + (c.j - cpu.cell.j).abs()
                        + (c.k - cpu.cell.k).abs()
                        == 1
                    && planet.solid(c)
            };
            let exact = drawn.is_some_and(|h| {
                h.level == 0
                    && (h.cell == cpu.cell || tie(h.cell))
                    && (h.distance - cpu.distance).abs() <= 2e-3 + cpu.distance * 1e-5 + frame
            });
            if !exact {
                mismatched += 1;
                if mismatched < 8 {
                    eprintln!(
                        "pixel {x},{y}: drawn {drawn:?}, cpu {:?} at {}",
                        cpu.cell, cpu.distance
                    );
                }
            }
        }
    }
    eprintln!("compared {compared}, mismatched {mismatched}");
    (compared, mismatched)
}

/// [`compare`] with a new world: at most one pixel in a thousand differs.
fn matches(gpu: &Gpu, planet: &Planet, view: View) {
    let (compared, mismatched) = compare(gpu, &mut World::new(gpu, planet), planet, &view);
    assert!(compared > 1000, "{compared} pixels compared");
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
}

/// [`compare`] with a new world: every pixel agrees.
fn matches_exactly(gpu: &Gpu, planet: &Planet, view: View, at_least: usize) {
    let (compared, mismatched) = compare(gpu, &mut World::new(gpu, planet), planet, &view);
    assert!(compared > at_least, "{compared} pixels compared");
    assert_eq!(mismatched, 0);
}

/// A land direction (surface above sea level) near `face` coordinates.
fn land(planet: &Planet, face: u8, fi: f64, fj: f64) -> DVec3 {
    let grid = planet.grid();
    let n = f64::from(grid.cells());
    for step in 0..400 {
        let a = fi + 0.013 * f64::from(step % 20);
        let b = fj + 0.017 * f64::from(step / 20);
        let (i, j) = ((a.fract() * n) as i32, (b.fract() * n) as i32);
        if planet.column_top(face, i, j, 0) > 40 {
            return grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        }
    }
    panic!("no land found");
}

/// Standing on the ground at `dir`, looking out and down.
fn ground_view(planet: &Planet, dir: DVec3, size: [u32; 2]) -> View {
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    View::new(
        eye,
        (up.any_orthonormal_vector() - up * 0.35)
            .normalize()
            .as_vec3(),
        size,
    )
}

fn xorshift(mut state: u64) -> impl FnMut() -> u64 {
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

#[test]
fn terrain_programs_are_bit_identical_to_cpu() {
    let Some(gpu) = gpu() else { return };
    let moon = TerrainLayers::moon().source(7);
    let flat = TerrainLayers {
        soil_depth_m: 2.0,
        ..TerrainLayers::flat_at(-1.25)
    }
    .source(7);
    let desert = TerrainLayers::desert().source(7);
    let mut deep = TerrainLayers::earth();
    deep.volumes.caves.depth_max_m = 700.0;
    deep.volumes.caves.max_depth_m = 900.0;
    let deep = deep.source(7);
    let mut cliffs = TerrainLayers::earth();
    cliffs.layers.push(Layer {
        coverage: 0.6,
        ..Layer::new(LayerKind::Cliffs)
    });
    let cliffs = cliffs.source(7);
    for (shape, size, terrain) in [
        (Shape::Sphere, 0.1, TerrainSource::default()),
        (Shape::Sphere, 0.3, TerrainSource::default()),
        (Shape::Sphere, 1.0, TerrainSource::default()),
        (Shape::Plane, 0.1, TerrainSource::default()),
        (Shape::InfinitePlane, 0.1, TerrainSource::default()),
        (Shape::InfinitePlane, 1.0, TerrainSource::default()),
        (Shape::Plane, 0.1, flat.clone()),
        (Shape::Sphere, 0.1, flat),
        (Shape::Sphere, 0.1, moon.clone()),
        (Shape::Sphere, 1.0, moon.clone()),
        (Shape::Plane, 0.1, moon.clone()),
        (Shape::InfinitePlane, 0.1, moon),
        (Shape::Sphere, 0.1, desert.clone()),
        (Shape::Plane, 0.3, desert),
        (Shape::Sphere, 0.1, deep),
        (Shape::Sphere, 0.1, cliffs.clone()),
        (Shape::Plane, 0.3, cliffs.clone()),
        (Shape::InfinitePlane, 1.0, cliffs),
    ] {
        let planet = Planet::new(PlanetRecipe {
            shape,
            voxel_size_m: size,
            plane_size_m: 5_000.0,
            terrain: terrain.clone(),
            ..Default::default()
        })
        .unwrap();
        helio_pass_voxel_planet::engine::verify_field(&gpu.device, &gpu.queue, &planet, 20_000)
            .unwrap_or_else(|e| panic!("{} on {shape:?} at {size} m: {e}", terrain.generator));
    }
}

/// Rivers' valleys, floodplains and channels on the GPU are the CPU's:
/// columns of every level around river segments' nodes, on spheres and an
/// infinite plane.
#[test]
fn rivers_are_bit_identical_to_cpu() {
    let Some(gpu) = gpu() else { return };
    for (shape, size) in [
        (Shape::Sphere, 0.1),
        (Shape::Sphere, 1.0),
        (Shape::InfinitePlane, 0.1),
    ] {
        let planet = Planet::new(PlanetRecipe {
            shape,
            voxel_size_m: size,
            ..Default::default()
        })
        .unwrap();
        let grid = *planet.grid();
        let field = TerrainLayers::earth().field(&grid, 7).unwrap();
        let nodes = field.maps().expect("Earth bakes maps").river_nodes();
        let mut next = xorshift(0x5eed_0f_d1ce);
        let mut columns = Vec::new();
        for (n, &(p, floodplain, _)) in nodes
            .iter()
            .enumerate()
            .step_by((nodes.len() / 4_000).max(1))
        {
            // The node's column, then a column across its valley.
            let world = if grid.is_plane() {
                DVec3::new(f64::from(p.x), 0.0, f64::from(p.z))
                    * helio_pass_voxel_planet::grid::DOMAIN_UNIT
            } else {
                p.as_dvec3().normalize() * grid.radius()
            };
            let (cell, _) = grid.locate(world);
            let level = (n as u32) % 12;
            let reach = ((floodplain / grid.voxel_size()) as u64).max(8);
            for _ in 0..4 {
                let d = |v: u64| (v % (2 * reach)) as i32 - reach as i32;
                let (i, j) = (cell.i + d(next()), cell.j + d(next()));
                let last = grid.cells() - 1;
                columns.push([
                    i32::from(cell.face),
                    i.clamp(0, last) >> level,
                    j.clamp(0, last) >> level,
                    level as i32,
                ]);
            }
        }
        eprintln!("{shape:?} {size} m: {} river columns", columns.len());
        helio_pass_voxel_planet::engine::verify_columns(&gpu.device, &gpu.queue, &planet, &columns)
            .unwrap_or_else(|e| panic!("rivers on {shape:?} at {size} m: {e}"));
    }
}

/// Streams (the maps' drainage refined to creeks) on the GPU are the
/// CPU's: columns of every level along stream levels' cells near the
/// planet's top and across a face edge, on a sphere and an infinite plane.
#[test]
fn streams_are_bit_identical_to_cpu() {
    let Some(gpu) = gpu() else { return };
    let mut stack = TerrainLayers::earth();
    for l in &mut stack.layers {
        if l.kind == helio_pass_voxel_planet::layers::LayerKind::Rivers {
            l.octaves = 3;
        }
    }
    for (shape, size) in [(Shape::Sphere, 0.1), (Shape::InfinitePlane, 0.1)] {
        let planet = Planet::new(PlanetRecipe {
            shape,
            voxel_size_m: size,
            terrain: stack.source(7),
            ..Default::default()
        })
        .unwrap();
        let grid = *planet.grid();
        let mut next = xorshift(0x57_7ea_3);
        let mut columns = Vec::new();
        // Spots 0..40 km apart near the top (or the plane's origin) and
        // along the +Y face's edge.
        let spots: Vec<DVec3> = (0..24)
            .map(|n| {
                let a = f64::from(n) * 2.4;
                let d = 1_700.0 * f64::from(n);
                if grid.is_plane() {
                    DVec3::new(a.cos() * d, 0.0, a.sin() * d)
                } else if n % 3 == 0 {
                    DVec3::new(1.0, 1.0, (f64::from(n) / 24.0) - 0.5).normalize() * grid.radius()
                } else {
                    (DVec3::Y * grid.radius() + DVec3::new(a.cos() * d, 0.0, a.sin() * d))
                        .normalize()
                        * grid.radius()
                }
            })
            .collect();
        for (n, &world) in spots.iter().enumerate() {
            let (cell, _) = grid.locate(world);
            for level in 0..14u32 {
                let reach = 40_000.0 / grid.voxel_size();
                for _ in 0..(if level < 6 { 60 } else { 12 }) {
                    let d = |v: u64| (v % (2 * reach as u64)) as i32 - reach as i32;
                    let (i, j) = (cell.i + d(next()), cell.j + d(next()));
                    let last = grid.cells() - 1;
                    columns.push([
                        i32::from(cell.face),
                        i.clamp(0, last) >> level,
                        j.clamp(0, last) >> level,
                        level as i32,
                    ]);
                }
            }
            let _ = n;
        }
        eprintln!("{shape:?} {size} m: {} stream columns", columns.len());
        helio_pass_voxel_planet::engine::verify_columns(&gpu.device, &gpu.queue, &planet, &columns)
            .unwrap_or_else(|e| panic!("streams on {shape:?} at {size} m: {e}"));
    }
}

#[test]
fn ground_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [320, 180]);
    matches(&gpu, &planet, view);
}

/// Level-0 columns near a face's far corner (indices past 2^26), looking
/// across the face's edge 3 m away.
#[test]
fn ground_view_across_a_face_edge_matches_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *planet.grid();
    let n = grid.cells();
    let i = n - 30;
    let j = (n - n / 10..n)
        .step_by(997)
        .find(|&j| planet.column_top(4, i, j, 0) > 40)
        .expect("land near the edge");
    assert!(i >= 1 << 26 && j >= 1 << 26);
    let dir = grid.direction(4, f64::from(i) + 0.5, f64::from(j) + 0.5);
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    // Toward the edge (+i): across it, the neighbouring face.
    let toward = grid.direction(4, f64::from(n) + 30.0, f64::from(j) + 0.5) - dir;
    let forward = ((toward - up * toward.dot(up)).normalize() - up * 0.35).normalize();
    matches(&gpu, &planet, View::new(eye, forward.as_vec3(), [320, 180]));
}

/// An eye inside a cave network (`docs/volumes.md`): over the floor of a
/// passage or chamber near the test sites, an air cell with air around it,
/// walls, floor and ceiling within 12 m, looking along the passage.
fn find_cave(planet: &Planet) -> Option<(DVec3, Vec3)> {
    let grid = *planet.grid();
    let shift = planet.volumes()?.region_shift();
    let sites = (0..6u8)
        .flat_map(|face| [(0.37, 0.61), (0.2, 0.3), (0.7, 0.45)].map(|(a, b)| (face, a, b)));
    for (face, a, b) in sites {
        let (base, _) = grid.locate(planet.surface_point(land(planet, face, a, b), 0.0));
        for n in 0..25 {
            let (ri, rj) = ((base.i >> shift) + n % 5 - 2, (base.j >> shift) + n / 5 - 2);
            for p in planet.cave_network(base.face, ri, rj).iter() {
                for t in [1, 2, 3] {
                    let at = |c: usize| (p.a[c] * (4 - t) + p.b[c] * t) / 4;
                    let k = ((at(2) + i64::from(p.rv) / 3) >> 2) as i32;
                    let (i, j) = ((at(0) >> 2) as i32, (at(1) >> 2) as i32);
                    let air = |di: i32, dj: i32, dk: i32| {
                        !planet.solid(Cell::new(base.face, i + di, j + dj, k + dk))
                    };
                    if !(-1..=1).all(|a| (-1..=1).all(|b| (-1..=1).all(|c| air(a, b, c)))) {
                        continue;
                    }
                    let eye = grid.cell_center(Cell::new(base.face, i, j, k));
                    let up = eye.normalize();
                    let end = grid.cell_center(Cell::new(
                        base.face,
                        (p.b[0] >> 2) as i32,
                        (p.b[1] >> 2) as i32,
                        k,
                    ));
                    let along = end - eye - up * (end - eye).dot(up);
                    let ahead = if along.length() > 1.0 {
                        along.normalize()
                    } else {
                        up.any_orthonormal_vector()
                    };
                    let forward = (ahead - up * 0.15).normalize();
                    if [-up, up]
                        .iter()
                        .all(|d| planet.raycast(eye, *d, 12.0).is_some())
                    {
                        return Some((eye, forward.as_vec3()));
                    }
                }
            }
        }
    }
    None
}

/// [`matches`] for an eye in a cave, printing what each side has there on
/// a mismatch: the CPU's cells and primitives over the eye's column, the
/// GPU's resident chunks over it.
fn cave_matches(gpu: &Gpu, planet: &Planet, eye: DVec3, forward: Vec3) {
    let view = View::new(eye, forward, [320, 180]);
    let mut world = World::new(gpu, planet);
    let (compared, mismatched) = compare(gpu, &mut world, planet, &view);
    if mismatched * 1000 > compared {
        let (cell, _) = planet.grid().locate(eye);
        let column = planet.volume_column(cell.face, cell.i, cell.j, 0);
        eprintln!(
            "eye {cell:?}: top {} below {} above {} primitives {:?}",
            column.top, column.below, column.above, column.primitives
        );
        let kinds: String = (cell.k - 40..cell.k + 40)
            .map(|k| {
                if planet.solid(Cell::new(cell.face, cell.i, cell.j, k)) {
                    '#'
                } else {
                    '.'
                }
            })
            .collect();
        eprintln!("cpu cells k {}..: {kinds}", cell.k - 40);
        for (key, occupied, drawn) in
            world
                .chunks
                .chunks_over(VolumeId(0), cell.face, cell.i, cell.j)
        {
            eprintln!("gpu {key:?} occupied {occupied:?} drawn {drawn}");
        }
    }
    assert!(compared > 1000, "{compared} pixels compared");
    assert!(mismatched * 1000 <= compared, "{mismatched}/{compared}");
}

/// Cave networks render exactly from inside: walls, floor and ceiling.
#[test]
fn cave_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let (eye, forward) = find_cave(&planet).expect("a cave near the test sites");
    cave_matches(&gpu, &planet, eye, forward);
}

/// Caves 600 m deep: networks far under the ground.
#[test]
fn deep_cave_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let mut stack = TerrainLayers::earth();
    stack.volumes.caves.depth_min_m = 400.0;
    stack.volumes.caves.depth_max_m = 600.0;
    stack.volumes.caves.max_depth_m = 900.0;
    let planet = Planet::new(PlanetRecipe {
        terrain: stack.source(7),
        ..Default::default()
    })
    .unwrap();
    let (eye, forward) = find_cave(&planet).expect("a cave near the test sites");
    cave_matches(&gpu, &planet, eye, forward);
}

/// Lava tubes on the Moon: wide flat-floored passages under the regolith.
#[test]
fn lava_tube_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe {
        radius_m: 1_737_400.0,
        terrain: TerrainLayers::moon().source(7),
        ..Default::default()
    })
    .unwrap();
    let (eye, forward) = find_cave(&planet).expect("a lava tube near the test sites");
    cave_matches(&gpu, &planet, eye, forward);
}

/// An eye over a cliff shape, looking up the slope: a column whose cells
/// the cliff layer moved off the heightfield by two cells or more.
fn find_overhang(planet: &Planet) -> Option<(DVec3, Vec3)> {
    let grid = *planet.grid();
    let mut next = xorshift(0x1D8E_4E27_C47D_124F);
    let sites = (0..6u8)
        .flat_map(|face| [(0.37, 0.61), (0.2, 0.3), (0.7, 0.45)].map(|(a, b)| (face, a, b)));
    for (face, a, b) in sites {
        let (base, _) = grid.locate(planet.surface_point(land(planet, face, a, b), 0.0));
        for _ in 0..20_000 {
            let i = base.i + (next() % 40_000) as i32 - 20_000;
            let j = base.j + (next() % 40_000) as i32 - 20_000;
            let column = planet.volume_column(base.face, i, j, 0);
            if column.cliff.cells < 20 {
                continue;
            }
            let top = |di: i32, dj: i32| planet.column_top(base.face, i + di, j + dj, 0);
            let (si, sj) = (top(20, 0) - top(-20, 0), top(0, 20) - top(0, -20));
            let t = top(0, 0);
            let above = column.cliff.cells;
            if (t - above..t + above)
                .filter(|&k| planet.solid(Cell::new(base.face, i, j, k)) != (k < t))
                .count()
                < 2
            {
                continue;
            }
            let eye = planet.surface_point(grid.cell_center(Cell::new(base.face, i, j, t)), 2.0);
            let up = eye.normalize();
            let (di, dj) = if si.abs() >= sj.abs() {
                (si.signum(), 0)
            } else {
                (0, sj.signum())
            };
            let along = grid.cell_center(Cell::new(base.face, i + di, j + dj, t))
                - grid.cell_center(Cell::new(base.face, i, j, t));
            let forward = ((along - up * along.dot(up)).normalize() - up * 0.3).normalize();
            return Some((eye, forward.as_vec3()));
        }
    }
    None
}

/// Cliff shapes render exactly: ledges and undercuts on steep ground.
#[test]
fn overhang_view_matches_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let mut stack = TerrainLayers::earth();
    stack.volumes.caves.enabled = false;
    stack.layers.push(Layer {
        coverage: 0.6,
        ..Layer::new(LayerKind::Cliffs)
    });
    let planet = Planet::new(PlanetRecipe {
        terrain: stack.source(7),
        ..Default::default()
    })
    .unwrap();
    let (eye, forward) = find_overhang(&planet).expect("an overhang near the test sites");
    matches(&gpu, &planet, View::new(eye, forward, [320, 180]));
}

/// Finite and infinite planes, from the ground and from above.
#[test]
fn plane_views_match_canonical_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    for shape in [Shape::Plane, Shape::InfinitePlane] {
        let planet = Planet::new(PlanetRecipe {
            shape,
            plane_size_m: 3_000.0,
            ..Default::default()
        })
        .unwrap();
        for (x, z, height, pitch) in [(12.3, -45.6, 1.7, -0.35), (-300.0, 250.0, 8.0, -1.2)] {
            let eye = planet.surface_point(DVec3::new(x, 0.0, z), height);
            matches(
                &gpu,
                &planet,
                View::new(eye, Vec3::new(0.8, pitch, 0.6), [320, 180]),
            );
        }
    }
}

/// A dug sphere and a raised cube.
#[test]
fn brushes_render_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let eye = planet.surface_point(dir, 3.0);
    let up = eye.normalize();
    let side = up.any_orthonormal_vector();
    let ground = planet.surface_point(dir, 0.0) + side * 4.0;
    planet
        .apply(Brush {
            center: ground.to_array(),
            radius: 2.5,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    planet
        .apply(Brush {
            center: (ground + side * 3.0 + up * 1.5).to_array(),
            radius: 0.8,
            shape: BrushShape::Cube,
            op: BrushOp::Add,
            material: material::BRICK,
            height: 0.0,
        })
        .unwrap();
    matches_exactly(
        &gpu,
        &planet,
        View::new(eye, (side - up * 0.6).normalize().as_vec3(), [320, 180]),
        1000,
    );
}

/// The floor of a ball 10 km across, dug from the surface: from a few metres
/// over it, looking across it.
#[test]
fn a_planet_scale_dig_renders_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 2, 0.47, 0.52);
    let ground = planet.surface_point(dir, 0.0);
    let up = ground.normalize();
    planet
        .apply(Brush {
            center: ground.to_array(),
            radius: 10_000.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    let side = up.any_orthonormal_vector();
    for (off, height, pitch) in [(0.0, 3.0, -0.5), (2_500.0, 4.0, -0.4), (6_000.0, 3.0, -0.3)] {
        let floor = ground - up * 10_000.0 + side * off;
        let floor = planet.surface_point(floor + up * 9_000.0, 0.0);
        let eye = floor + up * height;
        let view = View::new(
            eye,
            (up.cross(side) + up * pitch).normalize().as_vec3(),
            [320, 180],
        );
        let (compared, mismatched) = compare(&gpu, &mut World::new(&gpu, &planet), &planet, &view);
        eprintln!("off {off}: {mismatched}/{compared}");
        assert!(compared > 1000, "{compared} pixels compared");
        assert_eq!(mismatched, 0, "off {off}");
    }
}

/// The floor of a ball dug from the surface, seen from its centre across
/// it: every pixel's ray enters its nearest box. Far boxes lie metres apart
/// in depth, where a standard (not reversed) depth, precise to
/// distance^2 / near 2^-24, cannot tell them apart: a ray seeded in the
/// farther box entered it by its side, under the nearer one's cells. A 2 mm
/// near plane: 1.5 km away, ties 70 m deep.
#[test]
fn a_dug_floor_seen_from_afar_shows_its_nearest_cells() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 2, 0.47, 0.52);
    let ground = planet.surface_point(dir, 0.0);
    let up = ground.normalize();
    let radius = 1_500.0;
    planet
        .apply(Brush {
            center: ground.to_array(),
            radius,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    let side = up.any_orthonormal_vector();
    let view = View {
        near: 0.002,
        ..View::new(
            ground + up * 6.0,
            (side - up * 0.5).normalize().as_vec3(),
            [480, 270],
        )
    };
    let hits = World::new(&gpu, &planet).settle(&gpu, &planet, &view);
    let view_proj = view.view_proj(&planet);
    let (mut compared, mut beyond) = (0, 0);
    for y in (0..view.size[1]).step_by(12) {
        for x in (0..view.size[0]).step_by(12) {
            let dir = view.pixel_dir(view_proj, x, y);
            let Some(cpu) = planet.raycast(view.eye, dir, 4.0 * radius) else {
                continue;
            };
            // The ball's wall and floor, well under the natural ground: at
            // the rim the ground left over the ball thins to slivers a
            // coarse cell does not hold.
            let at = view.eye + dir * cpu.distance;
            let under = planet.column_top(cpu.cell.face, cpu.cell.i, cpu.cell.j, 0) - cpu.cell.k;
            if ((at - ground).length() - radius).abs() > 1.0
                || f64::from(under) * planet.grid().voxel_size() < 30.0
            {
                continue;
            }
            let Some(drawn) = hits[(y * view.size[0] + x) as usize] else {
                continue;
            };
            compared += 1;
            // A level's cells sit within a cell or so of the exact ground.
            let cell = planet.grid().voxel_size() * f64::from(1u32 << drawn.level);
            if (drawn.distance - cpu.distance).abs() > 3.0 * cell + 0.5 {
                beyond += 1;
                if beyond < 8 {
                    eprintln!(
                        "pixel {x},{y}: drawn {:.2} m at level {} entered {:?}, cpu {:.2} m",
                        drawn.distance, drawn.level, drawn.entered, cpu.distance
                    );
                }
            }
        }
    }
    eprintln!("compared {compared}, {beyond} beyond three cells");
    assert!(compared > 500, "{compared} compared");
    assert_eq!(beyond, 0);
}

/// Flat-topped boxes (what Flatten stamps): a slab removed above a level
/// and one filled below it, across a hillside.
#[test]
fn box_brushes_render_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let ground = planet.surface_point(dir, 0.0);
    for n in 0..6 {
        let at = ground + side * (f64::from(n) * 2.5);
        planet
            .apply(Brush {
                center: (at + up * 1.5).to_array(),
                radius: 1.3,
                shape: BrushShape::Cube,
                op: BrushOp::Remove,
                material: 0,
                height: 1.5,
            })
            .unwrap();
        planet
            .apply(Brush {
                center: (at - up * 0.75).to_array(),
                radius: 1.3,
                shape: BrushShape::Cube,
                op: BrushOp::Add,
                material: material::BRICK,
                height: 0.75,
            })
            .unwrap();
    }
    let eye = ground + up * 4.0 - side * 5.0 + up.cross(side) * 3.0;
    let forward = ((ground + side * 6.0) - eye).normalize().as_vec3();
    matches_exactly(&gpu, &planet, View::new(eye, forward, [320, 180]), 1000);
}

/// Destruction to any depth: at the bottom of a 600 m shaft, and 300 m up
/// it with the same world.
#[test]
fn a_deep_shaft_renders_exactly_at_any_depth() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 3, 0.41, 0.57);
    let ground = planet.surface_point(dir, 0.0);
    let up = planet.grid().up(ground);
    let mut depth = -10.0;
    while depth < 600.0 {
        planet
            .apply(Brush {
                center: (ground - up * depth).to_array(),
                radius: 2.0,
                shape: BrushShape::Cube,
                op: BrushOp::Remove,
                material: 0,
                height: 0.0,
            })
            .unwrap();
        depth += 3.0;
    }
    let mut world = World::new(&gpu, &planet);
    let side = up.any_orthonormal_vector();
    for (depth, pitch) in [(590.0, -0.6), (290.0, 0.4)] {
        let view = View::new(
            ground - up * depth + side * 0.7,
            (side + up * pitch).normalize().as_vec3(),
            [256, 144],
        );
        let (compared, mismatched) = compare(&gpu, &mut world, &planet, &view);
        assert!(compared > 1000, "{depth} m: {compared} compared");
        assert_eq!(mismatched, 0, "{depth} m");
    }
}

/// A crater 3 km wide: on its floor.
#[test]
fn a_planet_scale_crater_renders_exactly_on_its_floor() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let ground = planet.surface_point(land(&planet, 5, 0.33, 0.52), 0.0);
    let up = planet.grid().up(ground);
    planet
        .apply(Brush {
            center: ground.to_array(),
            radius: 3_000.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    let eye = ground - up * (3_000.0 - 1.7);
    assert!(
        !planet.solid(planet.grid().locate(eye).0),
        "the eye is in the crater"
    );
    let forward = (up.any_orthonormal_vector() - up * 0.4)
        .normalize()
        .as_vec3();
    matches_exactly(&gpu, &planet, View::new(eye, forward, [256, 144]), 1000);
}

/// The whole planet is destructible: a 3 km world with its core hollowed
/// out, seen from inside the hollow. The rock between the hollow's wall and
/// the surface takes no chunks (each tile's solid gap), the hollow is air
/// under each tile's chunks. On a world this small the chunks' affine
/// frames bend from the grid enough that a few grazing rays meet a corner's
/// neighbour.
#[test]
fn a_hollowed_core_renders_exactly_from_inside() {
    let Some(gpu) = gpu() else { return };
    let recipe = PlanetRecipe {
        radius_m: 3_000.0,
        terrain: TerrainLayers::earth().heightfield().source(7),
        ..Default::default()
    };
    let mut planet = Planet::new(recipe).unwrap();
    planet
        .apply(Brush {
            center: [0.0; 3],
            radius: 1_500.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    let up = DVec3::new(0.3, 0.9, 0.2).normalize();
    let eye = up * (1_500.0 - 1.7);
    assert!(
        !planet.solid(planet.grid().locate(eye).0),
        "the eye is in the hollow"
    );
    assert!(
        planet.solid(planet.grid().locate(up * 1_500.5).0),
        "the hollow has a wall"
    );
    let forward = (up.any_orthonormal_vector() + up * 0.5)
        .normalize()
        .as_vec3();
    matches(&gpu, &planet, View::new(eye, forward, [256, 144]));
}

/// A hall 40 m across dug 300 m under the ground: from its floor, its walls
/// and ceiling render exactly, and the rock between it and the surface takes
/// no chunks (a solid gap in each tile over it).
#[test]
fn an_underground_hall_renders_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let ground = planet.surface_point(land(&planet, 2, 0.43, 0.57), 0.0);
    let up = planet.grid().up(ground);
    let centre = ground - up * 300.0;
    planet
        .apply(Brush {
            center: centre.to_array(),
            radius: 20.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    let eye = centre - up * 18.0;
    assert!(
        !planet.solid(planet.grid().locate(eye).0),
        "the eye is in the hall"
    );
    let forward = (up.any_orthonormal_vector() + up * 0.3)
        .normalize()
        .as_vec3();
    matches_exactly(&gpu, &planet, View::new(eye, forward, [256, 144]), 1000);
}

/// A building of single 0.1 m blocks and a field of single-block holes
/// (about 18 000 block edits).
#[test]
fn thousands_of_block_edits_render_exactly() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let grid = *planet.grid();
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let ground = planet.surface_point(dir, 0.0);
    let block = |p: DVec3, op: BrushOp| {
        let (cell, _) = grid.locate(p);
        Brush {
            center: grid.cell_center(cell).to_array(),
            radius: grid.voxel_size() * 0.5,
            shape: BrushShape::Cube,
            op,
            material: if op == BrushOp::Add {
                material::BRICK
            } else {
                0
            },
            height: 0.0,
        }
    };
    let at = |x: i32, y: i32, z: i32| {
        ground
            + side * (f64::from(x) * 0.1)
            + fwd * (f64::from(z) * 0.1 + 6.0)
            + up * (f64::from(y) * 0.1 + 0.55)
    };
    let mut count = 0;
    for y in 0..60 {
        for x in -20..20 {
            for z in -20..20 {
                let wall = x == -20 || x == 19 || z == -20 || z == 19;
                if y % 10 == 0 || (wall && (x + z + y) % 7 != 0) {
                    planet.apply(block(at(x, y, z), BrushOp::Add)).unwrap();
                    count += 1;
                }
            }
        }
    }
    for x in -40..40 {
        for z in -30..-10 {
            if (x * 3 + z) % 2 == 0 {
                planet
                    .apply(block(
                        ground + side * (f64::from(x) * 0.1) + fwd * (f64::from(z) * 0.1)
                            - up * 0.05,
                        BrushOp::Remove,
                    ))
                    .unwrap();
                count += 1;
            }
        }
    }
    assert!(count > 12_000, "{count}");
    let eye = ground - fwd * 6.0 + up * 14.0 - side * 3.0;
    let forward = ((ground + fwd * 6.0) - eye).normalize().as_vec3();
    matches_exactly(&gpu, &planet, View::new(eye, forward, [320, 180]), 400);
}

/// Blocks of many materials over ground with single cells painted: every
/// pixel whose CPU hit is an edited cell draws that cell's own material, not
/// its leaf's.
#[test]
fn edited_cells_draw_their_own_materials() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let grid = *planet.grid();
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let ground = planet.surface_point(dir, 0.0);
    let kinds = [
        material::BRICK,
        material::PLANKS,
        material::COBBLE,
        material::WOOD,
        material::CLAY,
        material::SNOW,
        material::BASALT,
        material::SAND,
    ];
    let mut edited = std::collections::HashMap::new();
    let mut cell_brush = |planet: &mut Planet, cell: Cell, op: BrushOp, material: u32| {
        planet
            .apply(Brush {
                center: grid.cell_center(cell).to_array(),
                radius: grid.voxel_size() * 0.5,
                shape: BrushShape::Cube,
                op,
                material,
                height: 0.0,
            })
            .unwrap();
        edited.insert(cell, material);
    };
    // A wall of blocks, each cell its own material.
    for y in 0..24i32 {
        for x in -16..16 {
            let (cell, _) = grid.locate(
                ground + side * (f64::from(x) * 0.1) + fwd * 4.0 + up * (f64::from(y) * 0.1 + 0.55),
            );
            cell_brush(
                &mut planet,
                cell,
                BrushOp::Add,
                kinds[((x * 7 + y * 3).rem_euclid(8)) as usize],
            );
        }
    }
    // Ground cells painted in a checkerboard in front of it.
    for x in -16..16 {
        for z in 0..30i32 {
            if (x + z) % 2 != 0 {
                continue;
            }
            let (c, _) = grid
                .locate(ground + side * (f64::from(x) * 0.1) + fwd * (f64::from(z) * 0.1 + 0.5));
            let cell = Cell::new(c.face, c.i, c.j, planet.column_top(c.face, c.i, c.j, 0) - 1);
            assert!(planet.solid(cell));
            cell_brush(
                &mut planet,
                cell,
                BrushOp::Paint,
                kinds[((x + z * 5).rem_euclid(8)) as usize],
            );
        }
    }
    let eye = ground - fwd * 2.0 + up * 3.0;
    let view = View::new(
        eye,
        ((ground + fwd * 3.0 + up * 0.8) - eye)
            .normalize()
            .as_vec3(),
        [320, 180],
    );
    let hits = World::new(&gpu, &planet).settle(&gpu, &planet, &view);
    let view_proj = view.view_proj(&planet);
    let (mut compared, mut wrong) = (0, 0);
    for y in 0..view.size[1] {
        for x in 0..view.size[0] {
            let Some(cpu) = planet.raycast(view.eye, view.pixel_dir(view_proj, x, y), 50.0) else {
                continue;
            };
            let Some(&want) = edited.get(&cpu.cell) else {
                continue;
            };
            assert_eq!(planet.material(cpu.cell), want);
            let Some(drawn) = hits[(y * view.size[0] + x) as usize]
                .filter(|h| h.level == 0 && h.cell == cpu.cell)
            else {
                continue;
            };
            compared += 1;
            if drawn.material != want {
                wrong += 1;
                if wrong < 8 {
                    eprintln!(
                        "pixel {x},{y}: {:?} drawn {}, edited {want}",
                        cpu.cell, drawn.material
                    );
                }
            }
        }
    }
    eprintln!("compared {compared}, wrong {wrong}");
    assert!(compared > 3000, "{compared} pixels compared");
    assert_eq!(wrong, 0);
}

/// Land columns picked for relief and mixed top materials (flat grassland
/// shows one material everywhere): the `steep` with the greatest top
/// difference over 16 cells, then the `mixed` with the most distinct top
/// materials around them, of `candidates` random ones.
fn varied_spots(
    planet: &Planet,
    candidates: usize,
    steep: usize,
    mixed: usize,
) -> Vec<(String, u8, i32, i32)> {
    let n = planet.grid().cells() as u64;
    let mut next = xorshift(0x9e37_79b9_7f4a_7c15);
    let (mut by_relief, mut by_kinds) = (Vec::new(), Vec::new());
    for _ in 0..candidates {
        let face = (next() % 6) as u8;
        let (i, j) = (
            (next() % (n - 64)) as i32 + 32,
            (next() % (n - 64)) as i32 + 32,
        );
        if planet.column_top(face, i, j, 0) <= 40 {
            continue;
        }
        let (mut lo, mut hi) = (i32::MAX, i32::MIN);
        let mut kinds = std::collections::HashSet::new();
        for a in (-8..8).step_by(4) {
            for b in (-8..8).step_by(4) {
                let top = planet.column_top(face, i + a, j + b, 0);
                lo = lo.min(top);
                hi = hi.max(top);
                kinds.insert(planet.material(Cell::new(face, i + a, j + b, top - 1)));
            }
        }
        by_relief.push((hi - lo, face, i, j));
        by_kinds.push((kinds.len() as i32 * 1000 + hi - lo, face, i, j));
    }
    by_relief.sort_by(|a, b| b.0.cmp(&a.0));
    by_kinds.sort_by(|a, b| b.0.cmp(&a.0));
    let mut spots: Vec<(String, u8, i32, i32)> = by_relief
        .iter()
        .take(steep)
        .map(|s| (format!("steep {} cells", s.0), s.1, s.2, s.3))
        .collect();
    for s in &by_kinds {
        if spots.len() < steep + mixed && !spots.iter().any(|t| (t.1, t.2, t.3) == (s.1, s.2, s.3))
        {
            spots.push((format!("{} materials", s.0 / 1000), s.1, s.2, s.3));
        }
    }
    spots
}

/// The surface material of the leaf (4x4x4 cells) holding natural cell `c`
/// when every one of its 16 columns tops in it with the same canonical
/// material: what the leaf's vote must draw on its top faces.
fn uniform_leaf_top(planet: &Planet, c: Cell) -> Option<u32> {
    let (li, lj, lk) = (
        c.i.div_euclid(4) * 4,
        c.j.div_euclid(4) * 4,
        c.k.div_euclid(4) * 4,
    );
    let mut material = None;
    for a in 0..4 {
        for b in 0..4 {
            let (i, j) = (li + a, lj + b);
            let top = planet.generated_top(c.face, i, j, 0);
            if !(lk..lk + 4).contains(&(top - 1)) {
                return None;
            }
            let m = planet.material(Cell::new(c.face, i, j, top - 1));
            if material.is_some_and(|x| x != m) {
                return None;
            }
            material = Some(m);
        }
    }
    material
}

/// The drawn material against `Planet::material` over a view's level-0
/// pixels, as (pixels, wrong): top faces of uniform leaves, other cells
/// under air, cells under solid ground. Each wrong (canonical, drawn) pair
/// is counted into `confusions`.
fn material_agreement(
    gpu: &Gpu,
    planet: &Planet,
    view: &View,
    confusions: &mut std::collections::HashMap<(u32, u32), usize>,
) -> [(usize, usize); 3] {
    let hits = World::new(gpu, planet).settle(gpu, planet, view);
    let view_proj = view.view_proj(planet);
    let near = 0.7 * planet.grid().voxel_size() / view.pixel_angle();
    let mut counts = [(0, 0); 3];
    // Uniform leaves by leaf (16 columns' tops each).
    let mut leaves = std::collections::HashMap::new();
    for y in 0..view.size[1] {
        for x in 0..view.size[0] {
            let Some(cpu) = planet
                .raycast(view.eye, view.pixel_dir(view_proj, x, y), near)
                .filter(|c| c.distance < near)
            else {
                continue;
            };
            let Some(drawn) = hits[(y * view.size[0] + x) as usize]
                .filter(|h| h.level == 0 && h.cell == cpu.cell)
            else {
                continue;
            };
            let c = cpu.cell;
            let own = planet.material(c);
            let covered = planet.solid(Cell::new(c.face, c.i, c.j, c.k + 1));
            // A top face: the ray stepped down k into the cell.
            let kind = if covered {
                2
            } else if drawn.entered == Some(4)
                && leaves
                    .entry((
                        c.face,
                        c.i.div_euclid(4),
                        c.j.div_euclid(4),
                        c.k.div_euclid(4),
                    ))
                    .or_insert_with(|| uniform_leaf_top(planet, c))
                    .is_some()
            {
                0
            } else {
                1
            };
            counts[kind].0 += 1;
            if drawn.material != own {
                counts[kind].1 += 1;
                *confusions.entry((own, drawn.material)).or_default() += 1;
            }
        }
    }
    counts
}

/// Natural ground draws `Planet::material` exactly where the leaf's
/// materials leave no choice: the top faces of leaves whose 16 columns all
/// top in them with one material (the generation kernel evaluates
/// `ground_material` at each cell's volume point, with
/// `Planet::material_slope`).
#[test]
fn uniform_leaves_draw_canonical_materials() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *planet.grid();
    let mut confusions = std::collections::HashMap::new();
    let (mut compared, mut wrong) = (0, 0);
    for (label, face, i, j) in varied_spots(&planet, 600, 2, 1) {
        let dir = grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        let eye = planet.surface_point(dir, 1.7);
        let up = eye.normalize();
        let view = View::new(
            eye,
            (up.any_orthonormal_vector() - up * 0.35)
                .normalize()
                .as_vec3(),
            [320, 180],
        );
        let [uniform, ..] = material_agreement(&gpu, &planet, &view, &mut confusions);
        eprintln!(
            "{label}: {} of {} uniform-leaf top pixels wrong",
            uniform.1, uniform.0
        );
        compared += uniform.0;
        wrong += uniform.1;
    }
    assert!(compared > 2000, "{compared} pixels compared");
    assert_eq!(wrong, 0, "{confusions:?}");
}

/// A pit's walls and floor draw `Planet::material` exactly: chunks under
/// edits give exposed cells their own natural materials where their leaf's
/// pair would draw them wrong (soil a cell deep drawn as rock).
#[test]
fn dug_walls_draw_canonical_materials() {
    let Some(gpu) = gpu() else { return };
    let natural = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *natural.grid();
    let radius = 2.5;
    let (mut compared, mut wrong, mut pages) = (0, 0, 0);
    let mut confusions: std::collections::HashMap<(u32, u32), usize> =
        std::collections::HashMap::new();
    for (label, face, i, j) in varied_spots(&natural, 600, 2, 1) {
        let dir = grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        let eye = natural.surface_point(dir, 1.7);
        let up = eye.normalize();
        let side = up.any_orthonormal_vector();
        let centre = natural.surface_point(dir, 0.0) + side * 4.0;
        let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
        planet
            .apply(Brush {
                center: centre.to_array(),
                radius,
                shape: BrushShape::Sphere,
                op: BrushOp::Remove,
                material: 0,
                height: 0.0,
            })
            .unwrap();
        let view = View::new(eye, (side - up * 0.35).normalize().as_vec3(), [320, 180]);
        let mut world = World::new(&gpu, &planet);
        let hits = world.settle(&gpu, &planet, &view);
        pages += world.frame(&gpu, &planet, &view, None).detail_pages.0;
        let view_proj = view.view_proj(&planet);
        let near = 0.7 * grid.voxel_size() / view.pixel_angle();
        let (mut n, mut w) = (0, 0);
        for y in 0..view.size[1] {
            for x in 0..view.size[0] {
                let Some(cpu) = planet
                    .raycast(view.eye, view.pixel_dir(view_proj, x, y), near)
                    .filter(|c| c.distance < near)
                else {
                    continue;
                };
                // The pit's cells: within a cell of its sphere.
                if (grid.cell_center(cpu.cell) - centre).length() > radius + 2.0 * grid.voxel_size()
                {
                    continue;
                }
                let Some(drawn) = hits[(y * view.size[0] + x) as usize]
                    .filter(|h| h.level == 0 && h.cell == cpu.cell)
                else {
                    continue;
                };
                let own = planet.material(cpu.cell);
                n += 1;
                if drawn.material != own {
                    w += 1;
                    *confusions.entry((own, drawn.material)).or_default() += 1;
                }
            }
        }
        eprintln!("{label}: {w} of {n} pit pixels wrong");
        compared += n;
        wrong += w;
    }
    eprintln!("detail pages in use, summed over views: {pages}");
    assert!(compared > 5000, "{compared} pixels compared");
    assert_eq!(wrong, 0, "{confusions:?}");
}

/// How often unedited voxels draw another material than their own
/// (`Planet::material`): over level-0 pixels of views at spots picked for
/// relief and mixed materials, from 1.7 m and 6 m, then with a pit dug in
/// front (a removal sets no materials: natural cells in its walls). Split
/// by what decides the drawn material: top faces of uniform leaves (exact),
/// other cells under air (the leaf's surface vote), cells under ground
/// (the leaf's subsurface byte). A report, not a gate:
/// `cargo test --release --test chunks natural_cells -- --ignored --nocapture`.
#[test]
#[ignore]
fn natural_cells_against_the_leaf_rule() {
    let Some(gpu) = gpu() else { return };
    let natural = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *natural.grid();
    let name = |m: u32| {
        if m == 0 || m as usize > material::NAMES.len() {
            format!("#{m}")
        } else {
            material::NAMES[m as usize - 1].to_string()
        }
    };
    let pct = |(n, w): (usize, usize)| 100.0 * w as f64 / n.max(1) as f64;
    let sum = |c: &[(usize, usize); 3]| c.iter().fold((0, 0), |a, c| (a.0 + c.0, a.1 + c.1));
    let mut confusions = std::collections::HashMap::new();
    let mut all = [(0, 0); 3];
    for (label, face, i, j) in varied_spots(&natural, 1500, 4, 4) {
        let dir = grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        for (lift, pit) in [(1.7, false), (6.0, false), (1.7, true)] {
            let eye = natural.surface_point(dir, lift);
            let up = eye.normalize();
            let side = up.any_orthonormal_vector();
            let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
            if pit {
                let ground = natural.surface_point(dir, 0.0) + side * 4.0;
                planet
                    .apply(Brush {
                        center: ground.to_array(),
                        radius: 2.5,
                        shape: BrushShape::Sphere,
                        op: BrushOp::Remove,
                        material: 0,
                        height: 0.0,
                    })
                    .unwrap();
            }
            let view = View::new(
                eye,
                (side - up * if lift > 2.0 { 0.6 } else { 0.35 })
                    .normalize()
                    .as_vec3(),
                [320, 180],
            );
            let counts = material_agreement(&gpu, &planet, &view, &mut confusions);
            let total = sum(&counts);
            eprintln!(
                "{label} at {lift} m{}: {}/{} pixels wrong ({:.1}%); uniform tops {:.1}% of {}, other surface {:.1}% of {}, under ground {:.1}% of {}",
                if pit { ", pit" } else { "" },
                total.1,
                total.0,
                pct(total),
                pct(counts[0]),
                counts[0].0,
                pct(counts[1]),
                counts[1].0,
                pct(counts[2]),
                counts[2].0
            );
            for (a, c) in all.iter_mut().zip(counts) {
                *a = (a.0 + c.0, a.1 + c.1);
            }
        }
    }
    let total = sum(&all);
    eprintln!(
        "ALL: {}/{} pixels wrong ({:.2}%); uniform tops {:.2}% of {}, other surface {:.2}% of {}, under ground {:.2}% of {}",
        total.1,
        total.0,
        pct(total),
        pct(all[0]),
        all[0].0,
        pct(all[1]),
        all[1].0,
        pct(all[2]),
        all[2].0
    );
    let mut top: Vec<_> = confusions.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    for ((own, drawn), count) in top.iter().take(12) {
        eprintln!(
            "  own {} drawn {}: {count} pixels ({:.2}%)",
            name(*own),
            name(*drawn),
            100.0 * *count as f64 / total.0.max(1) as f64
        );
    }
}

/// What one more edit costs a running world as the history grows: the CPU
/// of `sync_edits` and of the frame that regenerates what it marked, after
/// `n` small digs (sealed into the baked store) and after `n` large ones
/// (kept analytic). A report, not a gate:
/// `cargo test --release --test chunks edit_cost -- --ignored --nocapture`.
#[test]
#[ignore]
fn edit_cost_against_history() {
    let Some(gpu) = gpu() else { return };
    let base = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&base, 1, 0.52, 0.48);
    let ground = base.surface_point(dir, 0.0);
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let view = ground_view(&base, dir, [320, 180]);
    for (label, radius, spread) in [
        ("small digs (0.4 m)", 0.4, 0.5),
        ("large digs (8 m)", 8.0, 9.0),
    ] {
        for n in [0usize, 100, 1000, 10000] {
            let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
            let mut next = xorshift(0x2545_f491_4f6c_dd1d);
            let mut dig = |planet: &mut Planet| {
                let a = (next() % 2001) as f64 / 1000.0 - 1.0;
                let b = (next() % 2001) as f64 / 1000.0 - 1.0;
                let centre =
                    ground + side * (a * 40.0 * spread) + fwd * (b * 40.0 * spread) + up * 6.0;
                planet
                    .apply(Brush {
                        center: centre.to_array(),
                        radius,
                        shape: BrushShape::Sphere,
                        op: BrushOp::Remove,
                        material: 0,
                        height: 0.0,
                    })
                    .unwrap();
            };
            for _ in 0..n {
                dig(&mut planet);
            }
            let mut world = World::new(&gpu, &planet);
            world.settle(&gpu, &planet, &view);
            // Ten more edits, each synced and its frame run: the update's
            // CPU, the generation's GPU (`FrameTimings`), the chunks it
            // generated.
            let (mut sync_ms, mut frame_ms, mut update_ms, mut gen_ms, mut chunks) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            world.chunks.take_timings();
            for _ in 0..10 {
                dig(&mut planet);
                let t = std::time::Instant::now();
                world.sync(&gpu, &planet);
                sync_ms.push(t.elapsed().as_secs_f64() * 1e3);
                let t = std::time::Instant::now();
                let stats = world.frame(&gpu, &planet, &view, None);
                frame_ms.push(t.elapsed().as_secs_f64() * 1e3);
                update_ms.push(stats.cpu_ms.iter().sum::<f64>());
                chunks.push(stats.generated_chunks as f64);
                // The frame's GPU times come back with the next frame's.
                world.frame(&gpu, &planet, &view, None);
                gen_ms.push(
                    world
                        .chunks
                        .take_timings()
                        .iter()
                        .map(|t| t.generate)
                        .sum::<f64>(),
                );
            }
            for v in [
                &mut sync_ms,
                &mut frame_ms,
                &mut update_ms,
                &mut gen_ms,
                &mut chunks,
            ] {
                v.sort_by(f64::total_cmp);
            }
            eprintln!("EDITPARTS history {n}: update CPU p50 {:.2} ms, generation GPU p50 {:.2} ms, chunks p50 {}", update_ms[5], gen_ms[5], chunks[5]);
            eprintln!(
                "EDITCOST {label}, history {n}: analytic brushes {}, sealed {}; sync p50 {:.2} max {:.2} ms; frame p50 {:.2} max {:.2} ms",
                planet.edits().len() - planet.edits().sealed_len(),
                planet.edits().sealed_len(),
                sync_ms[5],
                sync_ms[9],
                frame_ms[5],
                frame_ms[9]
            );
        }
    }
}

/// Settle `view`, summing the edited chunks restored and generated and the
/// generation's GPU milliseconds; its hits.
fn settle_counting(
    gpu: &Gpu,
    world: &mut World,
    planet: &Planet,
    view: &View,
) -> (Vec<Option<PickHit>>, (usize, usize), f64, u64) {
    let started = world.frame;
    let (mut restored, mut generated, mut gpu_ms) = (0, 0, 0.0);
    world.chunks.take_timings();
    loop {
        let stats = world.frame(gpu, planet, view, None);
        restored += stats.restored_chunks;
        generated += stats.generated_edited;

        gpu_ms += world
            .chunks
            .take_timings()
            .iter()
            .map(|t| t.generate)
            .sum::<f64>();
        if world.chunks.settled() {
            break;
        }
        assert!(
            world.frame - started < 4000,
            "residency did not settle: {stats:?}"
        );
    }
    (
        world.chunks.read_hits(),
        (restored, generated),
        gpu_ms,
        world.frame - started,
    )
}

/// Edited chunks the plan evicts are kept (packed, read back) and come
/// back by copy: a dig site under 2000 large digs, settled, left for a far
/// view (a world of 9000 chunk slots evicts it), then seen again, draws
/// every pixel as it did, its edited chunks restored, not generated again
/// from the field and the history (against a world that keeps nothing).
/// A site under 2000 digs 8 m across, a few metres ahead of the ground at
/// a land direction: the planet, the direction, and its frame (ground, up,
/// forward).
fn dug_site() -> (Planet, DVec3, [DVec3; 3]) {
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let ground = planet.surface_point(dir, 0.0);
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let mut next = xorshift(0x5eed_1234_9876_abcd);
    for _ in 0..2000 {
        let a = (next() % 2001) as f64 / 1000.0 - 1.0;
        let b = (next() % 2001) as f64 / 1000.0 - 1.0;
        let centre = ground + side * (a * 20.0) + fwd * (b * 20.0 + 12.0) + up * 5.0;
        planet
            .apply(Brush {
                center: centre.to_array(),
                radius: 8.0,
                shape: BrushShape::Sphere,
                op: BrushOp::Remove,
                material: 0,
                height: 0.0,
            })
            .unwrap();
    }
    (planet, dir, [ground, up, fwd])
}

#[test]
fn edited_chunks_come_back_by_copy() {
    let Some(gpu) = gpu() else { return };
    let (mut planet, dir, [ground, up, fwd]) = dug_site();
    let here = ground_view(&planet, dir, [320, 180]);
    let far = ground_view(&planet, land(&planet, 4, 0.3, 0.7), [320, 180]);
    let run = |planet: &Planet, kept_bytes: usize| {
        let capacity = Capacity {
            chunks: 9000,
            kept_bytes,
            ..TEST_CAPACITY
        };
        let mut world = World::with_capacity(gpu, planet, capacity);
        let (first, ..) = settle_counting(&gpu, &mut world, planet, &here);
        settle_counting(&gpu, &mut world, planet, &far);
        let stats = world.frame(&gpu, planet, &far, None);
        let (again, (restored, generated), gpu_ms, frames) =
            settle_counting(&gpu, &mut world, planet, &here);
        let same = first
            .iter()
            .zip(&again)
            .filter(|(a, b)| {
                a.map(|h| (h.cell, h.level, h.material, h.distance.to_bits()))
                    == b.map(|h| (h.cell, h.level, h.material, h.distance.to_bits()))
            })
            .count();
        eprintln!("KEPT {kept_bytes} bytes: {} kept at the far view ({} KiB); back: edited chunks {restored} restored, {generated} generated; {gpu_ms:.2} ms of generation GPU over {frames} frames; {same}/{} pixels as before", stats.kept_chunks, stats.kept_bytes / 1024, first.len());
        (restored, generated, same, first.len(), world)
    };
    let (none, regenerated, ..) = run(&planet, 0);
    let (restored, generated, same, pixels, mut world) = run(&planet, TEST_CAPACITY.kept_bytes);
    // An edit over restored chunks shows exactly; left and seen again, the
    // chunks it changed come back with it.
    planet
        .apply(Brush {
            center: (ground + fwd * 12.0 - up * 2.0).to_array(),
            radius: 3.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Add,
            material: material::BRICK,
            height: 0.0,
        })
        .unwrap();
    world.sync(&gpu, &planet);
    let (compared, mismatched) = compare(&gpu, &mut world, &planet, &here);
    assert!(
        compared > 1000 && mismatched * 1000 <= compared,
        "after the edit: {mismatched}/{compared}"
    );
    settle_counting(&gpu, &mut world, &planet, &far);
    let (_, (restored_again, _), ..) = settle_counting(&gpu, &mut world, &planet, &here);
    let (compared_again, mismatched_again) = compare(&gpu, &mut world, &planet, &here);
    eprintln!("KEPT after an edit: {mismatched}/{compared}; left and back, {restored_again} restored, {mismatched_again}/{compared_again}");
    assert!(
        compared_again > 1000 && mismatched_again * 1000 <= compared_again,
        "back after the edit: {mismatched_again}/{compared_again}"
    );

    // The edited chunks evicted come back restored (those evicted before
    // their pack came back, or past a frame's packs, generate again); a
    // world that keeps nothing generates them all.
    assert!(
        restored > 50 && generated * 4 < restored,
        "edited chunks: {restored} restored, {generated} generated"
    );
    assert!(
        none == 0 && regenerated > 50,
        "keeping nothing: {none} restored, {regenerated} generated"
    );
    assert_eq!(same, pixels, "every pixel as before");
}

/// Without slot pressure, chunks the cut left are released once unvisited
/// for `release_frames`: the dug site left for a far view goes (more edited
/// chunks a plan than a frame packs: the rest wait resident for later
/// frames' packs), residency falls to the far view's own, and the site
/// seen again comes back by copy, every pixel as before.
#[test]
fn edited_chunks_released_by_age_come_back_by_copy() {
    let Some(gpu) = gpu() else { return };
    let (planet, dir, _) = dug_site();
    let here = ground_view(&planet, dir, [320, 180]);
    let far = ground_view(&planet, land(&planet, 4, 0.3, 0.7), [320, 180]);
    let release = 300;
    let mut config = here.config();
    config.plan.release_frames = release;
    let mut world = World::new(&gpu, &planet);
    let settle = |world: &mut World,
                  view: &View,
                  config: &ResidencyConfig|
     -> (Vec<Option<PickHit>>, (usize, usize)) {
        let started = world.frame;
        let (mut restored, mut generated) = (0, 0);
        loop {
            let stats = world.frame_with(&gpu, &planet, view, config);
            restored += stats.restored_chunks;
            generated += stats.generated_edited;
            if world.chunks.settled() {
                return (world.chunks.read_hits(), (restored, generated));
            }
            assert!(
                world.frame - started < 4000,
                "residency did not settle: {stats:?}"
            );
        }
    };
    let (first, _) = settle(&mut world, &here, &config);
    let (_, _) = settle(&mut world, &far, &config);
    let settled_far = world.frame(&gpu, &planet, &far, None).resident_chunks;
    let (mut evicted, mut deferred, mut stats) = (0, 0, None);
    for _ in 0..release + 200 {
        let s = world.frame_with(&gpu, &planet, &far, &config);
        evicted += s.evicted_chunks;
        deferred += s.evictions_deferred;
        stats = Some(s);
    }
    let stats = stats.unwrap();
    let mut fresh = World::new(&gpu, &planet);
    let (_, _) = settle(&mut fresh, &far, &config);
    let wanted = fresh.frame(&gpu, &planet, &far, None).resident_chunks;
    eprintln!("RELEASE far view: {settled_far} resident settled, {} after {} frames ({evicted} evicted, {deferred} deferred to later packs), {} kept; a fresh world {wanted}", stats.resident_chunks, release + 200, stats.kept_chunks);
    let (again, (restored, generated)) = settle(&mut world, &here, &config);
    let same = first
        .iter()
        .zip(&again)
        .filter(|(a, b)| {
            a.map(|h| (h.cell, h.level, h.material, h.distance.to_bits()))
                == b.map(|h| (h.cell, h.level, h.material, h.distance.to_bits()))
        })
        .count();
    eprintln!("RELEASE back: edited chunks {restored} restored, {generated} generated; {same}/{} pixels as before", first.len());
    assert!(
        stats.resident_chunks * 5 <= wanted * 6,
        "{} resident after release, a fresh world {wanted}",
        stats.resident_chunks
    );
    assert!(
        deferred > 0 && stats.kept_chunks > 200,
        "edited chunks kept: {} ({deferred} deferred)",
        stats.kept_chunks
    );
    // Restores count every time a kept chunk comes back: 624 to 1323 over
    // runs (fewer with other tests on the GPU), the generated about 20.
    assert!(
        restored > 200 && generated * 4 < restored,
        "edited chunks: {restored} restored, {generated} generated"
    );
    assert_eq!(same, first.len(), "every pixel as before");
}

/// A saved world's edited chunks come back with it: packed (those resident
/// too, `keep_resident`) and saved as bytes beside the edit snapshot, a
/// world made from the loaded snapshot restores them by copy instead of
/// generating them from the field and the history, and draws every pixel
/// as the saved world did. Kept chunks of another history, or damaged, are
/// refused.
#[test]
fn a_loaded_world_restores_its_edited_chunks() {
    let Some(gpu) = gpu() else { return };
    let (planet, dir, _) = dug_site();
    let here = ground_view(&planet, dir, [320, 180]);
    let mut saved = World::new(&gpu, &planet);
    let (hits, ..) = settle_counting(&gpu, &mut saved, &planet, &here);
    saved.chunks.keep_resident();
    for _ in 0..400 {
        if saved.chunks.keeping() == 0 {
            break;
        }
        saved.frame(&gpu, &planet, &here, None);
    }
    assert_eq!(saved.chunks.keeping(), 0, "packing the resident chunks");
    let bytes = saved.chunks.kept_chunks(VolumeId(0), &planet).to_bytes();
    let snapshot = planet.snapshot();
    let loaded = Planet::from_snapshot(PlanetRecipe::default(), &snapshot).unwrap();
    let chunks = KeptChunks::from_bytes(&bytes).unwrap();
    let mut world = World::new(&gpu, &loaded);
    let kept = world
        .chunks
        .load_kept(VolumeId(0), &loaded, &chunks)
        .unwrap();
    let (again, (restored, generated), keeping_ms, _) =
        settle_counting(&gpu, &mut world, &loaded, &here);
    let mut cold = World::new(&gpu, &loaded);
    let (_, (_, regenerated), generating_ms, _) = settle_counting(&gpu, &mut cold, &loaded, &here);
    let same = hits
        .iter()
        .zip(&again)
        .filter(|(a, b)| {
            a.map(|h| (h.cell, h.level, h.material, h.distance.to_bits()))
                == b.map(|h| (h.cell, h.level, h.material, h.distance.to_bits()))
        })
        .count();
    eprintln!(
        "LOADED {kept} kept chunks, {} KiB of them besides the snapshot's {} KiB; edited chunks {restored} restored, {generated} generated ({regenerated} without them); {keeping_ms:.1} ms of generation GPU ({generating_ms:.1} without); {same}/{} pixels as saved",
        bytes.len() / 1024,
        snapshot.len() / 1024,
        hits.len()
    );
    assert!(
        kept > 100 && restored > 100,
        "{kept} kept, {restored} restored"
    );
    assert!(
        generated * 4 < restored && regenerated > 100,
        "edited chunks: {generated} generated with the kept ones, {regenerated} without"
    );
    assert_eq!(same, hits.len(), "every pixel as saved");
    // Another history: refused.
    let mut other = Planet::from_snapshot(PlanetRecipe::default(), &snapshot).unwrap();
    other
        .apply(Brush {
            center: other.surface_point(dir, 0.0).to_array(),
            radius: 1.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    assert!(World::new(&gpu, &other)
        .chunks
        .load_kept(VolumeId(0), &other, &chunks)
        .is_err());
    // Damaged bytes: refused.
    let mut damaged = bytes.clone();
    damaged[40] ^= 1;
    assert!(KeptChunks::from_bytes(&damaged).is_err());
}

/// Edits sealed under a running world: a crater and a mound too large to
/// bake, blocks baked between them, then undo across seals. After each
/// change the kept world matches CPU ray casts exactly.
#[test]
fn sealed_edits_render_exactly_under_a_running_world() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let grid = *planet.grid();
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let ground = planet.surface_point(dir, 0.0);
    let block = |p: DVec3| {
        let (cell, _) = grid.locate(p);
        Brush {
            center: grid.cell_center(cell).to_array(),
            radius: grid.voxel_size() * 0.5,
            shape: BrushShape::Cube,
            op: BrushOp::Add,
            material: material::BRICK,
            height: 0.0,
        }
    };
    let layer = |planet: &mut Planet, height: f64, every: i32| {
        for x in -12i32..12 {
            for z in -12..12 {
                if (x + z).rem_euclid(every) == 0 {
                    planet
                        .apply(block(
                            ground
                                + side * (f64::from(x) * 0.1)
                                + fwd * (f64::from(z) * 0.1)
                                + up * height,
                        ))
                        .unwrap();
                }
            }
        }
    };
    let eye = ground + up * 9.0 - fwd * 7.0;
    let view = View::new(
        eye,
        ((ground - up * 4.0) - eye).normalize().as_vec3(),
        [320, 180],
    );
    let mut world = World::new(&gpu, &planet);
    let mut check = |planet: &Planet| {
        world.sync(&gpu, planet);
        let (compared, mismatched) = compare(&gpu, &mut world, planet, &view);
        assert!(compared > 400, "{compared} compared");
        assert_eq!(mismatched, 0, "{} edits", planet.edits().len());
    };
    check(&planet);
    planet
        .apply(Brush {
            center: (ground - up).to_array(),
            radius: 6.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    layer(&mut planet, -5.0, 3);
    check(&planet);
    planet
        .apply(Brush {
            center: (ground - up * 5.5 + side * 2.5).to_array(),
            radius: 4.0,
            shape: BrushShape::Cube,
            op: BrushOp::Add,
            material: material::BRICK,
            height: 0.0,
        })
        .unwrap();
    layer(&mut planet, -3.5, 2);
    layer(&mut planet, -2.5, 2);
    assert_eq!(planet.edits().large.len(), 2, "both large brushes sealed");
    assert!(!planet.edits().baked.is_empty());
    check(&planet);
    // Undo across seals: their bricks and large brushes come back undone.
    let sealed = planet.edits().sealed_len();
    for _ in 0..300 {
        planet.undo().unwrap();
    }
    assert!(planet.edits().sealed_len() < sealed);
    check(&planet);
}

/// From orbit (300 km up) every pixel is drawn as the planet's shape
/// allows: a hit wherever the ray meets the sphere 20 km below the datum,
/// sky wherever it misses the sphere 20 km above it (no relief reaches
/// either).
/// The editor's sculpt strokes on the Moon (a drag lays a ball 120 m across
/// every few metres, built then dug) settle, seen from 300 m: residency
/// stops asking for chunks a few hundred frames after the last stroke. Two
/// such strokes once kept the Pulsar editor generating for 20 000 frames.
#[test]
fn large_sculpt_strokes_settle() {
    let Some(gpu) = gpu() else { return };
    let voxel = 0.1;
    let mut planet = Planet::new(PlanetRecipe {
        radius_m: 1_737_400.0,
        voxel_size_m: voxel,
        terrain: TerrainLayers::moon().source(75),
        features: helio_pass_voxel_planet::feature_recipe::FeatureRecipe::moon().grow(voxel, 75),
        ..Default::default()
    })
    .unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let ground = planet.surface_point(dir, 0.0);
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let view = View::new(
        ground + up * 300.0 - fwd * 300.0,
        (fwd - up * 0.7).normalize().as_vec3(),
        [480, 270],
    );
    let mut world = World::new(&gpu, &planet);
    world.jitter = Some(0);
    world.settle(&gpu, &planet, &view);
    for (op, material) in [(BrushOp::Add, material::COBBLE), (BrushOp::Remove, 0)] {
        for n in 0..40 {
            let along = (n as f64 - 20.0) * 4.0;
            let centre = planet.surface_point((ground + side * along).normalize(), 0.0);
            planet
                .apply(Brush {
                    center: centre.to_array(),
                    radius: 60.0,
                    shape: BrushShape::Sphere,
                    op,
                    material,
                    height: 0.0,
                })
                .unwrap();
            // A brush a frame, as a drag lays them.
            world.sync(&gpu, &planet);
            world.frame(&gpu, &planet, &view, None);
        }
    }
    let started = world.frame;
    let mut cpu = Vec::new();
    loop {
        let t = std::time::Instant::now();
        let stats = world.frame(&gpu, &planet, &view, None);
        cpu.push(t.elapsed().as_secs_f64() * 1e3);
        if world.chunks.settled() {
            break;
        }
        if (world.frame - started) % 250 == 0 {
            eprintln!(
                "frame {}: {} ms, {}",
                world.frame - started,
                cpu.last().unwrap(),
                world.chunks.settle_report()
            );
            eprintln!("  {stats:?}");
        }
        assert!(
            world.frame - started < 3000,
            "sculpted Moon did not settle: {stats:?}"
        );
    }
    cpu.sort_by(f64::total_cmp);
    eprintln!(
        "settled {} frames after the strokes; frame ms p50 {:.1} max {:.1}",
        world.frame - started,
        cpu[cpu.len() / 2],
        cpu[cpu.len() - 1]
    );
}

/// What a burst of sculpt strokes looks like while it happens, under a
/// host's generation budget (a report: `--ignored --nocapture`). Every
/// frame's hits (every fourth pixel each way) are kept, then judged
/// against the burst's own settled frame (CPU ray casts of the Moon cost
/// a third of a second each): coarse pixels (drawn three levels or more
/// coarser than their distance wants), wrong ones after the last stroke (off by
/// more than a metre and 2%, or a hit where there is none or none where
/// there is one), pixels drawing material 0 (magenta), and the frames
/// after the last stroke until the view is right again (`BURST_FRAMES=<dir>`
/// saves some frames).
#[test]
#[ignore]
fn sculpt_bursts_while_they_happen() {
    let Some(gpu) = gpu() else { return };
    let voxel = 0.1;
    let mut planet = Planet::new(PlanetRecipe {
        radius_m: 1_737_400.0,
        voxel_size_m: voxel,
        terrain: TerrainLayers::moon().source(75),
        features: helio_pass_voxel_planet::feature_recipe::FeatureRecipe::moon().grow(voxel, 75),
        ..Default::default()
    })
    .unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let ground = planet.surface_point(dir, 0.0);
    let up = dir.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let view = View::new(
        ground + up * 120.0 - fwd * 160.0,
        (fwd - up * 0.7).normalize().as_vec3(),
        [480, 270],
    );
    let mut config = ResidencyConfig::default();
    config.plan.pixel_angle = view.pixel_angle();
    let mut world = World::new(&gpu, &planet);
    world.jitter = Some(0);
    world.settle(&gpu, &planet, &view);
    let pixels: Vec<usize> = (0..view.size[1])
        .step_by(4)
        .flat_map(|y| {
            (0..view.size[0])
                .step_by(4)
                .map(move |x| (y * view.size[0] + x) as usize)
        })
        .collect();
    let sample = |world: &World| -> Vec<Option<PickHit>> {
        let hits = world.chunks.read_hits();
        pixels.iter().map(|&p| hits[p]).collect()
    };
    let save = |world: &World, name: String| {
        let Some(dir) = std::env::var_os("BURST_FRAMES") else {
            return;
        };
        let pixels = read_rgba8(gpu, &world.chunks.targets().unwrap().colour);
        image::RgbaImage::from_raw(
            view.size[0],
            view.size[1],
            // Brightened: the plain colour is dim.
            pixels
                .into_iter()
                .flat_map(|p| {
                    [p[0], p[1], p[2]]
                        .map(|c| c.saturating_mul(3))
                        .into_iter()
                        .chain([255])
                })
                .collect(),
        )
        .unwrap()
        .save(std::path::Path::new(&dir).join(format!("{name}.png")))
        .unwrap();
    };
    let wrong = |a: &Option<PickHit>, b: &Option<PickHit>| match (a, b) {
        (Some(a), Some(b)) => (a.distance - b.distance).abs() > 1.0f64.max(b.distance * 0.02),
        (None, None) => false,
        _ => true,
    };
    let bursts: [(&str, f64, BrushOp, u32, f64); 4] = [
        ("dig_3m", 3.0, BrushOp::Remove, 0, 1.0),
        ("dig_12m", 12.0, BrushOp::Remove, 0, 4.0),
        ("build_60m", 60.0, BrushOp::Add, material::COBBLE, 4.0),
        ("dig_60m", 60.0, BrushOp::Remove, 0, 4.0),
    ];
    let strokes = 40;
    for (label, radius, op, material, step) in bursts {
        let mut frames: Vec<Vec<Option<PickHit>>> = Vec::new();
        let mut cpu = Vec::new();
        let mut frame = 0;
        loop {
            if frame < strokes {
                let along = (frame as f64 - 20.0) * step;
                let centre = planet.surface_point((ground + side * along).normalize(), 0.0);
                planet
                    .apply(Brush {
                        center: centre.to_array(),
                        radius,
                        shape: BrushShape::Sphere,
                        op,
                        material,
                        height: 0.0,
                    })
                    .unwrap();
                world.sync(&gpu, &planet);
            }
            let t = std::time::Instant::now();
            world.frame_with(&gpu, &planet, &view, &config);
            cpu.push(t.elapsed().as_secs_f64() * 1e3);
            frames.push(sample(&world));
            if [4, 12, 24, 39, strokes + 8].contains(&frame) {
                save(&world, format!("{label}_{frame:03}"));
            }
            frame += 1;
            if frame >= strokes && (world.chunks.settled() || frame > strokes + 900) {
                break;
            }
        }
        let settled = frames.last().unwrap().clone();
        save(&world, format!("{label}_settled"));
        let (mut coarse_sum, mut coarse_max, mut magenta_sum) = (0, 0, 0);
        let (mut wrong_after, mut right_after) = (0, None);
        for (n, hits) in frames.iter().enumerate() {
            // Three levels or more coarser than its distance wants.
            let coarse = hits
                .iter()
                .flatten()
                .filter(|h| {
                    let wanted = (h.distance * view.pixel_angle() / voxel)
                        .log2()
                        .floor()
                        .max(0.0);
                    f64::from(h.level) >= wanted + 3.0
                })
                .count();
            coarse_sum += coarse;
            coarse_max = coarse_max.max(coarse);
            magenta_sum += hits.iter().flatten().filter(|h| h.material == 0).count();
            if n >= strokes {
                let w = hits
                    .iter()
                    .zip(&settled)
                    .filter(|(a, b)| wrong(a, b))
                    .count();
                wrong_after += w;
                if right_after.is_none() && w * 200 <= pixels.len() {
                    right_after = Some(n - strokes);
                }
            }
        }
        cpu.sort_by(f64::total_cmp);
        eprintln!(
            "BURST {label}: {frame} frames ({} after the strokes); coarse pixel-frames {coarse_sum} (most a frame {coarse_max} of {}), magenta pixel-frames {magenta_sum}; after the strokes wrong pixel-frames {wrong_after}, right {:?} frames after; frame ms p50 {:.1} p95 {:.1} max {:.1}",
            frame - strokes,
            pixels.len(),
            right_after,
            cpu[cpu.len() / 2],
            cpu[cpu.len() * 95 / 100],
            cpu[cpu.len() - 1]
        );
    }
}

/// The Moon's pole at seed 75 (Pulsar's Moon example), 300 m over the
/// ground. Capsule parts once held cells far outside their box (the
/// distance test overflowed i32 there): the CPU met a boulder floating
/// 290 m over its anchor that the GPU, bounding stamps by their boxes,
/// never drew, and every shadow ray toward it came back black.
#[test]
fn moon_pole_view_matches_cpu_ray_casts() {
    let Some(gpu) = gpu() else { return };
    let voxel = 0.1;
    let planet = Planet::new(PlanetRecipe {
        radius_m: 1_737_400.0,
        voxel_size_m: voxel,
        terrain: TerrainLayers::moon().source(75),
        features: helio_pass_voxel_planet::feature_recipe::FeatureRecipe::moon().grow(voxel, 75),
        ..Default::default()
    })
    .unwrap();
    let eye = DVec3::new(0.0, 1_738_104.0, 0.0);
    let up = eye.length() - planet.surface_point(eye.normalize(), 0.0).length();
    let view = View::new(
        eye,
        Vec3::new(-0.07329183, -0.57357645, 0.81586665),
        [320, 180],
    );
    // The top-left pixel's ray once met the floating boulder 9 m away.
    let wall = planet
        .raycast(eye, view.pixel_dir(view.view_proj(&planet), 0, 0), 2000.0)
        .expect("ground");
    assert!(
        wall.distance > up * 0.5,
        "the top-left ray met {:?} {:.1} m away, {up:.0} m over the ground",
        wall.cell,
        wall.distance
    );
    // Nothing here is near enough for level 0 (`compare`): the drawn
    // levels must meet the exact ground within a few of their cells.
    let hits = World::new(&gpu, &planet).settle(&gpu, &planet, &view);
    let view_proj = view.view_proj(&planet);
    let (mut compared, mut mismatched) = (0, 0);
    for y in (0..view.size[1]).step_by(12) {
        for x in (0..view.size[0]).step_by(12) {
            let Some(cpu) = planet.raycast(eye, view.pixel_dir(view_proj, x, y), 2000.0) else {
                continue;
            };
            compared += 1;
            let drawn = hits[(y * view.size[0] + x) as usize];
            let near = drawn.is_some_and(|h| {
                (h.distance - cpu.distance).abs()
                    <= 8.0 * voxel * f64::from(1u32 << h.level) + cpu.distance * 0.02
            });
            if !near {
                mismatched += 1;
                eprintln!(
                    "pixel {x},{y}: drawn {drawn:?}, cpu {:?} at {:.1}",
                    cpu.cell, cpu.distance
                );
            }
        }
    }
    eprintln!("compared {compared}, mismatched {mismatched}");
    assert!(compared > 100, "{compared} pixels compared");
    assert!(mismatched * 50 <= compared, "{mismatched} of {compared}");
}

#[test]
fn orbital_view_has_complete_coverage() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *planet.grid();
    let dir = grid.direction(
        2,
        f64::from(grid.cells()) * 0.3,
        f64::from(grid.cells()) * 0.7,
    );
    let eye = dir * (grid.radius() + 300_000.0);
    let view = View::new(
        eye,
        (-dir + dir.any_orthonormal_vector() * 0.5)
            .normalize()
            .as_vec3(),
        [256, 144],
    );
    let hits = World::new(&gpu, &planet).settle(&gpu, &planet, &view);
    let view_proj = view.view_proj(&planet);
    let meets = |d: DVec3, r: f64| {
        let b = eye.dot(d);
        b * b - (eye.length_squared() - r * r) >= 0.0 && b < 0.0
    };
    let (mut ground, mut wrong) = (0, 0);
    for y in 0..view.size[1] {
        for x in 0..view.size[0] {
            let d = view.pixel_dir(view_proj, x, y);
            let hit = hits[(y * view.size[0] + x) as usize].is_some();
            if meets(d, grid.radius() - 20_000.0) {
                ground += 1;
                wrong += usize::from(!hit);
            } else if !meets(d, grid.radius() + 20_000.0) {
                wrong += usize::from(hit);
            }
        }
    }
    assert!(ground > hits.len() / 3, "{ground} ground pixels");
    assert_eq!(wrong, 0);
}

/// The sun at each pixel (`directional_visibility`'s x): lit (1) or not.
fn read_sun(gpu: &Gpu, world: &World) -> Vec<bool> {
    let targets = world.chunks.targets().expect("drawn");
    let [w, h] = [targets.sun.width(), targets.sun.height()];
    let row = (w * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        targets.sun.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    // x as f16: lit is 1.0 (0x3c00), shaded 0.
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            let at = (y * row + x * 8) as usize;
            u16::from_le_bytes([bytes[at], bytes[at + 1]]) >= 0x3800
        })
        .collect()
}

/// Terrain shades the meshes in front of it: a mesh wall across a ground
/// view (the scene's depth one view depth everywhere, as a mesh pass leaves
/// it), with a raised ball between it and the sun, is in the sun where a
/// CPU ray from its point towards the sun reaches the sky, at points before
/// hits and before the sky alike; and sees a point light past the ball
/// where a CPU ray from its point reaches the light.
#[test]
fn terrain_shadows_fall_on_meshes_in_front_of_it() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [320, 180]);
    let forward = view.forward.as_dvec3();
    let up = planet.grid().up(view.eye);
    let side = up.cross(forward).normalize();
    // Half way up, across the view and away from the eye.
    let sun = (up * 0.5 + side * 0.75 + forward * 0.4).normalize();
    // The wall 12 m out, its depth the reversed depth's near / w; the ball
    // 6 m behind its point 4.5 m over the eye's line, towards the sun.
    let wall_m = 12.0f32;
    let ball = view.eye + forward * f64::from(wall_m) + up * 4.5 + sun * 6.0;
    planet
        .apply(Brush {
            center: ball.to_array(),
            radius: 2.5,
            shape: BrushShape::Sphere,
            op: BrushOp::Add,
            material: material::BRICK,
            height: 0.0,
        })
        .unwrap();
    let depth = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mesh wall depth"),
        size: wgpu::Extent3d {
            width: view.size[0],
            height: view.size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("mesh wall"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(view.near / wall_m),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    gpu.queue.submit([encoder.finish()]);
    let mut world = World::new(gpu, &planet);
    world.sun = sun.as_vec3();
    world.scene_depth = Some(depth_view);
    let light = ball + sun * 4.0;
    world.lights = Some(one_light(
        gpu,
        (light - view.eye).as_vec3(),
        40.0,
        view.size,
    ));
    world.chunks.set_steady_shadows(true);
    let hits = world.settle(gpu, &planet, &view);
    let lit = read_sun(gpu, &world);
    let seen = read_local(gpu, &world);
    let (mut light_mismatched, mut light_blocked_points) = (0, 0);

    let view_proj = view.view_proj(&planet);
    let voxel = planet.grid().voxel_size();
    let (mut before_hits, mut before_sky, mut shaded, mut sunny, mut shaded_sky, mut mismatched) =
        (0, 0, 0, 0, 0, 0);
    for y in (0..view.size[1]).step_by(4) {
        for x in (0..view.size[0]).step_by(4) {
            let dir = view.pixel_dir(view_proj, x, y);
            let t = f64::from(wall_m) / dir.dot(forward);
            let hit = hits[(y * view.size[0] + x) as usize];
            if hit.is_some_and(|h| h.distance <= t) {
                continue;
            }
            let point = view.eye + dir * t;
            // From just off the point, as the walk starts.
            let blocked = planet.raycast(point + sun * (0.05 * voxel), sun, 600.0);
            // A ray grazing a voxel's corner: either answer.
            if blocked.is_some_and(|b| b.distance < voxel) {
                continue;
            }
            // Rays end at the wall: the terrain past it is the CPU's.
            let terrain_behind = planet.raycast(point, dir, 600.0).is_some();
            if terrain_behind {
                before_hits += 1
            } else {
                before_sky += 1
            }
            if blocked.is_some() {
                shaded += 1
            } else {
                sunny += 1
            }
            if !terrain_behind && blocked.is_some() {
                shaded_sky += 1
            }
            if lit[(y * view.size[0] + x) as usize] == blocked.is_some() {
                mismatched += 1;
                if mismatched < 8 {
                    eprintln!(
                        "pixel {x},{y}: gpu lit {}, cpu {blocked:?}",
                        lit[(y * view.size[0] + x) as usize]
                    );
                }
            }
            let to = light - point;
            let light_blocked = planet
                .raycast(
                    point + to.normalize() * (0.05 * voxel),
                    to.normalize(),
                    to.length() - voxel,
                )
                .is_some();
            light_blocked_points += usize::from(light_blocked);
            light_mismatched +=
                usize::from((seen[(y * view.size[0] + x) as usize] > 0.5) == light_blocked);
        }
    }
    eprintln!("the light: {light_blocked_points} blocked, {light_mismatched} mismatched");
    assert!(
        light_blocked_points > 50,
        "{light_blocked_points} wall points blocked from the light"
    );
    assert!(
        light_mismatched * 100 <= before_hits + before_sky,
        "{light_mismatched} mismatched for the light"
    );
    eprintln!("wall points: {before_hits} before hits, {before_sky} before the sky ({shaded_sky} shaded); {shaded} shaded, {sunny} sunny; {mismatched} mismatched");
    assert!(
        before_hits > 200 && before_sky > 50,
        "{before_hits} before hits, {before_sky} before the sky"
    );
    assert!(shaded_sky > 10, "{shaded_sky} shaded before the sky");
    assert!(
        shaded > 100 && sunny > 100,
        "{shaded} shaded, {sunny} sunny"
    );
    assert!(
        mismatched * 100 <= before_hits + before_sky,
        "{mismatched} mismatched"
    );
}

/// The publish shows a hit only where no mesh is in front of it, as the
/// trace found against the meshes' depth: a mesh plane across a ground view
/// (the scene's depth one view depth everywhere, the gbuffer a marker
/// colour, as a mesh pass leaves them) keeps its colour and depth wherever
/// the ground is past it and over the sky; nearer, the ground's colour and
/// depth replace them. The publish does not test the depth itself: on
/// NVIDIA's Vulkan driver its depth test once let every hit through, whole
/// runs at a time (black grass).
#[test]
fn publish_shows_hits_only_in_front_of_meshes() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [320, 180]);
    let forward = view.forward.as_dvec3();
    let plane_m = 6.0f32;
    let plane_depth = view.near / plane_m;
    let [w, h] = view.size;
    let texture = |label: &str, format: wgpu::TextureFormat| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let depth = texture("mesh plane depth", wgpu::TextureFormat::Depth32Float);
    let depth_view = depth.create_view(&Default::default());
    let gbuffer: Vec<wgpu::Texture> = helio_frame_api::gbuffer::TARGET_FORMATS
        .iter()
        .map(|&f| texture("gbuffer", f))
        .collect();
    let gbuffer_views: Vec<wgpu::TextureView> = gbuffer
        .iter()
        .map(|t| t.create_view(&Default::default()))
        .collect();
    let marker = wgpu::Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    // The mesh pass: its colour and depth everywhere.
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("mesh plane"),
        color_attachments: &gbuffer_views
            .iter()
            .map(|v| {
                Some(wgpu::RenderPassColorAttachment {
                    view: v,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(marker),
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect::<Vec<_>>(),
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(plane_depth),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    gpu.queue.submit([encoder.finish()]);
    let mut world = World::new(gpu, &planet);
    world.scene_depth = Some(depth.create_view(&Default::default()));
    let hits = world.settle(gpu, &planet, &view);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let views: [&wgpu::TextureView; 8] = std::array::from_fn(|i| &gbuffer_views[i]);
    world.chunks.publish_scene(
        &mut helio_core::CommandRecorder::from_encoder(&mut encoder),
        views,
        &depth_view,
        None,
    );
    gpu.queue.submit([encoder.finish()]);
    let albedo = read_rgba8(gpu, &gbuffer[0]);
    let published_depth = read_depth(gpu, &depth);

    let view_proj = view.view_proj(&planet);
    let (mut shown, mut behind, mut wrong) = (0, 0, 0);
    for y in 0..h {
        for x in 0..w {
            let at = (y * w + x) as usize;
            let t = f64::from(plane_m) / view.pixel_dir(view_proj, x, y).dot(forward);
            let hit = hits[at].map(|h| h.distance);
            // Within a hundredth of the plane: either answer.
            if hit.is_some_and(|d| (d - t).abs() < 0.01 * t) {
                continue;
            }
            let marked = albedo[at] == [255, 0, 255, 255];
            let bad = if hit.is_some_and(|d| d < t) {
                shown += 1;
                marked || published_depth[at] <= plane_depth
            } else {
                behind += 1;
                !marked || published_depth[at] != plane_depth
            };
            if bad {
                wrong += 1;
                if wrong < 8 {
                    eprintln!(
                        "pixel {x},{y}: hit {hit:?}, plane at {t:.3}, albedo {:?}, depth {}",
                        albedo[at], published_depth[at]
                    );
                }
            }
        }
    }
    eprintln!("{shown} hits shown, {behind} behind the plane or sky");
    assert!(
        shown > 2000 && behind > 2000,
        "{shown} shown, {behind} behind"
    );
    assert_eq!(wrong, 0);
}

/// Terrain behind meshes is not traced: with a mesh plane across a ground
/// view (the scene's depth one view depth everywhere), the hits in front of
/// it are the open view's, there are none past it, the trace takes a
/// fraction of the open view's steps, and the ray reports (the nodes hits
/// want) come only from the terrain in front.
#[test]
fn terrain_behind_meshes_is_not_traced() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [320, 180]);
    let forward = view.forward.as_dvec3();
    let plane_m = 10.0f32;
    let [w, h] = view.size;
    let depth = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mesh plane depth"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("mesh plane"),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(view.near / plane_m),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    gpu.queue.submit([encoder.finish()]);
    let counting = Capacity {
        counters: true,
        ..TEST_CAPACITY
    };
    let mut open = World::with_capacity(gpu, &planet, counting);
    let mut covered = World::with_capacity(gpu, &planet, counting);
    covered.scene_depth = Some(depth_view);
    let open_hits = open.settle(gpu, &planet, &view);
    let covered_hits = covered.settle(gpu, &planet, &view);
    // Settled frames' steps and ray reports.
    let measure = |world: &mut World| {
        world.chunks.take_counts();
        let seen: usize = (0..16)
            .map(|_| world.frame(gpu, &planet, &view, None).seen)
            .sum();
        (world.chunks.take_counts().steps, seen)
    };
    let (open_steps, open_seen) = measure(&mut open);
    let (covered_steps, covered_seen) = measure(&mut covered);

    let view_proj = view.view_proj(&planet);
    let (mut front, mut past, mut wrong) = (0, 0, 0);
    for y in 0..h {
        for x in 0..w {
            let at = (y * w + x) as usize;
            let t = f64::from(plane_m) / view.pixel_dir(view_proj, x, y).dot(forward);
            let (open_hit, covered_hit) = (open_hits[at], covered_hits[at]);
            // Within a hundredth of the plane: either answer.
            let d = open_hit.map_or(f64::INFINITY, |h| h.distance);
            if (d - t).abs() < 0.01 * t {
                continue;
            }
            let bad = if d < t {
                front += 1;
                covered_hit != open_hit
            } else {
                past += usize::from(open_hit.is_some());
                covered_hit.is_some()
            };
            if bad {
                wrong += 1;
                if wrong < 8 {
                    eprintln!("pixel {x},{y}: open {open_hit:?}, covered {covered_hit:?}, plane at {t:.3}");
                }
            }
        }
    }
    eprintln!(
        "{front} hits in front of the plane, {past} past it, {wrong} wrong; steps {open_steps} open, {covered_steps} covered; reports {open_seen} open, {covered_seen} covered"
    );
    assert!(
        front > 2000 && past > 10000,
        "{front} in front, {past} past"
    );
    assert!(wrong * 1000 <= front + past, "{wrong} wrong");
    // Measured 2026-10-27: 48% of the steps, 2% of the reports.
    assert!(
        covered_steps * 3 < open_steps * 2,
        "{covered_steps} steps covered, {open_steps} open"
    );
    assert!(
        covered_seen * 10 < open_seen,
        "{covered_seen} reports covered, {open_seen} open"
    );
}

/// An Rgba8Unorm target, row by row.
fn read_rgba8(gpu: &Gpu, texture: &wgpu::Texture) -> Vec<[u8; 4]> {
    let [w, h] = [texture.width(), texture.height()];
    let row = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            let at = (y * row + x * 4) as usize;
            [bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]
        })
        .collect()
}

/// A Depth32Float target, row by row.
fn read_depth(gpu: &Gpu, texture: &wgpu::Texture) -> Vec<f32> {
    let [w, h] = [texture.width(), texture.height()];
    let row = (w * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            let at = (y * row + x * 4) as usize;
            f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        })
        .collect()
}

/// An f16's value.
fn f16_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa / 1024.0 * 2f32.powi(-14),
        31 => f32::INFINITY,
        e => (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Channel `channel` of an Rgba16Float target, row by row.
fn read_rgba16(gpu: &Gpu, texture: &wgpu::Texture, channel: usize) -> Vec<f32> {
    let [w, h] = [texture.width(), texture.height()];
    let row = (w * 8).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            let at = (y * row + x * 8) as usize + channel * 2;
            f16_value(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
        })
        .collect()
}

/// The sun's visibility as lighting takes it (`directional_visibility`'s x).
fn read_sun_visibility(gpu: &Gpu, world: &World) -> Vec<f32> {
    read_rgba16(gpu, &world.chunks.targets().expect("drawn").sun, 0)
}

/// The sun as the walks found it (each pixel's or its block's sample).
fn read_sun_walked(gpu: &Gpu, world: &World) -> Vec<f32> {
    read_words(gpu, &world.chunks.targets().expect("drawn").sun_walked)
        .iter()
        .map(|w| (w & 0xff) as f32 / 255.0)
        .collect()
}

/// The sky's visibility as the gbuffer takes it, and the walks' own
/// (each pixel's block's sample).
fn read_sky_visibility(gpu: &Gpu, world: &World) -> Vec<f32> {
    read_f32(gpu, &world.chunks.targets().expect("drawn").sky, 1, 0)
}

fn read_sky_samples(gpu: &Gpu, world: &World) -> Vec<f32> {
    read_rgba16(gpu, &world.chunks.targets().expect("drawn").normal, 3)
}

/// Channel `channel` of a 32-bit float target of `channels`, row by row.
fn read_f32(gpu: &Gpu, texture: &wgpu::Texture, channels: u32, channel: u32) -> Vec<f32> {
    let [w, h] = [texture.width(), texture.height()];
    let row = (w * 4 * channels).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .map(|(y, x)| {
            let at = (y * row + (x * channels + channel) * 4) as usize;
            f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
        })
        .collect()
}

/// Frame to frame change of one visibility term over `frames` (mean
/// absolute change, share of pixels changing by more than a tenth) at the
/// pixels of `mask`.
fn flicker(frames: &[Vec<f32>], mask: &[bool]) -> (f64, f64) {
    let (mut sum, mut big, mut n) = (0.0, 0usize, 0usize);
    for pair in frames.windows(2) {
        for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
            let d = f64::from((pair[1][i] - pair[0][i]).abs());
            sum += d;
            big += usize::from(d > 0.1);
            n += 1;
        }
    }
    (sum / n.max(1) as f64, big as f64 / n.max(1) as f64)
}

/// Error of `frames` against `truth` at the pixels of `mask`: mean absolute,
/// share off by more than a half.
fn error(frames: &[Vec<f32>], truth: &[f32], mask: &[bool]) -> (f64, f64) {
    let (mut sum, mut wrong, mut n) = (0.0, 0usize, 0usize);
    for frame in frames {
        for (i, _) in mask.iter().enumerate().filter(|(_, m)| **m) {
            let d = f64::from((frame[i] - truth[i]).abs());
            sum += d;
            wrong += usize::from(d > 0.5);
            n += 1;
        }
    }
    (sum / n.max(1) as f64, wrong as f64 / n.max(1) as f64)
}

/// A ground view with a raised ball between it and a low sun: natural
/// steps, the ball and its shadow over the ground.
fn shadowed_ground() -> (Planet, View, Vec3) {
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [320, 180]);
    let forward = view.forward.as_dvec3();
    let up = planet.grid().up(view.eye);
    let side = up.cross(forward).normalize();
    let sun = (up * 0.45 + side * 0.8 + forward * 0.3).normalize();
    let ball = view.eye + forward * 9.0 + up * 1.5 + sun * 4.0;
    planet
        .apply(Brush {
            center: ball.to_array(),
            radius: 2.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Add,
            material: material::BRICK,
            height: 0.0,
        })
        .unwrap();
    (planet, view, sun.as_vec3())
}

/// The sun's and the sky's visibility as lighting takes them (denoised),
/// against steady shadows (every hit walking SUN_DISK_RAYS rays over the
/// sun's disk: the sun's truth, penumbrae included) and the walks' own sky samples averaged over many frames (the
/// sky's), with a still camera: within a small error, and steadier from
/// frame to frame than the walks' own by far. A sudden turn of the sun
/// shows within a few frames. Then with the projection jittered as a
/// host's TSR jitters it: on pixels whose hits the jitter leaves alone,
/// the denoised sun (read where each frame's jitter puts the pixel's
/// sample) follows steady shadows frame by frame (the same jitter),
/// changing no more than they do, and the sky stays near steady.
/// Then the shadow stage's and the denoiser's GPU time.
#[test]
fn terrain_shadows_match_steady_shadows() {
    let Some(gpu) = gpu() else { return };
    let (planet, view, sun) = shadowed_ground();
    // Later the sun turns 20 degrees about the vertical and sinks a little.
    let up = planet.grid().up(view.eye).as_vec3();
    let turned =
        (glam::Quat::from_axis_angle(up, 20f32.to_radians()) * sun - up * 0.08).normalize();
    let mut world = World::new(gpu, &planet);
    world.sun = turned;
    world.chunks.set_steady_shadows(true);
    let hits = world.settle(gpu, &planet, &view);
    for _ in 0..4 {
        world.frame(gpu, &planet, &view, None);
    }
    let truth_turned = read_sun_visibility(gpu, &world);
    world.sun = sun;
    for _ in 0..4 {
        world.frame(gpu, &planet, &view, None);
    }
    let truth = read_sun_visibility(gpu, &world);
    let terrain: Vec<bool> = hits.iter().map(|h| h.is_some()).collect();
    let lit = terrain
        .iter()
        .zip(&truth)
        .filter(|(t, s)| **t && **s > 0.5)
        .count();
    let terrain_count = terrain.iter().filter(|t| **t).count();
    eprintln!("terrain pixels {terrain_count}, lit {lit}");
    assert!(
        lit > terrain_count / 5 && lit * 5 < terrain_count * 4,
        "{lit} of {terrain_count} lit"
    );
    world.chunks.set_steady_shadows(false);
    // The sky's reference: the walks' samples over 256 frames.
    let mut reference = vec![0.0f32; terrain.len()];
    for _ in 0..256 {
        world.frame(gpu, &planet, &view, None);
        for (r, s) in reference.iter_mut().zip(read_sky_samples(gpu, &world)) {
            *r += s / 256.0;
        }
    }
    let (mut suns, mut skies, mut walked_suns, mut walked_skies) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..16 {
        world.frame(gpu, &planet, &view, None);
        suns.push(read_sun_visibility(gpu, &world));
        skies.push(read_sky_visibility(gpu, &world));
        walked_suns.push(read_sun_walked(gpu, &world));
        walked_skies.push(read_sky_samples(gpu, &world));
    }
    let report = |name: &str, frames: &[Vec<f32>], truth: &[f32], mask: &[bool]| {
        let (e, w) = error(frames, truth, mask);
        let (f, j) = flicker(frames, mask);
        eprintln!(
            "{name}: error {e:.4} (off by > 0.5: {:.2}%), flicker {f:.4} (> 0.1: {:.2}%)",
            w * 100.0,
            j * 100.0
        );
        (e, w, f, j)
    };
    let sun_before = report("sun, walked", &walked_suns, &truth, &terrain);
    let sun_after = report("sun, denoised", &suns, &truth, &terrain);
    let in_penumbra: Vec<bool> = terrain
        .iter()
        .zip(&truth)
        .map(|(t, s)| *t && *s > 0.05 && *s < 0.95)
        .collect();
    report(
        "sun in penumbrae, walked",
        &walked_suns,
        &truth,
        &in_penumbra,
    );
    report("sun in penumbrae, denoised", &suns, &truth, &in_penumbra);
    let sky_before = report("sky, walked", &walked_skies, &reference, &terrain);
    let sky_after = report("sky, denoised", &skies, &reference, &terrain);
    assert!(
        sun_after.0 <= 0.01 && sun_after.1 <= 0.002,
        "sun against steady shadows: {sun_after:?}"
    );
    assert!(
        sun_after.2 * 10.0 <= sun_before.2 && sun_after.3 * 10.0 <= sun_before.3,
        "sun flicker {sun_after:?} against the walks' {sun_before:?}"
    );
    assert!(
        sky_after.0 <= 0.03
            && sky_after.2 * 10.0 <= sky_before.2
            && sky_after.3 * 10.0 <= sky_before.3,
        "sky {sky_after:?} against the walks' {sky_before:?}"
    );
    // A sudden turn of the sun: the error frame by frame after it.
    world.sun = turned;
    let (mut after, mut walked) = (Vec::new(), Vec::new());
    for _ in 0..16 {
        world.frame(gpu, &planet, &view, None);
        after.push(error(
            &[read_sun_visibility(gpu, &world)],
            &truth_turned,
            &terrain,
        ));
        walked.push(read_sun_walked(gpu, &world));
    }
    for _ in 0..48 {
        world.frame(gpu, &planet, &view, None);
    }
    let settled = error(&[read_sun_visibility(gpu, &world)], &truth_turned, &terrain);
    let walked = error(&walked, &truth_turned, &terrain);
    let line: Vec<String> = after
        .iter()
        .enumerate()
        .map(|(k, (e, w))| format!("{}: {e:.4}/{:.2}%", k + 1, w * 100.0))
        .collect();
    eprintln!("sun turned 20 degrees, error (mean/off by > 0.5) by frames after: {}; 64 frames after: {:.4}/{:.2}%; the walks' own: {:.4}/{:.2}%", line.join(", "), settled.0, settled.1 * 100.0, walked.0, walked.1 * 100.0);
    // Within a few frames as near as the walks themselves, and settling.
    assert!(
        after[3].1 <= walked.1 && after[3].1 <= settled.1 + 0.01,
        "4 frames after the sun turned: {:?}, the walks' {walked:?}, settled {settled:?}",
        after[3]
    );
    // Jittered: steady shadows, then denoised, on the pixels whose hits
    // (level, face, distance within 2%) hold through every jittered frame.
    world.sun = sun;
    world.jitter = Some(0);
    let mut stable = terrain.clone();
    let jittered = |world: &mut World, stable: &mut Vec<bool>, steady: bool| {
        world.chunks.set_steady_shadows(steady);
        for _ in 0..64 {
            world.frame(gpu, &planet, &view, None);
        }
        // Both runs measured over the same jitter.
        world.jitter = Some(0);
        let (mut suns, mut skies) = (Vec::new(), Vec::new());
        for _ in 0..16 {
            world.frame(gpu, &planet, &view, None);
            suns.push(read_sun_visibility(gpu, &world));
            skies.push(read_sky_visibility(gpu, &world));
            for (s, (now, then)) in stable
                .iter_mut()
                .zip(world.chunks.read_hits().iter().zip(&hits))
            {
                *s &= now.zip(*then).is_some_and(|(a, b)| {
                    a.level == b.level
                        && a.entered == b.entered
                        && (a.distance - b.distance).abs() < 0.02 * b.distance
                });
            }
        }
        (suns, skies)
    };
    let (steady_suns, _) = jittered(&mut world, &mut stable, true);
    let (suns, skies) = jittered(&mut world, &mut stable, false);
    eprintln!(
        "jittered: {} of {terrain_count} pixels' hits hold",
        stable.iter().filter(|s| **s).count()
    );
    let (mut e, mut w, mut n) = (0.0, 0usize, 0usize);
    for (a, b) in suns.iter().zip(&steady_suns) {
        for i in (0..a.len()).filter(|i| stable[*i]) {
            let d = f64::from((a[i] - b[i]).abs());
            e += d;
            w += usize::from(d > 0.5);
            n += 1;
        }
    }
    let (sun_error, sun_wrong) = (e / n.max(1) as f64, w as f64 / n.max(1) as f64);
    let (steady_flicker, steady_jumps) = flicker(&steady_suns, &stable);
    let (sun_flicker, sun_jumps) = flicker(&suns, &stable);
    eprintln!("sun, jittered, frame by frame against steady shadows: error {sun_error:.4} (off by > 0.5: {:.2}%); flicker {sun_flicker:.4} (> 0.1: {:.2}%), steady shadows' {steady_flicker:.4} (> 0.1: {:.2}%)", sun_wrong * 100.0, sun_jumps * 100.0, steady_jumps * 100.0);
    let sky_jittered = report("sky, denoised, jittered", &skies, &reference, &stable);
    // (Read at the jittered points, both change with the jitter at edges:
    // TSR gathers that as it does the edges of the hits themselves.)
    assert!(sun_error <= 0.02 && sun_wrong <= 0.01 && sun_flicker <= steady_flicker, "jittered sun: error {sun_error}, wrong {sun_wrong}, flicker {sun_flicker} (steady {steady_flicker})");
    assert!(
        sky_jittered.0 <= 0.03 && sky_jittered.2 <= 0.01,
        "jittered sky {sky_jittered:?}"
    );
    world.chunks.take_timings();
    for _ in 0..60 {
        world.frame(gpu, &planet, &view, None);
    }
    let t = world.chunks.take_timings();
    let n = t.len().max(1) as f64;
    eprintln!(
        "shadow stage {:.3} ms, denoiser {:.3} ms ({} frames timed)",
        t.iter().map(|f| f.shadow).sum::<f64>() / n,
        t.iter().map(|f| f.denoise).sum::<f64>() / n,
        t.len()
    );
}

/// The shadow stage's and the denoiser's GPU time at 1920x1080 over the
/// shadowed ground (`cargo test --release --test chunks -- --ignored
/// shadow_denoise_cost --nocapture`): steady shadows (every hit walking),
/// samples shared and denoised.
#[test]
#[ignore]
fn shadow_denoise_cost() {
    let Some(gpu) = gpu() else { return };
    let (planet, mut view, sun) = shadowed_ground();
    view.size = [1920, 1080];
    let mut world = World::new(gpu, &planet);
    world.sun = sun;
    // Residency fills most of the view (a full one outgrows the tests'
    // capacity), then the stages are timed.
    for _ in 0..400 {
        world.frame(gpu, &planet, &view, None);
    }
    let stage = |world: &mut World| {
        for _ in 0..30 {
            world.frame(gpu, &planet, &view, None);
        }
        world.chunks.take_timings();
        for _ in 0..240 {
            world.frame(gpu, &planet, &view, None);
        }
        let mut t = world.chunks.take_timings();
        t.sort_by(|a, b| (a.shadow + a.denoise).total_cmp(&(b.shadow + b.denoise)));
        let mid = t[t.len() / 2];
        (mid.shadow, mid.denoise)
    };
    let sampled = stage(&mut world);
    world.chunks.set_steady_shadows(true);
    let steady = stage(&mut world);
    eprintln!("1920x1080, median frame: samples shared: shadow stage {:.3} ms, denoiser {:.3} ms; every hit walking: shadow stage {:.3} ms", sampled.0, sampled.1, steady.0);
}

/// Tools ask for the terrain hit under a view point (the editor's brush):
/// the answer, frames later, is the hit drawn there; the sky has none.
#[test]
fn picks_report_the_drawn_hit() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let mut view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [320, 180]);
    // The upper rows see the sky.
    view.forward = (view.forward + planet.grid().up(view.eye).as_vec3() * 0.6).normalize();
    let mut world = World::new(&gpu, &planet);
    let hits = world.settle(&gpu, &planet, &view);
    let pixels: Vec<[u32; 2]> = (0..8).map(|n| [20 + n * 37, 5 + n * 24]).collect();
    let mut picks = Picks::default();
    picks.waiting = pixels
        .iter()
        .enumerate()
        .map(|(n, p)| {
            let uv = [(p[0] as f32 + 0.5) / 320.0, (p[1] as f32 + 0.5) / 180.0];
            (n as u64, uv)
        })
        .collect();
    for _ in 0..8 {
        world.frame(&gpu, &planet, &view, Some(&mut picks));
    }
    let results = picks.results();
    assert_eq!(results.len(), pixels.len());
    assert!(
        results.iter().any(|r| r.hit.is_none()) && results.iter().any(|r| r.hit.is_some()),
        "sky and ground picked"
    );
    for r in results {
        let [x, y] = pixels[r.id as usize];
        let drawn = hits[(y * 320 + x) as usize];
        assert_eq!(
            r.hit.map(|h| (h.cell, h.level)),
            drawn.map(|h| (h.cell, h.level)),
            "pixel {x},{y}"
        );
    }
}

/// Huts, trees and boulders (`features`): spawns widest first, each spacing
/// at least its stamps' footprint.
fn forest(ground_m: f64) -> FeatureSet {
    let mut hut = Stamp::new();
    hut.fill([-4, -1, -4], [4, 6, 4], material::PLANKS);
    hut.hollow([-3, 0, -3], [3, 5, 3]);
    hut.hollow([0, 0, -4], [1, 2, -4]);
    hut.fill([-4, 7, -4], [4, 7, 4], material::BRICK);
    let mut tree = Stamp::new();
    tree.fill([-3, 9, -3], [3, 13, 3], material::LEAVES);
    tree.fill([-2, 14, -2], [2, 15, 2], material::LEAVES);
    tree.fill([0, 0, 0], [0, 11, 0], material::WOOD);
    let mut boulder = Stamp::new();
    boulder.fill([-1, -1, -1], [1, 1, 1], material::BASALT);
    boulder.set([0, 2, 0], material::BASALT);
    // A round tree of shapes: a leaning tapered trunk, a branch, sparse
    // leaf balls with broken edges, a hollow carved in its crown; and a
    // rough round boulder.
    let mut round = Stamp::new();
    round.capsule(
        [0.5, -1.0, 0.5],
        [1.2, 10.0, 0.3],
        [1.4, 0.6],
        material::WOOD,
    );
    round.capsule(
        [1.0, 7.0, 0.5],
        [4.5, 11.0, 2.5],
        [0.7, 0.35],
        material::WOOD,
    );
    for (c, r) in [
        ([1.0, 12.0, 0.5], 4.5),
        ([4.0, 11.5, 2.5], 3.0),
        ([-1.5, 11.0, -1.0], 2.5),
    ] {
        round.add(Part {
            sparse: 0.3,
            rough: 1.5,
            lump: 1.2,
            lump_size: 2,
            ..Part::solid(
                Form::Capsule {
                    a: c,
                    b: c,
                    radius: [r; 2],
                },
                material::LEAVES,
            )
        });
    }
    round.cone(
        [-3.5, 2.0, 3.5],
        [-3.5, 7.0, 3.5],
        [2.0, 0.3],
        material::LEAVES,
    );
    round.add(Part::air(Form::Capsule {
        a: [2.0, 13.0, 1.0],
        b: [2.0, 13.0, 1.0],
        radius: [1.5; 2],
    }));
    round.add(Part {
        rough: 1.0,
        ..Part::solid(
            Form::Capsule {
                a: [5.5, 0.5, -3.5],
                b: [5.5, 0.5, -3.5],
                radius: [2.2; 2],
            },
            material::BASALT,
        )
    });
    FeatureSet {
        stamps: vec![hut, tree, boulder, round],
        spawns: vec![
            Spawn {
                name: "huts".into(),
                stamps: vec![0],
                spacing: 48,
                density: 0.5,
                max_relief: 4,
                sink: 1,
                ..Spawn::default()
            },
            // Trees up to a line wandering 20 to 100 m over the ground at
            // the forest's land (far canopies stop at it), boulders up to
            // one wandering through it (both sides in view).
            Spawn {
                name: "trees".into(),
                stamps: vec![1, 3],
                spacing: 16,
                density: 0.8,
                field_scale_m: 30.0,
                field_threshold: -0.4,
                heights_m: [-1e9, ground_m + 60.0],
                height_wander_m: [40.0, 300.0],
                max_relief: 3,
                far_material: material::LEAVES,
                ..Spawn::default()
            },
            Spawn {
                name: "boulders".into(),
                stamps: vec![2],
                spacing: 8,
                density: 0.15,
                heights_m: [-1e9, ground_m + 2.0],
                height_wander_m: [6.0, 40.0],
                sink: 1,
                ..Spawn::default()
            },
        ],
    }
}

/// A forest's planet and a land direction on it.
fn forest_planet() -> (Planet, DVec3) {
    let bare = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&bare, 2, 0.31, 0.62);
    let ground_m = bare.surface_point(dir, 0.0).length() - bare.grid().radius();
    let planet = Planet::new(PlanetRecipe {
        features: forest(ground_m),
        ..PlanetRecipe::default()
    })
    .unwrap();
    (planet, dir)
}

/// Features render exactly as the CPU places them: standing among trees,
/// huts and boulders, and from a treetop.
#[test]
fn features_render_exactly() {
    let Some(gpu) = gpu() else { return };
    let (planet, dir) = forest_planet();
    let grid = *planet.grid();
    let (c, _) = grid.locate(planet.surface_point(dir, 0.0));
    let feature_cells = (-60..60).flat_map(|d| [(d, 0), (0, d)]).filter(|&(a, b)| {
        let top = planet.column_top(c.face, c.i + a, c.j + b, 0);
        (top..top + 16).any(|k| {
            matches!(
                planet.features_at(0, c.face, c.i + a, c.j + b, k),
                Some(Some(_))
            )
        })
    });
    assert!(feature_cells.count() > 20, "few features near the view");
    matches_exactly(&gpu, &planet, ground_view(&planet, dir, [320, 180]), 5000);
    let eye = planet.surface_point(dir, 3.0);
    let up = eye.normalize();
    matches_exactly(
        &gpu,
        &planet,
        View::new(
            eye,
            (up.any_orthonormal_vector() - up * 0.6)
                .normalize()
                .as_vec3(),
            [320, 180],
        ),
        5000,
    );
}

/// Every pixel whose CPU hit is a feature's cell draws its stamp's
/// material at level 0.
#[test]
fn feature_cells_draw_their_own_materials() {
    let Some(gpu) = gpu() else { return };
    let (planet, dir) = forest_planet();
    let eye = planet.surface_point(dir, 2.5);
    let up = eye.normalize();
    let view = View::new(
        eye,
        (up.any_orthonormal_vector() - up * 0.3)
            .normalize()
            .as_vec3(),
        [320, 180],
    );
    let hits = World::new(&gpu, &planet).settle(&gpu, &planet, &view);
    let view_proj = view.view_proj(&planet);
    let (mut compared, mut wrong) = (0, 0);
    let mut seen = std::collections::BTreeSet::new();
    for y in 0..view.size[1] {
        for x in 0..view.size[0] {
            let Some(cpu) = planet.raycast(view.eye, view.pixel_dir(view_proj, x, y), 40.0) else {
                continue;
            };
            let c = cpu.cell;
            let Some(Some(want)) = planet.features_at(0, c.face, c.i, c.j, c.k) else {
                continue;
            };
            assert_eq!(planet.material(c), want);
            let Some(drawn) =
                hits[(y * view.size[0] + x) as usize].filter(|h| h.level == 0 && h.cell == c)
            else {
                continue;
            };
            compared += 1;
            seen.insert(want);
            if drawn.material != want {
                wrong += 1;
                if wrong < 8 {
                    eprintln!(
                        "pixel {x},{y}: {c:?} drawn {}, stamp {want}",
                        drawn.material
                    );
                }
            }
        }
    }
    eprintln!("compared {compared}, wrong {wrong}, materials {seen:?}");
    assert!(
        compared > 2000 && seen.len() >= 3,
        "{compared} pixels, materials {seen:?}"
    );
    assert_eq!(wrong, 0);
}

/// Coarse levels draw features' cells as the CPU makes them: from 60 m up
/// near stamps and the forest's first far levels (levels 1 and 2), from
/// 300 m the far canopy (levels 3 to 5): every hit on a cell features
/// decide is solid there for the CPU too.
#[test]
fn coarse_feature_cells_are_the_cpu_cells() {
    let Some(gpu) = gpu() else { return };
    let (planet, dir) = forest_planet();
    for (height, down, levels) in [(60.0, 0.8, 1..3), (300.0, 0.35, 3..6)] {
        coarse_features_match(&gpu, &planet, dir, height, down, levels);
    }
}

fn coarse_features_match(
    gpu: &Gpu,
    planet: &Planet,
    dir: DVec3,
    height: f64,
    down: f64,
    levels: std::ops::Range<u32>,
) {
    let eye = planet.surface_point(dir, height);
    let up = eye.normalize();
    let view = View::new(
        eye,
        (up.any_orthonormal_vector() - up * down)
            .normalize()
            .as_vec3(),
        [480, 270],
    );
    let hits = World::new(gpu, planet).settle(gpu, planet, &view);
    let mut by_level = [0usize; 8];
    let mut drawn = [0usize; 12];
    for h in hits.iter().flatten() {
        drawn[(h.level as usize).min(11)] += 1;
    }
    let mut wrong = 0;
    for h in hits.iter().flatten().filter(|h| (1..8).contains(&h.level)) {
        let c = h.cell;
        if planet.features_at(h.level, c.face, c.i, c.j, c.k).is_none() {
            continue;
        }
        by_level[h.level as usize] += 1;
        if planet.sample_kind(h.level, c.face, c.i, c.j, c.k).0 != 1 {
            wrong += 1;
            if wrong < 8 {
                eprintln!("level {} {c:?} drawn, air on the CPU", h.level);
            }
        }
    }
    eprintln!(
        "{height} m up: hits by level {drawn:?}, on feature cells {by_level:?}, wrong {wrong}"
    );
    assert!(
        levels.clone().all(|l| by_level[l as usize] > 100),
        "{height} m up: {by_level:?}"
    );
    assert_eq!(wrong, 0);
}

/// Every footprint's placements are the CPU's, square by square, at every
/// near level (the forest's: 0 and 1): from the ground and 60 m up, each
/// square of each near spawn holds its raw candidate
/// (`CompiledFeatures::candidate`), marked excluded where it reaches the
/// footprint and an earlier spawn's raw candidate meets its own
/// (`FeatureEval::placement`).
#[test]
fn footprint_placements_are_the_cpus() {
    let Some(gpu) = gpu() else { return };
    let (planet, dir) = forest_planet();
    let (features, grid) = (planet.features(), planet.grid());
    let top = |f: u8, i: i32, j: i32| {
        helio_pass_voxel_planet::features::ground(
            grid,
            planet.field(),
            f,
            i,
            j,
            0,
            planet.column_height(f, i, j, 0),
        )
    };
    let mut cache = Default::default();
    let mut eval = helio_pass_voxel_planet::features::FeatureEval::new(
        features,
        grid,
        planet.field(),
        &top,
        &mut cache,
    );
    let mut by_level = [0usize; 10];
    let (mut placed, mut excluded) = (0, 0);
    for (height, down) in [(1.7, 0.3), (60.0, 0.8)] {
        let eye = planet.surface_point(dir, height);
        let up = eye.normalize();
        let view = View::new(
            eye,
            (up.any_orthonormal_vector() - up * down)
                .normalize()
                .as_vec3(),
            [320, 180],
        );
        let mut world = World::new(&gpu, &planet);
        world.settle(&gpu, &planet, &view);
        let (footprints, buffer) = world.chunks.footprint_placements(VolumeId(0));
        let words = read_words(&gpu, buffer);
        for (fp, slot) in footprints {
            let near = features.near(fp.level);
            let first = [fp.cx * 64 << fp.level, fp.cy * 64 << fp.level];
            let last = first.map(|f| f + (64 << fp.level) - 1);
            for s in 0..near {
                let sp = &features.spawns[s];
                let (g0, side) = features.squares(fp.level, s, first);
                let offset = features.levels[fp.level as usize].spawns[s].offset as usize;
                for n in 0..(side * side) as usize {
                    let (gx, gy) = (g0[0] + n as i32 % side, g0[1] + n as i32 / side);
                    let at = 2 * (slot as usize * features.stride + offset + n);
                    let (x, y) = (words[at], words[at + 1]);
                    let want = features.candidate(s, grid, planet.field(), &top, fp.face, gx, gy);
                    let got = (x != 0).then(|| {
                        (
                            sp.stamps[((x >> 21) & 0x7f) as usize],
                            gx * sp.spacing + ((x >> 1) & 1023) as i32,
                            gy * sp.spacing + ((x >> 11) & 1023) as i32,
                            y as i32,
                            (x >> 29) & 3,
                            x >> 31 == 1,
                        )
                    });
                    assert_eq!(
                        got,
                        want.map(|p| (p.stamp, p.i, p.j, p.k, p.turn, p.mirror)),
                        "level {} footprint {fp:?} spawn {s} square {gx},{gy}",
                        fp.level
                    );
                    let Some(p) = want else { continue };
                    let r = features.stamps[p.stamp as usize].reach;
                    let reaches = p.i - r <= last[0]
                        && p.i + r >= first[0]
                        && p.j - r <= last[1]
                        && p.j + r >= first[1];
                    let out = reaches && eval.placement(s, fp.face, gx, gy).is_none();
                    assert_eq!(
                        (x >> 28) & 1 == 1,
                        out,
                        "level {} footprint {fp:?} spawn {s} square {gx},{gy}: excluded",
                        fp.level
                    );
                    by_level[fp.level as usize] += 1;
                    placed += usize::from(reaches && !out);
                    excluded += usize::from(out);
                }
            }
        }
    }
    eprintln!("candidates compared by level {by_level:?}: {placed} placed and {excluded} excluded over their footprints");
    assert!(
        by_level[..2].iter().all(|&n| n > 1000) && placed > 1000 && excluded > 100,
        "{by_level:?} {placed} {excluded}"
    );
}

/// What features cost to generate (`cargo test --release --test chunks
/// feature_generation_cost -- --ignored --nocapture`): the bare planet's
/// and the forest's views settled, generated again in full three times,
/// the GPU's time a tile and a chunk by stage, and the footprints'
/// placements' memory.
#[test]
#[ignore]
fn feature_generation_cost() {
    let Some(gpu) = gpu() else { return };
    let (forest, dir) = forest_planet();
    let bare = Planet::new(PlanetRecipe::default()).unwrap();
    let f = forest.features();
    eprintln!(
        "placements: {} squares a footprint, {:.1} MB over {} footprints",
        f.stride,
        (f.stride * 8 * TEST_CAPACITY.footprints) as f64 / 1048576.0,
        TEST_CAPACITY.footprints
    );
    for (name, planet) in [("bare", &bare), ("forest", &forest)] {
        for (height, down) in [(2.0, 0.3), (60.0, 0.8), (400.0, 0.35)] {
            let eye = planet.surface_point(dir, height);
            let up = eye.normalize();
            let view = View::new(
                eye,
                (up.any_orthonormal_vector() - up * down)
                    .normalize()
                    .as_vec3(),
                [480, 270],
            );
            let mut world = World::new(&gpu, planet);
            world.settle(&gpu, planet, &view);
            world.chunks.take_timings();
            for _ in 0..3 {
                world.chunks.regenerate_all();
                world.settle(&gpu, planet, &view);
            }
            let t = world.chunks.take_timings();
            let (tiles, chunks) = (
                t.iter().map(|f| f.tiles).sum::<usize>().max(1) as f64,
                t.iter().map(|f| f.chunks).sum::<usize>().max(1) as f64,
            );
            let stage = |k: usize| t.iter().map(|f| f.stages[k]).sum::<f64>() * 1e3;
            let part = |k: usize| t.iter().map(|f| f.tile_stages[k]).sum::<f64>() * 1e3 / tiles;
            eprintln!(
                "COST {name} {height} m: {tiles} tiles {chunks} chunks: tiles {:.1} us/tile (nodes {:.1}, columns {:.1}, halos {:.1}, features {:.1}), chunk columns {:.2} us/chunk, builds {:.2} us/chunk; {:.1} MB resident",
                stage(0) / tiles,
                part(0),
                part(1),
                part(2),
                part(3),
                stage(1) / chunks,
                stage(2) / chunks,
                world.chunks.resident_bytes() as f64 / 1048576.0
            );
        }
    }
}

/// How much of a tile's light 0 each pixel sees past the terrain, as
/// lighting takes it (`local_visibility`: the tracked lights' slots, their
/// visibility, the other slots' bits).
fn read_local(gpu: &Gpu, world: &World) -> Vec<f32> {
    let words = read_words(
        gpu,
        world.chunks.local_visibility().expect("drawn with lights"),
    );
    words
        .chunks(3)
        .map(|w| match (0..4).find(|k| (w[0] >> (8 * k)) & 0xff == 0) {
            Some(k) => ((w[1] >> (8 * k)) & 0xff) as f32 / 255.0,
            None => (w[2] & 1) as f32,
        })
        .collect()
}

/// The same as the walks found it (each pixel's or its block's).
fn read_local_walked(gpu: &Gpu, world: &World) -> Vec<f32> {
    read_words(gpu, world.chunks.local_walked().expect("drawn"))
        .chunks(2)
        .map(|w| (w[0] & 1) as f32)
        .collect()
}

/// Helio's lights and tile lists for one point light at `at`
/// (eye-relative) of range `range`, in every 16 px tile of `size`.
fn one_light(gpu: &Gpu, at: Vec3, range: f32, size: [u32; 2]) -> [wgpu::Buffer; 3] {
    use wgpu::util::DeviceExt;
    let mut light = [0u32; 32];
    light[..4].copy_from_slice(&[
        at.x.to_bits(),
        at.y.to_bits(),
        at.z.to_bits(),
        range.to_bits(),
    ]);
    light[8..12].copy_from_slice(&[
        1f32.to_bits(),
        1f32.to_bits(),
        1f32.to_bits(),
        10f32.to_bits(),
    ]);
    // Shadowed (shadow index 0), a point light.
    light[12] = 0;
    light[13] = 1;
    let tiles = (size[0].div_ceil(16) * size[1].div_ceil(16)) as usize;
    let init = |label, contents: &[u32]| {
        gpu.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(contents),
                usage: wgpu::BufferUsages::STORAGE,
            })
    };
    [
        init("lights", &light),
        init("tile lists", &vec![0u32; tiles * 64]),
        init("tile counts", &vec![1u32; tiles]),
    ]
}

/// A local light behind two-sided foliage reaches its back faces through
/// the leaf cell, as the sun does: a slab of leaves a cell or two thick
/// stands between the eye and a point light, a brick block hides the light
/// from part of it. Every leaf pixel turned away from the light sees it (in
/// `local_visibility`, steady shadows) where a CPU ray from the far side of
/// its cell reaches it; a few may differ (grazing cells). Before, a back
/// face skipped its walk and saw every light, through walls too.
#[test]
fn local_lights_reach_leaves_through_their_cells() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 4, 0.37, 0.61);
    let ground = planet.surface_point(dir, 0.0);
    let up = planet.grid().up(ground);
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let cell = planet.grid().voxel_size();
    // The slab: single leaf cells over a 2 m square across the view (a cell
    // or two thick wherever the grid's axes cross its plane).
    let centre = ground + up * 1.2 + fwd * 2.0;
    let brush = |center: DVec3, radius: f64, op: BrushOp, material: u32| Brush {
        center: center.to_array(),
        radius,
        shape: BrushShape::Cube,
        op,
        material,
        height: 0.0,
    };
    let steps = (2.0 / cell).round() as i32;
    for a in 0..=steps {
        for b in 0..=steps {
            let at = centre + side * (f64::from(a) * cell - 1.0) + up * (f64::from(b) * cell - 1.0);
            planet
                .apply(brush(at, 0.55 * cell, BrushOp::Add, material::LEAVES))
                .unwrap();
        }
    }
    // The light behind it; brick between them over one side.
    let light = centre + fwd * 1.6;
    planet
        .apply(brush(
            centre + fwd * 0.8 + side * 0.6,
            0.35,
            BrushOp::Add,
            material::BRICK,
        ))
        .unwrap();
    let eye = centre - fwd * 3.0 + up * 0.3;
    let view = View::new(eye, (centre - eye).normalize().as_vec3(), [320, 180]);
    let mut world = World::new(&gpu, &planet);
    let range = 8.0;
    world.lights = Some(one_light(&gpu, (light - eye).as_vec3(), range, view.size));
    world.chunks.set_steady_shadows(true);
    let hits = world.settle(&gpu, &planet, &view);
    let mask = read_local(&gpu, &world);
    let view_proj = view.view_proj(&planet);
    let (mut compared, mut lit, mut shadowed, mut wrong) = (0, 0, 0, 0);
    for y in (0..view.size[1]).step_by(2) {
        for x in (0..view.size[0]).step_by(2) {
            let ray = view.pixel_dir(view_proj, x, y);
            let Some(cpu) = planet.raycast(view.eye, ray, 30.0) else {
                continue;
            };
            if planet.material(cpu.cell) != material::LEAVES
                || hits[(y * view.size[0] + x) as usize].is_none_or(|h| h.level != 0)
            {
                continue;
            }
            let at = view.eye + ray * cpu.distance;
            let to = light - at;
            if to.length() > f64::from(range) - cell || cpu.normal.dot(to.normalize()) > -0.3 {
                continue;
            }
            // From the far side of the hit's cell, as the walk starts.
            let from = at - cpu.normal * (1.01 * cell);
            let rest = light - from;
            let seen = planet
                .raycast(from, rest.normalize(), rest.length() - cell)
                .is_none();
            compared += 1;
            lit += usize::from(seen);
            shadowed += usize::from(!seen);
            if seen != (mask[(y * view.size[0] + x) as usize] > 0.5) {
                wrong += 1;
                if wrong < 8 {
                    eprintln!(
                        "pixel {x},{y}: cpu {}, drawn otherwise",
                        if seen { "lit" } else { "shadowed" }
                    );
                }
            }
        }
    }
    eprintln!("compared {compared}, lit {lit}, shadowed {shadowed}, wrong {wrong}");
    assert!(
        lit > 100 && shadowed > 100,
        "{compared} compared, {lit} lit, {shadowed} shadowed"
    );
    assert!(wrong * 50 <= compared, "{wrong} of {compared} wrong");
}

/// The terrain shadows local lights: a point light 1 m over the ground
/// beside a raised ball, seen from above. With steady shadows (every hit
/// walks its own), every pixel facing the light sees it (in
/// `local_visibility`) where a CPU ray from its hit to the light reaches
/// it; a hundredth may differ (grazing cells). Denoised (samples shared,
/// turning each frame), the same within a small error and steady from
/// frame to frame; moved, its shadow follows within a few frames. Then the
/// shadow stage's and the denoiser's GPU time: every hit walking, a 2x2
/// block's sample shared, no light.
#[test]
fn terrain_shadows_local_lights() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 4, 0.37, 0.61);
    let ground = planet.surface_point(dir, 0.0);
    let up = planet.grid().up(ground);
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let light = ground + up * 1.0;
    planet
        .apply(Brush {
            center: (ground + fwd * 2.0 + up * 0.8).to_array(),
            radius: 1.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Add,
            material: material::BRICK,
            height: 0.0,
        })
        .unwrap();
    let eye = ground + up * 7.0 - fwd * 4.0 + side * 1.5;
    let view = View::new(
        eye,
        ((ground + fwd * 2.5) - eye).normalize().as_vec3(),
        [320, 180],
    );
    let mut world = World::new(&gpu, &planet);
    let range = 12.0;
    world.lights = Some(one_light(&gpu, (light - eye).as_vec3(), range, view.size));
    world.chunks.set_steady_shadows(true);
    let hits = world.settle(&gpu, &planet, &view);
    let mask = read_local(&gpu, &world);
    let view_proj = view.view_proj(&planet);
    let cell = planet.grid().voxel_size();
    let (mut compared, mut wrong, mut shadowed) = (0, 0, 0);
    let mut lit_pixels = vec![false; mask.len()];
    for y in (0..view.size[1]).step_by(2) {
        for x in (0..view.size[0]).step_by(2) {
            let ray = view.pixel_dir(view_proj, x, y);
            let Some(cpu) = planet.raycast(view.eye, ray, 60.0) else {
                continue;
            };
            if hits[(y * view.size[0] + x) as usize].is_none_or(|h| h.level != 0) {
                continue;
            }
            let at = view.eye + ray * cpu.distance;
            let to = light - at;
            if to.length() > f64::from(range) - cell || cpu.normal.dot(to.normalize()) < 0.3 {
                continue;
            }
            let seen = planet
                .raycast(at + cpu.normal * 1e-3, to.normalize(), to.length() - cell)
                .is_none();
            compared += 1;
            shadowed += usize::from(!seen);
            lit_pixels[(y * view.size[0] + x) as usize] = true;
            if seen != (mask[(y * view.size[0] + x) as usize] > 0.5) {
                wrong += 1;
                if wrong < 8 {
                    eprintln!(
                        "pixel {x},{y}: cpu {}, drawn {}",
                        if seen { "lit" } else { "shadowed" },
                        if seen { "shadowed" } else { "lit" }
                    );
                }
            }
        }
    }
    eprintln!("compared {compared}, shadowed {shadowed}, wrong {wrong}");
    assert!(
        compared > 2000 && shadowed > 200,
        "{compared} compared, {shadowed} shadowed"
    );
    assert!(wrong * 100 <= compared, "{wrong} of {compared} wrong");
    // Against steady shadows: the walks' own (each pixel taking its
    // block's sample) and denoised.
    world.chunks.set_steady_shadows(false);
    for _ in 0..64 {
        world.frame(&gpu, &planet, &view, None);
    }
    let (mut walked, mut denoised) = (Vec::new(), Vec::new());
    for _ in 0..16 {
        world.frame(&gpu, &planet, &view, None);
        walked.push(read_local_walked(&gpu, &world));
        denoised.push(read_local(&gpu, &world));
    }
    let report = |name: &str, frames: &[Vec<f32>]| {
        let (e, w) = error(frames, &mask, &lit_pixels);
        let (f, j) = flicker(frames, &lit_pixels);
        eprintln!(
            "light, {name}: error {e:.4} (off by > 0.5: {:.2}%), flicker {f:.4} (> 0.1: {:.2}%)",
            w * 100.0,
            j * 100.0
        );
        (e, w, f, j)
    };
    let before = report("walked", &walked);
    let after = report("denoised", &denoised);
    assert!(
        after.0 <= 0.01 && after.1 <= 0.005,
        "denoised against steady: {after:?}"
    );
    assert!(
        after.2 * 10.0 <= before.2 && after.3 * 10.0 <= before.3,
        "denoised flicker {after:?} against the walks' {before:?}"
    );
    // The light moves a metre aside: the steady shadows there, then frames
    // until the denoised ones follow.
    let moved = light + side * 1.0;
    world.lights = Some(one_light(&gpu, (moved - eye).as_vec3(), range, view.size));
    world.chunks.set_steady_shadows(true);
    for _ in 0..2 {
        world.frame(&gpu, &planet, &view, None);
    }
    let truth = read_local(&gpu, &world);
    world.chunks.set_steady_shadows(false);
    world.lights = Some(one_light(&gpu, (light - eye).as_vec3(), range, view.size));
    for _ in 0..64 {
        world.frame(&gpu, &planet, &view, None);
    }
    world.lights = Some(one_light(&gpu, (moved - eye).as_vec3(), range, view.size));
    let mut after_move = Vec::new();
    for _ in 0..8 {
        world.frame(&gpu, &planet, &view, None);
        after_move.push(error(&[read_local(&gpu, &world)], &truth, &lit_pixels));
    }
    let line: Vec<String> = after_move
        .iter()
        .enumerate()
        .map(|(k, (e, w))| format!("{}: {e:.4}/{:.2}%", k + 1, w * 100.0))
        .collect();
    eprintln!(
        "light moved 1 m, error (mean/off by > 0.5) by frames after: {}",
        line.join(", ")
    );
    assert!(
        after_move[3].1 <= 0.02,
        "4 frames after the light moved: {:?}",
        after_move[3]
    );
    // The shadow stage's and the denoiser's GPU time with the light and
    // without.
    let stage = |world: &mut World| {
        world.chunks.take_timings();
        for _ in 0..60 {
            world.frame(&gpu, &planet, &view, None);
        }
        let t = world.chunks.take_timings();
        let n = t.len().max(1) as f64;
        (
            t.iter().map(|f| f.shadow).sum::<f64>() / n,
            t.iter().map(|f| f.denoise).sum::<f64>() / n,
        )
    };
    world.chunks.set_steady_shadows(true);
    let steady = stage(&mut world);
    world.chunks.set_steady_shadows(false);
    let sampled = stage(&mut world);
    world.lights = None;
    let unlit = stage(&mut world);
    eprintln!("shadow stage, denoiser: {:.3}, {:.3} ms with the light (every hit walking); {:.3}, {:.3} ms (samples shared); {:.3}, {:.3} ms without", steady.0, steady.1, sampled.0, sampled.1, unlit.0, unlit.1);
}

fn read_words(gpu: &Gpu, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec()
}

/// The terrain occludes the sun for the volumetric fog and fine
/// cells (`sun_occlusion`): towers out to a kilometre cast shadows under a
/// low sun; points around each, between a quarter and three quarters of
/// its height, read at the footprint a pixel there has (or the nearest
/// coarser level resident), are lit by the pages as a CPU ray toward their
/// epoch's sun finds them, but for occluders within a few texels (the
/// bias) and reads across a shadow's edge (the filter's blend); a fiftieth
/// may differ (a texel's edge along a step). The same with the sun turning
/// a little each frame (as a day goes by): new epochs traced, each page
/// read with the sun it was traced for.
#[test]
fn sun_occlusion_matches_cpu_rays_to_the_sun() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 1, 0.52, 0.48);
    let eye = planet.surface_point(dir, 2.0);
    let up = eye.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    // (distance from the eye, height, width), m.
    let towers = [
        (10.0, 4.0, 1.5),
        (60.0, 16.0, 4.0),
        (250.0, 50.0, 16.0),
        (1000.0, 160.0, 50.0),
    ];
    let bases: Vec<DVec3> = towers
        .iter()
        .map(|&(at, height, width)| {
            let base = planet.surface_point((eye + fwd * at).normalize(), 0.0);
            let up = base.normalize();
            planet
                .apply(Brush {
                    center: (base + up * (height * 0.5 - 4.0)).to_array(),
                    radius: width * 0.5,
                    shape: BrushShape::Cube,
                    op: BrushOp::Add,
                    material: material::BRICK,
                    height: height * 0.5 + 4.0,
                })
                .unwrap();
            base
        })
        .collect();
    let view = View::new(eye, (fwd - up * 0.1).normalize().as_vec3(), [320, 180]);
    let mut world = World::new(gpu, &planet);
    world.sun_occlusion = true;
    // A low sun, across the line of towers: long shadows beside them.
    world.sun = (up * 0.5 + side * 0.87).normalize().as_vec3();
    world.settle(gpu, &planet, &view);
    world.settle_sun_pages(gpu, &planet, &view);
    let check = |world: &World| {
        let clock = std::time::Instant::now();
        let pages = world.chunks.read_sun_occlusion().expect("drawn");
        let mut next = xorshift(0x2545_f491_4f6c_dd1d);
        let mut unit = || (next() >> 11) as f64 / (1u64 << 53) as f64;
        let (mut compared, mut shadowed, mut wrong) = (0usize, 0usize, 0usize);
        let mut levels = std::collections::BTreeMap::new();
        for k in 0..960 {
            let (&(_, height, width), &base) = (&towers[k % 4], &bases[k % 4]);
            let up = base.normalize();
            // Beside the tower, from its sunward side out past its shadow.
            let (a, b) = ((unit() * 2.0 - 0.25) * height, (unit() - 0.5) * 1.5 * width);
            let ground = planet.surface_point((base - side * a + fwd * b).normalize(), 0.0);
            let p = ground + ground.normalize() * (0.25 + unit() * 0.5) * height;
            if planet.solid(planet.grid().locate(p).0) || (p - base).dot(up) < 0.2 * height {
                continue;
            }
            let footprint = ((p - eye).length() * view.pixel_angle()) as f32;
            let Some(read) = pages.read((p - eye).as_vec3().to_array(), [0.0; 3], footprint) else {
                continue;
            };
            // Across a shadow's edge the filter blends.
            if read.lit > 0.0 && read.lit < 1.0 {
                continue;
            }
            // Toward the sun the page was traced for, as far as the tower's
            // shadow reaches under this 30-degree sun, twice over (the pages
            // see to the ceiling; a CPU walk of 10 cm cells that far takes
            // seconds a point).
            let sun = DVec3::from_array(read.sun.map(f64::from)).normalize();
            let hit = planet.raycast(p, sun, f64::from(read.depth).min(4.0 * height));
            // Occluders within a few texels: the bias and texels decide.
            if hit
                .as_ref()
                .is_some_and(|h| h.distance < 4.0 * f64::from(read.texel))
            {
                continue;
            }
            let cpu = hit.is_none();
            compared += 1;
            shadowed += usize::from(!cpu);
            *levels.entry(read.level).or_insert(0usize) += 1;
            if (read.lit >= 0.5) != cpu {
                wrong += 1;
                if wrong < 8 {
                    eprintln!(
                        "point {k} level {} ({a:.1}, {b:.1}): pages {}, cpu {cpu}",
                        read.level, read.lit
                    );
                }
            }
        }
        eprintln!(
            "sun occlusion: compared {compared} by level {levels:?}, shadowed {shadowed}, wrong {wrong} ({:.1} s)",
            clock.elapsed().as_secs_f64()
        );
        assert!(
            compared > 200 && shadowed > 40,
            "{compared} compared, {shadowed} shadowed"
        );
        assert!(wrong * 50 <= compared, "{wrong} of {compared} wrong");
    };
    check(&world);
    // The sun turns 0.03 degrees a frame for 40 frames: epochs traced under
    // suns up to a degree apart, each page read with its own.
    let axis = up.cross(world.sun.as_dvec3()).normalize();
    for _ in 0..40 {
        world.sun = glam::DQuat::from_axis_angle(axis, 0.03_f64.to_radians())
            .mul_vec3(world.sun.as_dvec3())
            .as_vec3();
        world.frame(gpu, &planet, &view, None);
    }
    check(&world);
}

/// The sun's occlusion pages a pixel reads have texels about its footprint
/// at any eye height: at the Moon's pole 700 m up, and over the Earth from
/// orbit to the ground, hits of cells under a pixel and a half (those the
/// pages light) read texels no wider than two of their pixels' footprints
/// once the pages they asked for are traced.
#[test]
fn sun_occlusion_texels_follow_pixel_footprints() {
    let Some(gpu) = gpu() else { return };
    let clock = std::time::Instant::now();
    let voxel = 0.1;
    let moon = Planet::new(PlanetRecipe {
        radius_m: 1_737_400.0,
        voxel_size_m: voxel,
        terrain: TerrainLayers::moon().source(75),
        features: helio_pass_voxel_planet::feature_recipe::FeatureRecipe::moon().grow(voxel, 75),
        ..Default::default()
    })
    .unwrap();
    let earth = Planet::new(PlanetRecipe::default()).unwrap();
    eprintln!("planets built: {:.1} s", clock.elapsed().as_secs_f64());
    let mut poses = vec![(
        "moon pole",
        &moon,
        View::new(
            DVec3::new(0.0, 1_738_104.0, 0.0),
            Vec3::new(-0.07329183, -0.57357645, 0.81586665),
            [320, 180],
        ),
    )];
    let grid = *earth.grid();
    let dir = grid.direction(
        2,
        f64::from(grid.cells()) * 0.3,
        f64::from(grid.cells()) * 0.7,
    );
    let side = dir.any_orthonormal_vector();
    for (name, height, down) in [("earth orbit", 300_000.0, 1.0), ("earth ground", 2.0, 0.15)] {
        let eye = earth.surface_point(dir, height);
        poses.push((
            name,
            &earth,
            View::new(eye, (side - dir * down).normalize().as_vec3(), [320, 180]),
        ));
    }
    for (name, planet, view) in poses {
        let mut world = World::new(gpu, planet);
        world.sun_occlusion = true;
        // The sun 50 degrees over the eye's horizon.
        let up = view.eye.normalize();
        let across = up.cross(side).normalize();
        world.sun = (up * 50f64.to_radians().sin() + across * 50f64.to_radians().cos())
            .normalize()
            .as_vec3();
        let hits = world.settle(gpu, planet, &view);
        world.settle_sun_pages(gpu, planet, &view);
        eprintln!("{name}: settled at {:.1} s", clock.elapsed().as_secs_f64());
        let pages = world.chunks.read_sun_occlusion().expect("drawn");
        let view_proj = view.view_proj(planet);
        let (mut read, mut wide, mut none, mut worst) = (0usize, 0usize, 0usize, 0.0f64);
        for y in (0..view.size[1]).step_by(2) {
            for x in (0..view.size[0]).step_by(2) {
                let Some(hit) = hits[(y * view.size[0] + x) as usize] else {
                    continue;
                };
                // The pixel's width there, as the shadow pass takes it
                // (`pixel_width`: its ray's step to the next pixel's).
                let d = view.pixel_dir(view_proj, x, y);
                let footprint = hit.distance * (view.pixel_dir(view_proj, x + 1, y) - d).length();
                // Cells of a pixel and a half or more walk their own sun.
                if planet.grid().voxel_size() * f64::from(1u32 << hit.level) >= 1.5 * footprint {
                    continue;
                }
                let p = d * hit.distance;
                let Some(r) = pages.read(p.as_vec3().to_array(), [0.0; 3], footprint as f32) else {
                    none += 1;
                    continue;
                };
                read += 1;
                let ratio = f64::from(r.texel) / footprint;
                worst = worst.max(ratio);
                wide += usize::from(ratio > 2.0);
            }
        }
        eprintln!(
            "{name}: {read} read, {none} unread, {wide} wider than two footprints (worst {worst:.2})"
        );
        assert!(read > 1000, "{name}: {read} read");
        assert!(worst <= 2.0, "{name}: a texel {worst:.2} footprints wide");
        assert!(none * 100 <= read, "{name}: {none} unread of {read}");
        assert!(
            wide * 100 <= read,
            "{name}: {wide} of {read} wider than two footprints"
        );
    }
}

/// The sun's occlusion pages under a canopy against the walk's exact answer
/// for the same points: GI-shaped receivers (hits of rays from the finest
/// probes' places about an eye standing in a forest, a centimetre off their
/// face), each against an occlusion ray along the epoch's sun walked through
/// the same cut.
///
/// - **What the pages hold**: at every level 0..12 around the eye (each
///   level's always-resident pages), each receiver's nearest texel's centre
///   ray, walked from above the texel's stored occluder, meets what the
///   texel stores (within two texels), at 99 % of texels or more: the
///   trace is exact where it samples.
/// - **Why their read leaks** (printed): the pages' 2x2 read at each level,
///   with the receiver's normal and without, against the walk, and of the
///   receivers it lights that the walk shades, why: every one within the
///   read's bias of a texel (an occluder less than a texel over the receiver
///   along the sun), at every level.
/// - **The footprint answer** (`paged_occlusion`): the same receivers'
///   occlusion rays as wide as GI's shadow rays (the probe ray's footprint
///   at the hit, `GI_RAY_SPREAD` a metre), walked two texels of the level
///   their width reads and answered by the pages past that, let through on
///   average what the walk does (within 0.02, by level), and differ from it
///   at no more than one receiver in twelve (the pages' texels are point
///   samples of a footprint the walk's one ray does not average).
#[test]
fn sun_pages_under_a_canopy() {
    use helio_frame_api::rays::stream::{unpack_unit, RayClass};
    let Some(gpu) = gpu() else { return };
    let (planet, dir) = forest_planet();
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    let view = View::new(eye, (fwd - up * 0.25).normalize().as_vec3(), [320, 180]);
    let mut world = ray_world(gpu, &planet);
    world.sun_occlusion = true;
    // 50 degrees up.
    let elevation = 50f64.to_radians();
    world.sun = (up * elevation.sin() + side * elevation.cos())
        .normalize()
        .as_vec3();
    world.settle(gpu, &planet, &view);
    world.settle_sun_pages(gpu, &planet, &view);
    let pages = world.chunks.read_sun_occlusion().expect("drawn");
    let ep = &pages.frame.epochs[0];
    let sun = -DVec3::new(
        f64::from(ep.forward[0]),
        f64::from(ep.forward[1]),
        f64::from(ep.forward[2]),
    );
    // Receivers: probe rays' hits within 12 m, and the probe ray's
    // footprint there.
    let mut unit = units(0x3c6e_f372_fe94_f82b);
    let mut rays = Vec::new();
    for z in 9..15 {
        for y in 9..15 {
            for x in 9..15 {
                let o = probe_place(eye, 0, glam::I64Vec3::new(x, y, z));
                if planet.solid(planet.grid().locate(o).0) {
                    continue;
                }
                for _ in 0..32 {
                    rays.push((o, unit()));
                }
            }
        }
    }
    let stream: Vec<SceneRay> = rays
        .iter()
        .map(|&(o, d)| common::scene_ray(eye, o, d, 12.0, 0.0))
        .collect();
    let (hits, surfaces) =
        common::trace_stream(gpu, &world.chunks, RayClass::Nearest, eye, &stream, None, 0);
    let mut receivers = Vec::new();
    for (k, &(o, d)) in rays.iter().enumerate() {
        let Some((t, false)) = terrain_hit(&hits[k]) else {
            continue;
        };
        let face = DVec3::from_array(unpack_unit(surfaces[k].face).map(f64::from));
        receivers.push((o + d * t + face * 0.01, face, 0.443 * t));
    }
    // The walk toward the epoch's sun (no width: walked all the way).
    let walk = |width: bool| -> Vec<SceneHit> {
        let rays: Vec<SceneRay> = receivers
            .iter()
            .map(|&(p, _, w)| {
                let mut ray = common::scene_ray(eye, p, sun, 1e30, 0.0);
                if width {
                    ray.cone =
                        SceneRay::cone(w as f32, 0.0, helio_frame_api::rays::stream::MASK_ALL);
                }
                ray
            })
            .collect();
        common::trace_stream(gpu, &world.chunks, RayClass::Occlusion, eye, &rays, None, 0).0
    };
    let walked = walk(false);
    let walk_lit: Vec<bool> = walked.iter().map(|h| h.t <= 0.0).collect();
    let shaded = walk_lit.iter().filter(|l| !**l).count();
    eprintln!("{} receivers, {shaded} shaded by the walk", receivers.len());
    assert!(
        receivers.len() > 2000 && shaded > 500,
        "{} receivers, {shaded} shaded",
        receivers.len()
    );
    let finest = pages.frame.info[0];
    let v3 = |a: [f32; 4]| DVec3::new(f64::from(a[0]), f64::from(a[1]), f64::from(a[2]));
    let (right, down) = (v3(ep.right), v3(ep.down));
    for level in 0..13usize {
        let texel = finest * (level as f32).exp2();
        let (mut n, mut walk_sum, mut normal_sum, mut bare_sum) = (0usize, 0.0, 0.0, 0.0);
        let (mut wrong_lit, mut in_bias, mut wrong_dark) = (0usize, 0usize, 0usize);
        // The nearest texels' centre rays: from 30 m up the sun from the
        // stored occluder (or the receiver), down to the receiver's depth.
        let mut centres = Vec::new();
        let mut stored = Vec::new();
        for (k, &(p, face, _)) in receivers.iter().enumerate() {
            let rel = (p - eye).as_vec3().to_array();
            let Some(with) = pages.read_level(rel, face.as_vec3().to_array(), 0, level) else {
                continue;
            };
            let Some(bare) = pages.read_level(rel, [0.0; 3], 0, level) else {
                continue;
            };
            n += 1;
            walk_sum += f64::from(u8::from(walk_lit[k]));
            normal_sum += f64::from(with.lit);
            bare_sum += f64::from(bare.lit);
            if bare.lit >= 0.5 && !walk_lit[k] {
                wrong_lit += 1;
                in_bias += usize::from(bare.stored > 0.0 && bare.depth <= bare.stored + texel);
            }
            wrong_dark += usize::from(bare.lit < 0.5 && walk_lit[k]);
            let across = [0, 1].map(|c| f64::from((0.5 - with.frac[c]) * texel));
            let at_depth = p + right * across[0] + down * across[1];
            let top = f64::from(if bare.stored > 0.0 {
                bare.stored.min(bare.depth)
            } else {
                bare.depth
            }) - 30.0;
            let length = f64::from(bare.depth) - top;
            centres.push((at_depth + sun * length, length));
            stored.push((bare.stored, bare.depth));
        }
        if n == 0 {
            continue;
        }
        let centre_rays: Vec<SceneRay> = centres
            .iter()
            .map(|&(s, length)| common::scene_ray(eye, s, -sun, (length + 2.0) as f32, 0.0))
            .collect();
        let (centre_hits, _) = common::trace_stream(
            gpu,
            &world.chunks,
            RayClass::Nearest,
            eye,
            &centre_rays,
            None,
            0,
        );
        let mut agree = 0;
        for (k, &(s, d)) in stored.iter().enumerate() {
            let length = centres[k].1;
            let met = terrain_hit(&centre_hits[k]).map(|(t, _)| f64::from(d) - length + t);
            let (s, d) = (f64::from(s), f64::from(d));
            agree += usize::from(match (s > 0.0 && s < d + 1.0, met) {
                (true, Some(m)) => (m - s).abs() <= 2.0 * f64::from(texel) + 0.2,
                (true, None) => false,
                (false, Some(m)) => m >= d - 0.05,
                (false, None) => true,
            });
        }
        eprintln!(
            "level {level:2} (texel {texel:.3} m): {n} read; lit: walk {:.3}, pages with the \
             normal {:.3}, without {:.3}; wrongly lit {wrong_lit} ({in_bias} within the bias), \
             wrongly dark {wrong_dark}; texel centres agreeing with the walk {agree}",
            walk_sum / n as f64,
            normal_sum / n as f64,
            bare_sum / n as f64,
        );
        assert!(
            agree * 100 >= n * 99,
            "level {level}: {agree} of {n} texels agree"
        );
    }
    // The footprint answer, by the level each ray's width reads.
    let answered = walk(true);
    let mut by_level = std::collections::BTreeMap::<usize, (usize, f64, f64, usize)>::new();
    for (k, &(_, _, w)) in receivers.iter().enumerate() {
        let level = helio_light_api::sun_occlusion_level(finest, 20, 0.5 * w as f32);
        let lit = f64::from(answered[k].through()[0]);
        let walk = f64::from(u8::from(walk_lit[k]));
        let e = by_level.entry(level).or_default();
        e.0 += 1;
        e.1 += lit;
        e.2 += walk;
        e.3 += usize::from((lit >= 0.5) != walk_lit[k]);
    }
    let (mut all, mut wrong, mut lit_sum, mut walk_sum) = (0usize, 0usize, 0.0, 0.0);
    for (&level, &(n, lit, walked, differ)) in &by_level {
        eprintln!(
            "footprint answer, level {level:2}: {n} rays, lit {:.3} against the walk's {:.3}, \
             {differ} differ",
            lit / n as f64,
            walked / n as f64
        );
        if n >= 100 {
            assert!(
                (lit - walked).abs() / n as f64 <= 0.02,
                "level {level}: lit {:.3}, the walk {:.3}",
                lit / n as f64,
                walked / n as f64
            );
        }
        all += n;
        wrong += differ;
        lit_sum += lit;
        walk_sum += walked;
    }
    eprintln!(
        "footprint answer: {all} rays, lit {:.3} against {:.3}, {wrong} differ",
        lit_sum / all as f64,
        walk_sum / all as f64
    );
    assert!((lit_sum - walk_sum).abs() / all as f64 <= 0.01);
    assert!(wrong * 12 <= all, "{wrong} of {all} differ");
}

/// Ground probes for foliage (`chunks::ground`) over a 3x3 block of tiles around the
/// eye, 16x16 queries a tile, as `FoliagePlacePass` asks them: queries in render space
/// (origin at the eye), each a segment 768 m down its tile's up from 256 m over the eye.
struct Probes {
    starts: Vec<DVec3>,
    ups: Vec<DVec3>,
    tiles: Vec<helio_foliage_api::AskedTile>,
    queries: wgpu::Buffer,
    hits: wgpu::Buffer,
}

const PROBE_SIDE: u32 = 16;
const PROBE_LENGTH: f32 = 768.0;

fn ground_probes(gpu: &Gpu, planet: &Planet, eye: DVec3) -> Probes {
    use helio_foliage_api::{AskedTile, GpuGroundQuery, GroundGeometry};
    use wgpu::util::DeviceExt;
    let ground = helio_pass_voxel_planet::chunks::ground::PlanetGround::default();
    ground.set_grid(*planet.grid());
    let chart = ground.chart(eye).unwrap();
    let centre = [
        (chart.eye[0] / 8.0).floor() as i32,
        (chart.eye[1] / 8.0).floor() as i32,
    ];
    let (mut starts, mut ups, mut tiles, mut queries) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for dj in -1..=1 {
        for di in -1..=1 {
            let frame = ground
                .tile_frame(chart.id, [centre[0] + di, centre[1] + dj], 1)
                .unwrap();
            let start_height = (eye - frame.origin).dot(frame.up) + 256.0;
            tiles.push(AskedTile {
                frame,
                start_height,
                first: queries.len() as u32,
                count: PROBE_SIDE * PROBE_SIDE,
            });
            for b in 0..PROBE_SIDE {
                for a in 0..PROBE_SIDE {
                    let (u, v) = (
                        (f64::from(a) + 0.37) / f64::from(PROBE_SIDE),
                        (f64::from(b) + 0.61) / f64::from(PROBE_SIDE),
                    );
                    let start = frame.origin
                        + frame.axis_u * u
                        + frame.axis_v * v
                        + frame.up * start_height;
                    starts.push(start);
                    ups.push(frame.up);
                    queries.push(GpuGroundQuery {
                        start: (start - eye).as_vec3().to_array(),
                        length: PROBE_LENGTH,
                        up: frame.up.as_vec3().to_array(),
                        _pad: 0,
                    });
                }
            }
        }
    }
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let hits = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: queries.len() as u64 * 16,
        usage,
        mapped_at_creation: false,
    });
    let queries = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&queries),
            usage,
        });
    Probes {
        starts,
        ups,
        tiles,
        queries,
        hits,
    }
}

/// The probes answered against the world's last drawn cut (eye at render origin).
fn probe(
    gpu: &Gpu,
    world: &mut World,
    planet: &Planet,
    probes: &Probes,
    eye: DVec3,
) -> Vec<helio_foliage_api::GpuGroundHit> {
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let mut recorder = helio_core::CommandRecorder::from_encoder(&mut encoder);
    let answered = world.chunks.probe_ground(
        &mut recorder,
        &probes.queries,
        &probes.hits,
        &probes.tiles,
        planet,
        eye,
        Vec3::ZERO,
    );
    assert!(answered, "a drawn world answers ground queries");
    drop(recorder);
    gpu.queue.submit([encoder.finish()]);
    let bytes = common::read_buffer(gpu, &probes.hits, probes.starts.len() as u64 * 16);
    bytemuck::cast_slice(&bytes).to_vec()
}

/// Probe hits against CPU ray casts down the same segments: where the CPU meets a
/// top face within the level-0 range, the GPU meets it at the same distance with the
/// cell's material; where it meets a wall or nothing, no ground. (compared, wrong.)
fn probes_match(
    planet: &Planet,
    probes: &Probes,
    hits: &[helio_foliage_api::GpuGroundHit],
    eye: DVec3,
    near: f64,
) -> (usize, usize) {
    let (mut compared, mut wrong) = (0, 0);
    for (n, hit) in hits.iter().enumerate() {
        let (start, up) = (probes.starts[n], probes.ups[n]);
        let cpu = planet.raycast(start, -up, f64::from(PROBE_LENGTH));
        let Some(cpu) = cpu else {
            compared += 1;
            wrong += usize::from(hit.ground != helio_foliage_api::GROUND_NONE);
            continue;
        };
        let at = start - up * cpu.distance;
        if (at - eye).length() > near {
            continue;
        }
        compared += 1;
        let top = cpu.normal.dot(up) > 0.5;
        let gpu = f32::from_bits(hit.distance);
        let ok = if top {
            hit.ground == planet.material(cpu.cell) && (f64::from(gpu) - cpu.distance).abs() < 2e-3
        } else {
            hit.ground == helio_foliage_api::GROUND_NONE
        };
        if !ok {
            wrong += 1;
            if wrong < 8 {
                eprintln!(
                    "query {n}: gpu {gpu} ground {}, cpu {:?} at {} top {top} material {}",
                    hit.ground,
                    cpu.cell,
                    cpu.distance,
                    planet.material(cpu.cell)
                );
            }
        }
    }
    (compared, wrong)
}

/// Foliage's ground (`chunks::ground`): probes down the drawn cut meet the terrain's
/// top faces where the CPU does, at the same distance, with their cells' materials;
/// a still eye's world reports no ground changes and answers the same; a dig reports
/// its box and the probes over it meet the pit's floor.
#[test]
fn ground_probes_meet_the_drawn_surface() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 2, 0.47, 0.53);
    let view = ground_view(&planet, dir, [320, 180]);
    let near = 0.7 * planet.grid().voxel_size() / view.pixel_angle();
    let mut world = World::new(&gpu, &planet);
    world.settle(&gpu, &planet, &view);
    let probes = ground_probes(&gpu, &planet, view.eye);
    let hits = probe(&gpu, &mut world, &planet, &probes, view.eye);
    let (compared, wrong) = probes_match(&planet, &probes, &hits, view.eye, near);
    let grounded = hits
        .iter()
        .filter(|h| h.ground != helio_foliage_api::GROUND_NONE)
        .count();
    eprintln!(
        "compared {compared}, wrong {wrong}, {grounded} of {} on ground",
        hits.len()
    );
    assert!(
        compared > 1000 && grounded > 1000,
        "{compared} compared, {grounded} grounded"
    );
    assert_eq!(wrong, 0);
    // Still: no changes, the same answers.
    world.chunks.take_ground_changes();
    for _ in 0..(4 + 17) {
        world.frame(&gpu, &planet, &view, None);
    }
    let quiet = world.chunks.take_ground_changes();
    assert!(
        quiet.is_empty(),
        "{} ground changes under a still eye",
        quiet.len()
    );
    assert_eq!(probe(&gpu, &mut world, &planet, &probes, view.eye), hits);
    // A dig 3 m from the eye's column: its box changes, and the probes see the pit.
    let up = view.eye.normalize();
    let ground = planet.surface_point(dir, 0.0) + up.any_orthonormal_vector() * 3.0;
    planet
        .apply(Brush {
            center: ground.to_array(),
            radius: 1.5,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    world.sync(&gpu, &planet);
    world.settle(&gpu, &planet, &view);
    let changes = world.chunks.take_ground_changes();
    assert!(
        changes
            .iter()
            .any(|[lo, hi]| ground.cmpge(*lo - DVec3::splat(1e-6)).all()
                && ground.cmple(*hi + DVec3::splat(1e-6)).all()),
        "no change over the dig in {}",
        changes.len()
    );
    let dug = probe(&gpu, &mut world, &planet, &probes, view.eye);
    let (compared, wrong) = probes_match(&planet, &probes, &dug, view.eye, near);
    let moved = dug.iter().zip(&hits).filter(|(a, b)| a != b).count();
    eprintln!("after the dig: compared {compared}, wrong {wrong}, {moved} answers changed");
    assert!(moved > 20, "{moved} answers changed over the pit");
    assert_eq!(wrong, 0);
}

/// The Moon's pole over the crater-strewn plain as voxel_flight's moon
/// preset draws it (seed 75; the renderer's residency and near plane, a
/// quarter of the height over the ground: 74 m), the eye on the plane
/// between the face's middle columns. Pixels seeded by any of the many
/// boxes the cull counts as holding the eye once walked from that box,
/// past the ground in front: chunk-sized walls and slabs. Rays nearly
/// parallel to that plane, seeded in a box they never enter in its frame
/// or fallen between two boxes' tops to the Moon's far side, once crossed
/// the ground untraced and ended on Solid nodes; rays nearly parallel to
/// any side two chunks share went on under the ground between their
/// frames. Once residency settles, no primary ray ends on a Solid or
/// Buried node, and every drawn hit is a solid cell entered from air as
/// the cut draws it (at the level of the chunk the ray came from), but
/// along an edge with air, where frames meet to a 64th of a cell.
#[test]
fn moon_pole_rays_end_on_chunks() {
    use rayon::prelude::*;
    let Some(gpu) = gpu() else { return };
    let voxel = 0.1;
    let planet = Planet::new(PlanetRecipe {
        radius_m: 1_737_400.0,
        voxel_size_m: voxel,
        terrain: TerrainLayers::moon().source(75),
        features: helio_pass_voxel_planet::feature_recipe::FeatureRecipe::moon().grow(voxel, 75),
        ..Default::default()
    })
    .unwrap();
    let mut view = View::new(
        DVec3::new(0.0, 1_738_104.0, 0.0),
        Vec3::new(-0.073_291_83, -0.573_576_45, 0.815_866_65),
        [1600, 900],
    );
    // The camera's near plane as hosts set it (a quarter of the height
    // over the ground).
    view.near = (planet.air_clearance(view.eye) * 0.25).clamp(0.05, 50_000.0) as f32;
    eprintln!("near plane {:.2} m", view.near);
    // The view settles with ~37k chunks resident (past TEST_CAPACITY's).
    let mut world = World::with_capacity(
        &gpu,
        &planet,
        Capacity {
            chunks: 49152,
            detail_pages: 32768,
            counters: true,
            ..TEST_CAPACITY
        },
    );
    world.jitter = Some(0);
    // Residency as the renderer's pass runs it (generation within its
    // budget a frame, over hundreds of frames).
    let config = ResidencyConfig {
        plan: residency::PlanConfig {
            pixel_angle: view.pixel_angle(),
            ..ResidencyConfig::default().plan
        },
        ..ResidencyConfig::default()
    };
    let started = world.frame;
    while !world.chunks.settled() {
        let stats = world.frame_with(&gpu, &planet, &view, &config);
        world.chunks.take_counts();
        assert!(
            world.frame - started < 4000,
            "residency did not settle: {stats:?}"
        );
    }
    let mut ends = [0u32; 2];
    for _ in 0..30 {
        world.frame_with(&gpu, &planet, &view, &config);
        let counts = world.chunks.take_counts();
        ends = [0, 1].map(|b| ends[b] + counts.rock_ends[b]);
        if counts.rock_ends != [0, 0] {
            eprintln!(
                "rock/buried ends {:?} at pixels {:?}, nodes {:?}",
                counts.rock_ends, counts.rock_pixels, counts.rock_nodes
            );
        }
    }
    // Every 2nd pixel's hit against the CPU's cells at its level.
    let hits = world.chunks.read_hits();
    let [w, h] = view.size;
    let wrong: Vec<_> = (0..(w / 2) * (h / 2))
        .into_par_iter()
        .filter_map(|n| {
            let (x, y) = ((n % (w / 2)) * 2, (n / (w / 2)) * 2);
            let hit = hits[(y * w + x) as usize]?;
            let c = hit.cell;
            let solid = |level: u32, cell: [i32; 3]| {
                planet
                    .sample_kind(level, c.face, cell[0], cell[1], cell[2])
                    .0
                    == 1
            };
            let mut before = [c.i, c.j, c.k];
            if let Some(code) = hit.entered {
                before[(code >> 1) as usize] += if code & 1 == 1 { -1 } else { 1 };
            }
            // Entered from air: at its level, or across a switch from a
            // level beside it (a coarser cell rounds its top down, a finer
            // one holds air the coarse cell does not).
            let from_air = || {
                !solid(hit.level, before)
                    || !solid(hit.level + 1, before.map(|v| v >> 1))
                    || hit.level > 0
                        && (0..8).any(|n| {
                            !solid(
                                hit.level - 1,
                                [0, 1, 2].map(|a| before[a] * 2 + (n >> a & 1)),
                            )
                        })
            };
            let right = solid(hit.level, [c.i, c.j, c.k]) && (hit.entered.is_none() || from_air());
            (!right).then_some((x, y, hit))
        })
        .collect();
    // Of those, the ones entered from a cell the cut draws as solid (at the
    // level of the drawn chunk it lies in; none: air, an Empty node's),
    // with the hit on its face past its edges with air cells by more than
    // a 64th of a cell: a ray along such an edge meets either cell (each
    // chunk's frame places its sides to about that, as the raster's
    // `SEED_REACH`).
    let view_proj = view.view_proj(&planet);
    let wrong: Vec<_> = wrong
        .into_iter()
        .filter(|&(x, y, hit)| {
            let c = hit.cell;
            let solid = |level: u32, cell: [i32; 3]| {
                planet
                    .sample_kind(level, c.face, cell[0], cell[1], cell[2])
                    .0
                    == 1
            };
            if !solid(hit.level, [c.i, c.j, c.k]) {
                return true;
            }
            let Some(code) = hit.entered else { return true };
            let axis = (code >> 1) as usize;
            let mut before = [c.i, c.j, c.k];
            before[axis] += if code & 1 == 1 { -1 } else { 1 };
            // The base cell at the middle of the one entered from.
            let base = before.map(|v| (v << hit.level) + ((1 << hit.level) >> 1));
            let drawn_solid = world
                .chunks
                .chunks_over(VolumeId(0), c.face, base[0], base[1])
                .into_iter()
                .find(|(key, _, drawn)| *drawn && (base[2] >> key.level).div_euclid(64) == key.cz)
                .is_some_and(|(key, _, _)| solid(key.level, base.map(|v| v >> key.level)));
            if !drawn_solid {
                return false;
            }
            let at = view.eye + view.pixel_dir(view_proj, x, y) * hit.distance;
            let Some(q) = planet.grid().face_coords(c.face, at) else {
                return true;
            };
            let q = q.map(|v| v / f64::from(1u32 << hit.level));
            let edge = (0..3).filter(|&a| a != axis).any(|a| {
                let f = q[a] - f64::from(before[a]);
                let (mut lo, mut hi) = (before, before);
                lo[a] -= 1;
                hi[a] += 1;
                (f < 1.0 / 64.0 && !solid(hit.level, lo))
                    || (f > 1.0 - 1.0 / 64.0 && !solid(hit.level, hi))
            });
            !edge
        })
        .collect();
    for &(x, y, hit) in wrong.iter().take(8) {
        let cpu = planet.raycast(view.eye, view.pixel_dir(view_proj, x, y), 600.0);
        eprintln!(
            "pixel {x},{y}: drawn {hit:?}; cpu {:?}",
            cpu.map(|c| (c.distance, c.cell, c.previous))
        );
    }
    eprintln!(
        "settled in {} frames; rock/buried ends over 30 frames {ends:?}; {} of {} hits wrong",
        world.frame - started - 30,
        wrong.len(),
        (w / 2) * (h / 2)
    );
    assert_eq!(ends, [0, 0], "rays ended on rock and buried boxes");
    assert!(
        wrong.is_empty(),
        "{} hits on cells the CPU has as air, or entered from cells the cut draws as rock",
        wrong.len()
    );
}

/// The desert at 10 cm voxels (seed 7), an eye 3 m over its ground at
/// face 4's (0.41, 0.57), and the face's cell rows there (i, j: unit,
/// horizontal).
fn desert_steps() -> (Planet, DVec3, DVec3, DVec3) {
    let planet = Planet::new(PlanetRecipe {
        voxel_size_m: 0.1,
        terrain: TerrainLayers::desert().source(7),
        ..Default::default()
    })
    .unwrap();
    let grid = planet.grid();
    let dir = land(&planet, 4, 0.41, 0.57);
    let eye = planet.surface_point(dir, 3.0);
    let up = grid.up(eye);
    let (cell, _) = grid.locate(eye);
    let at = [f64::from(cell.i), f64::from(cell.j), f64::from(cell.k)];
    let along = |a: usize| {
        let mut b = at;
        b[a] += 1.0;
        let v = grid.position(cell.face, b) - grid.position(cell.face, at);
        (v - up * v.dot(up)).normalize()
    };
    let (row_i, row_j) = (along(0), along(1));
    (planet, eye, row_i, row_j)
}

/// Every second pixel's hit (each way) of cells three pixels wide or more
/// in `view`, under each of `suns` (10 degrees up toward each direction),
/// with step softness 0 (each step lit by its own faces, as levels storing
/// `detail[3] = 0` draw), a point sun and steady shadows: its walked sun
/// against the CPU's ray from a hundredth of a cell off its face, wherever
/// the CPU's rays from points a few thousandths of a cell around that agree
/// (a ray along an edge meets either side of it). The suns with wrong hits,
/// and how many.
fn sun_rays_against_cpu(
    gpu: &Gpu,
    planet: &Planet,
    view: &View,
    suns: &[(&str, DVec3)],
) -> Vec<(String, usize)> {
    use rayon::prelude::*;
    let grid = planet.grid();
    let up = grid.up(view.eye);
    let mut world = World::new(gpu, planet);
    let appearance = helio_pass_voxel_planet::engine::TerrainAppearance {
        detail: [0.3, 0.4, 0.08, 0.0],
        ..Default::default()
    };
    world.chunks.set_appearance(VolumeId(0), &appearance);
    world.chunks.set_sun_angular_radius(0.0);
    world.chunks.set_steady_shadows(true);
    let low = 10f64.to_radians();
    let view_proj = view.view_proj(planet);
    let [w, h] = view.size;
    let mut failed = Vec::new();
    for &(name, toward) in suns {
        let toward = (toward.normalize() * low.cos() + up * low.sin()).normalize();
        world.sun = toward.as_vec3();
        let hits = world.settle(gpu, planet, view);
        for _ in 0..3 {
            world.frame(gpu, planet, view, None);
        }
        let walked = read_sun_walked(gpu, &world);
        let ties = std::sync::atomic::AtomicUsize::new(0);
        // Every second pixel each way.
        let checked: Vec<_> = (0..(w / 2) * (h / 2))
            .into_par_iter()
            .filter_map(|n| {
                let (x, y) = ((n % (w / 2)) * 2, (n / (w / 2)) * 2);
                let n = y * w + x;
                let hit = hits[n as usize]?;
                let code = hit.entered?;
                let size = grid.voxel_size() * f64::from(1u32 << hit.level);
                if size < 3.0 * hit.distance * view.pixel_angle() {
                    return None;
                }
                let p = view.eye + view.pixel_dir(view_proj, x, y) * hit.distance;
                let c = hit.cell;
                let q = grid.face_coords(c.face, p)?;
                let axis = (code >> 1) as usize;
                // A cell's axis there (the GPU offsets a ray along it).
                let unit = |a: usize| {
                    let mut b = q;
                    b[a] += 1e-3;
                    (grid.position(c.face, b) - grid.position(c.face, q)) * 1e3
                };
                // Entered stepping up an axis: through its low side.
                let normal = unit(axis).normalize() * if code & 1 == 1 { -1.0 } else { 1.0 };
                let facing = normal.dot(toward) > 0.0;
                let lit_from = |o: DVec3| facing && planet.raycast(o, toward, 60.0).is_none();
                let o = p + normal * (0.01 * size);
                let cpu = lit_from(o);
                let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);
                let (u1, u2) = (unit(t1).normalize(), unit(t2).normalize());
                let tie = [u1, -u1, u2, -u2, normal * 0.5, -normal * 0.5]
                    .iter()
                    .any(|&v| lit_from(o + v * (0.004 * size)) != cpu);
                if tie {
                    ties.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return None;
                }
                Some((x, y, hit, cpu, walked[n as usize] > 0.5))
            })
            .collect();
        let wrong: Vec<_> = checked.iter().filter(|c| c.3 != c.4).collect();
        let dark = wrong.iter().filter(|c| c.3).count();
        eprintln!(
            "sun {name}: {} hits checked ({} ties left out), {} lit; {} wrong ({dark} dark where the CPU's ray sees the sun)",
            checked.len(),
            ties.into_inner(),
            checked.iter().filter(|c| c.3).count(),
            wrong.len()
        );
        for &&(x, y, hit, cpu, _) in wrong.iter().take(12) {
            eprintln!(
                "  ({x}, {y}): level {} cell {:?} entered {:?} at {:.3} m; cpu lit {cpu}",
                hit.level, hit.cell, hit.entered, hit.distance
            );
        }
        assert!(checked.len() > 10_000, "{} hits checked", checked.len());
        if !wrong.is_empty() {
            failed.push((name.to_string(), wrong.len()));
        }
    }
    failed
}

/// Sun rays from drawn hits against CPU rays, on the desert's stepped
/// ground under low suns (`sun_rays_against_cpu`): along the face's cell
/// rows (the walls of every step and every chunk side parallel to them), a
/// fraction of a degree off them, and across them.
#[test]
fn sun_rays_from_steps_match_cpu_rays() {
    let Some(gpu) = gpu() else { return };
    let (planet, eye, row_i, row_j) = desert_steps();
    let up = planet.grid().up(eye);
    let view = View::new(
        eye,
        (row_i * 0.8 + row_j * 0.6 - up * 0.6).normalize().as_vec3(),
        [640, 360],
    );
    let off = 0.3f64.to_radians();
    let suns = [
        ("along i", row_i),
        ("along j", -row_j),
        ("0.3 degrees off i", row_i * off.cos() + row_j * off.sin()),
        ("across", row_i - row_j * 0.7),
    ];
    let failed = sun_rays_against_cpu(gpu, &planet, &view, &suns);
    assert!(failed.is_empty(), "sun rays off the CPU's: {failed:?}");
}

/// As `sun_rays_from_steps_match_cpu_rays`, over walls on chunk sides: two
/// trenches dug half a metre into the desert, crossing at a chunk's corner
/// (each one's wall on the chunk side beside it), and a block raised on the
/// next chunk corner, seen from 2 m over them looking steeply down (as an
/// editor's camera), with the host's near plane.
#[test]
fn sun_rays_from_walls_on_chunk_sides_match_cpu_rays() {
    let Some(gpu) = gpu() else { return };
    let (mut planet, eye, row_i, row_j) = desert_steps();
    let (cell, _) = planet.grid().locate(eye);
    let face = cell.face;
    let (i0, j0) = (
        f64::from((cell.i >> 6) * 64 + 64),
        f64::from((cell.j >> 6) * 64 + 64),
    );
    let ground = |planet: &Planet, i: f64, j: f64| {
        planet.surface_point(planet.grid().direction(face, i, j), 0.0)
    };
    let corner = ground(&planet, i0, j0);
    let up = planet.grid().up(corner);
    // Cube brushes are squares of the face's cells: one 0.3 m in radius
    // centred 3 cells past a chunk side spans the 6 cells from it.
    let mut boxes = Vec::new();
    for n in -5..6 {
        let n = f64::from(n) * 6.0;
        boxes.push((i0 + 3.0, j0 + n, BrushOp::Remove));
        boxes.push((i0 + n, j0 + 3.0, BrushOp::Remove));
    }
    // 8 cells square over the chunk corner 64 cells along i.
    boxes.push((i0 + 64.0, j0, BrushOp::Add));
    for &(i, j, op) in &boxes {
        let at = ground(&planet, i, j);
        let up = planet.grid().up(at);
        let add = op == BrushOp::Add;
        planet
            .apply(Brush {
                center: (at + up * if add { 0.1 } else { 0.25 }).to_array(),
                radius: if add { 0.4 } else { 0.3 },
                shape: BrushShape::Cube,
                op,
                material: if add { material::BRICK } else { 0 },
                height: if add { 0.3 } else { 0.5 },
            })
            .unwrap();
    }
    let eye = corner + up * 2.0 - row_i * 1.2 - row_j * 1.6;
    let forward = (corner + row_i * 0.6 - eye).normalize().as_vec3();
    let near = (planet.air_clearance(eye) * 0.25).clamp(0.05, 50_000.0) as f32;
    // The hits themselves first: the CPU's cells.
    matches_exactly(
        gpu,
        &planet,
        View {
            near,
            ..View::new(eye, forward, [640, 360])
        },
        20_000,
    );
    let view = View {
        near,
        ..View::new(eye, forward, [640, 360])
    };
    let off = 0.3f64.to_radians();
    let suns = [
        ("along i", row_i),
        ("along j", -row_j),
        ("0.3 degrees off j", -row_j * off.cos() + row_i * off.sin()),
        ("across", row_i - row_j * 0.7),
        ("diagonal", row_i + row_j),
    ];
    let failed = sun_rays_against_cpu(gpu, &planet, &view, &suns);
    assert!(failed.is_empty(), "sun rays off the CPU's: {failed:?}");
}

/// Smoothing changes how a hit is lit, not where it is: a dug bowl in the
/// ground (edited cells, no natural steps) drawn blocky and then smooth hits
/// the same cells at the same distances, and its sun stays a valid
/// visibility that smoothing changes somewhere (the shading normals).
#[test]
fn smooth_surfaces_are_lit_not_moved() {
    let Some(gpu) = gpu() else { return };
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [320, 180]);
    let bowl = view.eye + view.forward.as_dvec3() * 8.0;
    planet
        .apply(Brush {
            center: bowl.to_array(),
            radius: 3.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    let mut world = World::new(gpu, &planet);
    world.chunks.set_steady_shadows(true);
    let blocky = world.settle(gpu, &planet, &view);
    let blocky_sun = read_sun_visibility(gpu, &world);

    let smooth_look = helio_pass_voxel_planet::engine::TerrainAppearance {
        smoothing: 1.0,
        ..Default::default()
    };
    world.chunks.set_appearance(VolumeId(0), &smooth_look);
    for _ in 0..4 {
        world.frame(gpu, &planet, &view, None);
    }
    let smooth = world.settle(gpu, &planet, &view);
    let smooth_sun = read_sun_visibility(gpu, &world);

    assert_eq!(blocky.len(), smooth.len());
    let moved = blocky.iter().zip(&smooth).filter(|(a, b)| a != b).count();
    assert_eq!(moved, 0, "smoothing moved {moved} hits");
    assert!(
        smooth_sun
            .iter()
            .all(|v| v.is_finite() && (-1e-3..=1.001).contains(v)),
        "smooth sun visibility is not a visibility"
    );
    let relit = blocky_sun
        .iter()
        .zip(&smooth_sun)
        .filter(|(a, b)| (*a - *b).abs() > 1e-3)
        .count();
    eprintln!("smoothing relit {relit} of {} pixels", smooth_sun.len());
    assert!(relit > 0, "smoothing changed no pixel's light");
}

// ---- Ray streams (`chunks::rays`) ----

use helio_frame_api::rays::stream::{
    unpack_radiance, unpack_unit, RayClass, SceneHit, SceneRay, SceneSurface,
};

/// A world drawn for ray streams: its ray kernels compiled.
fn ray_world(gpu: &Gpu, planet: &Planet) -> World {
    let mut world = World::new(gpu, planet);
    world.chunks.want_rays();
    world
        .chunks
        .wait_for_pipelines()
        .expect("the ray kernels compile");
    world
}

/// The terrain's hit in row `hit`, if any: its distance and whether it is
/// from inside.
fn terrain_hit(hit: &SceneHit) -> Option<(f64, bool)> {
    (hit.t > 0.0 && hit.tracer().0 == common::VOXEL_TRACER)
        .then(|| (f64::from(hit.t), hit.inside()))
}

/// Whether a CPU hit `distance` along `dir` from `origin` lies in view near
/// the eye, where level 0 is drawn.
fn near_in_view(view: &View, planet: &Planet, origin: DVec3, dir: DVec3, distance: f64) -> bool {
    let at = origin + dir * distance;
    let clip = view.view_proj(planet) * (at - view.eye).as_vec3().extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let near = 0.7 * planet.grid().voxel_size() / view.pixel_angle();
    clip.w > 0.0 && ndc.x.abs() <= 0.95 && ndc.y.abs() <= 0.95 && (at - view.eye).length() <= near
}

/// What a terrain surface written for a CPU hit gets wrong against the
/// world's appearance: its face (against the CPU's normal), its lit normal
/// (not on the face's side), base colour (outside the material's patch
/// colours), metalness, roughness, the share it lets through, emission.
fn surface_errors(
    s: &SceneSurface,
    appearance: &helio_pass_voxel_planet::engine::TerrainAppearance,
    material: u32,
    normal: DVec3,
) -> Vec<&'static str> {
    let mut out = Vec::new();
    let face = DVec3::from_array(unpack_unit(s.face).map(f64::from));
    let lit = DVec3::from_array(unpack_unit(s.normal).map(f64::from));
    if face.dot(normal) < 0.99 {
        out.push("face");
    }
    if lit.dot(face) <= 0.0 {
        out.push("shading normal");
    }
    let m = &appearance.materials[material as usize];
    let linear = |c: f32| f64::from(c.clamp(0.0, 1.0)).powf(2.2);
    let base = s.base_color();
    for c in 0..3 {
        let (lo, hi) = match m.patches {
            Some(p) => (
                (0..3).map(|k| linear(p[k][c])).fold(f64::MAX, f64::min),
                (0..3).map(|k| linear(p[k][c])).fold(f64::MIN, f64::max),
            ),
            None => (linear(m.colour[c]), linear(m.colour[c])),
        };
        let got = f64::from(base[c]);
        if got < lo - 2.0 / 255.0 || got > hi + 2.0 / 255.0 {
            out.push("base colour");
            break;
        }
    }
    if base[3] != 0.0 {
        out.push("metallic");
    }
    let roughness = f64::from(m.colour[3].clamp(0.04, 1.0));
    if (f64::from(s.roughness()) - roughness).abs() > 1.0 / 255.0 {
        out.push("roughness");
    }
    // The share let through from behind, tinted by its colour (the frame's
    // two-sided surface: `SURFACE_FLAG_TWO_SIDED`).
    let share = m.transmission.clamp(0.0, 1.0);
    let through = s.transmission();
    if (0..3).any(|c| (through[c] - base[c] * share).abs() > 1.0 / 255.0 + base[c] * share / 256.0)
    {
        out.push("transmission");
    }
    let emitted = unpack_radiance(s.emissive);
    for c in 0..3 {
        let want = base[c] * m.emission;
        if (emitted[c] - want).abs() > want / 64.0 + 2.0 / 255.0 * m.emission + 1e-6 {
            out.push("emission");
            break;
        }
    }
    out
}

/// Random unit vectors.
fn units(seed: u64) -> impl FnMut() -> DVec3 {
    let mut next = xorshift(seed);
    move || loop {
        let v =
            DVec3::new(next() as f64, next() as f64, next() as f64) / u64::MAX as f64 * 2.0 - 1.0;
        if v.length_squared() > 1e-4 && v.length_squared() <= 1.0 {
            return v.normalize();
        }
    }
}

/// A brick ball raised beside the eye on the ground; the planet, eye,
/// view, up.
fn ball_ground() -> (Planet, DVec3, View, DVec3, DVec3) {
    let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
    let dir = land(&planet, 4, 0.37, 0.61);
    let ground = planet.surface_point(dir, 0.0);
    let up = planet.grid().up(ground);
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    planet
        .apply(Brush {
            center: (ground + fwd * 2.0 + up * 0.8).to_array(),
            radius: 1.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Add,
            material: material::BRICK,
            height: 0.0,
        })
        .unwrap();
    let eye = ground + up * 3.0 - fwd * 4.0 + side * 1.5;
    let view = View::new(
        eye,
        ((ground + fwd * 2.5) - eye).normalize().as_vec3(),
        [320, 180],
    );
    (planet, eye, view, up, ground)
}

/// The terrain traces ray streams from anywhere on an Earth (6371 km, 10 cm
/// voxels): from GI's finest probes about the eye (open air; in the ground:
/// a hit from inside at once), from hits on the ground and on a glowing,
/// translucent brick ball, from points hundreds of metres off, and from far
/// over the ground (outside the volume, entering it). Every ray whose CPU
/// hit is in view near the eye (level 0 drawn there) hits at the CPU's
/// distance and face, its surface the cell's material as the world's
/// appearance gives it; the far ones at the drawn level's cells. A ray ends
/// at its length, a nearer hit already in its row stays and a farther one
/// is replaced, and rows past the header's count are left untouched. The
/// rays are relative to a frame origin off the eye (as Helio's snapped
/// origin is).
#[test]
fn terrain_traces_stream_rays() {
    let Some(gpu) = gpu() else { return };
    let (planet, eye, view, up, ground) = ball_ground();
    // The ball glows and lets light through: its surfaces carry both.
    let mut appearance = planet.field().appearance();
    appearance.materials[material::BRICK as usize].emission = 3.0;
    appearance.materials[material::BRICK as usize].transmission = 0.4;
    let mut world = ray_world(gpu, &planet);
    world.chunks.set_appearance(VolumeId(0), &appearance);
    world.settle(gpu, &planet, &view);
    // Helio's frame origin is snapped near the eye, not at it.
    let frame = eye + DVec3::new(3.75, -1.5, 2.25);
    let mut unit = units(0x2545_f491_4f6c_dd1d);
    let voxel = planet.grid().voxel_size();

    // 1. From the finest probes' places: open air and rock.
    let mut origins = Vec::new();
    for z in 8..16 {
        for y in 8..16 {
            for x in 8..16 {
                origins.push(probe_place(eye, 0, glam::I64Vec3::new(x, y, z)));
            }
        }
    }
    let rays: Vec<(DVec3, DVec3)> = origins
        .iter()
        .flat_map(|&o| std::iter::repeat_n(o, 64))
        .map(|o| (o, unit()))
        .collect();
    let stream: Vec<SceneRay> = rays
        .iter()
        .map(|&(o, d)| common::scene_ray(frame, o, d, 1e30, 0.0))
        .collect();
    let (hits, surfaces) = common::trace_stream(
        gpu,
        &world.chunks,
        RayClass::Nearest,
        frame,
        &stream,
        None,
        0,
    );
    // The same rays as runs of a probe's 64 (`Stream::run`: each origin
    // placed once) hit and shade the same, bit for bit (a settled world
    // fades no level, so no dither tells a run's start from a ray's).
    let runs = common::trace_stream_in_runs(
        gpu,
        &world.chunks,
        RayClass::Nearest,
        frame,
        &stream,
        None,
        0,
        64,
    );
    let differ = (0..rays.len())
        .filter(|&k| runs.0[k] != hits[k] || runs.1[k] != surfaces[k])
        .count();
    assert_eq!(differ, 0, "{differ} rays differ traced in runs");
    let (mut compared, mut wrong, mut rock, mut rock_wrong, mut surface_wrong) = (0, 0, 0, 0, 0);
    let mut balls = 0;
    // The CPU hits near the eye: ray, origin, direction, hit.
    let mut near_hits = Vec::new();
    for (k, &(o, d)) in rays.iter().enumerate() {
        let got = terrain_hit(&hits[k]);
        if planet.solid(planet.grid().locate(o).0) {
            rock += 1;
            let face = DVec3::from_array(unpack_unit(surfaces[k].face).map(f64::from));
            if !got.is_some_and(|(t, inside)| inside && t < 1e-5) || face.dot(-d) < 0.99 {
                rock_wrong += 1;
            }
            continue;
        }
        let Some(cpu) = planet.raycast(o, d, 60.0) else {
            continue;
        };
        if !near_in_view(&view, &planet, o, d, cpu.distance) {
            continue;
        }
        compared += 1;
        let exact = got.is_some_and(|(t, inside)| {
            !inside && (t - cpu.distance).abs() <= 2e-3 + cpu.distance * 1e-5
        });
        if !exact {
            wrong += 1;
            if wrong < 8 {
                eprintln!(
                    "ray {k}: traced {:?}, cpu {} n {}",
                    hits[k], cpu.distance, cpu.normal
                );
            }
            continue;
        }
        let m = planet.material(cpu.cell);
        balls += usize::from(m == material::BRICK);
        let errors = surface_errors(&surfaces[k], &appearance, m, cpu.normal);
        if !errors.is_empty() {
            surface_wrong += 1;
            if surface_wrong < 8 {
                eprintln!(
                    "ray {k}: surface {:?} of material {m}: {errors:?}",
                    surfaces[k]
                );
            }
        }
        near_hits.push((o, d, cpu));
    }
    eprintln!(
        "probe rays: compared {compared}, wrong {wrong}, surfaces wrong {surface_wrong}, \
         ball {balls}; rock {rock}, wrong {rock_wrong}"
    );
    assert!(
        rock > 50 && rock_wrong == 0,
        "{rock} in rock, {rock_wrong} wrong"
    );
    assert!(
        compared > 1500 && wrong * 100 <= compared,
        "{wrong} of {compared}"
    );
    assert!(balls > 100, "{balls} hits on the ball");
    assert!(
        surface_wrong * 100 <= compared,
        "{surface_wrong} surfaces of {compared}"
    );

    // 2. From those hits (a twentieth of a cell off the face) out over it,
    // with a footprint (the lit normal at it).
    let surface_rays: Vec<(DVec3, DVec3)> = near_hits
        .iter()
        .step_by(2)
        .map(|&(o, d, cpu)| {
            let from = o + d * cpu.distance + cpu.normal * (0.05 * voxel);
            let out = unit();
            (from, if out.dot(cpu.normal) < 0.0 { -out } else { out })
        })
        .collect();
    let stream: Vec<SceneRay> = surface_rays
        .iter()
        .map(|&(o, d)| common::scene_ray(frame, o, d, 1e30, 0.01))
        .collect();
    let (hits, surfaces) = common::trace_stream(
        gpu,
        &world.chunks,
        RayClass::Nearest,
        frame,
        &stream,
        None,
        0,
    );
    let (mut compared, mut wrong, mut inside, mut surface_wrong) = (0, 0, 0, 0);
    for (k, &(o, d)) in surface_rays.iter().enumerate() {
        inside += usize::from(terrain_hit(&hits[k]).is_some_and(|h| h.1));
        let Some(cpu) = planet.raycast(o, d, 60.0) else {
            continue;
        };
        if !near_in_view(&view, &planet, o, d, cpu.distance) {
            continue;
        }
        compared += 1;
        let exact = terrain_hit(&hits[k])
            .is_some_and(|(t, _)| (t - cpu.distance).abs() <= 2e-3 + cpu.distance * 1e-5);
        if !exact {
            wrong += 1;
            continue;
        }
        let errors = surface_errors(
            &surfaces[k],
            &appearance,
            planet.material(cpu.cell),
            cpu.normal,
        );
        surface_wrong += usize::from(!errors.is_empty());
    }
    eprintln!(
        "surface rays: {} rays, compared {compared}, wrong {wrong}, surfaces wrong \
         {surface_wrong}, inside {inside}",
        surface_rays.len()
    );
    // A start over a face can lie on the side of a step beside it
    // (`ON_FACE`): in that cell, a few in a thousand.
    assert!(
        inside * 100 <= surface_rays.len(),
        "{inside} rays from over a face start in rock"
    );
    // From exactly on those faces (no offset: the consumer's to give), out
    // and in: in the cell at once, whichever way they go.
    let on_face: Vec<SceneRay> = near_hits
        .iter()
        .step_by(4)
        .flat_map(|&(o, d, cpu)| {
            let at = o + d * cpu.distance;
            [
                common::scene_ray(frame, at, cpu.normal, 1e30, 0.0),
                common::scene_ray(frame, at, -cpu.normal, 1e30, 0.0),
            ]
        })
        .collect();
    let (hits, _) = common::trace_stream(
        gpu,
        &world.chunks,
        RayClass::Nearest,
        frame,
        &on_face,
        None,
        0,
    );
    let not_inside = hits
        .iter()
        .filter(|h| !terrain_hit(h).is_some_and(|(t, inside)| inside && t < 1e-5))
        .count();
    eprintln!(
        "rays from on faces: {} rays, {not_inside} not inside",
        on_face.len()
    );
    assert!(
        not_inside * 100 <= on_face.len(),
        "{not_inside} of {} on faces",
        on_face.len()
    );
    assert!(
        compared > 200 && wrong * 50 <= compared,
        "{wrong} of {compared}"
    );
    assert!(
        surface_wrong * 50 <= compared,
        "{surface_wrong} surfaces of {compared}"
    );

    // 3. From 100 to 300 m off, 2 to 20 m over the ground there, down at
    // it: at the drawn level's cells (coarser out there).
    let mut far_rays = Vec::new();
    while far_rays.len() < 400 {
        let a = unit();
        let across = (a - up * a.dot(up)).normalize_or_zero();
        if across == DVec3::ZERO {
            continue;
        }
        let r = 100.0 + 200.0 * (unit().x * 0.5 + 0.5);
        let under = planet.surface_point(eye + across * r, 0.0);
        let at = under + planet.grid().up(under) * (2.0 + 18.0 * (unit().y * 0.5 + 0.5));
        let d = unit();
        if d.dot(planet.grid().up(at)) > -0.3 || planet.solid(planet.grid().locate(at).0) {
            continue;
        }
        far_rays.push((at, d));
    }
    let stream: Vec<SceneRay> = far_rays
        .iter()
        .map(|&(o, d)| common::scene_ray(frame, o, d, 1e30, 0.0))
        .collect();
    let (hits, _) = common::trace_stream(
        gpu,
        &world.chunks,
        RayClass::Nearest,
        frame,
        &stream,
        None,
        0,
    );
    let (mut compared, mut wrong) = (0, 0);
    for (k, &(o, d)) in far_rays.iter().enumerate() {
        let Some(cpu) = planet.raycast(o, d, 200.0) else {
            continue;
        };
        compared += 1;
        let level = world
            .chunks
            .chunk_at(hits[k].b)
            .map_or(0, |(key, _)| key.level);
        let cell = voxel * f64::from(1u32 << level);
        let close = terrain_hit(&hits[k]).is_some_and(|(t, inside)| {
            !inside && (t - cpu.distance).abs() <= 3.0 * cell / d.dot(cpu.normal).abs().max(0.25)
        });
        if !close {
            wrong += 1;
            if wrong < 8 {
                // The drawn chunks over its start's column (level, layer).
                let (c, _) = planet.grid().locate(o);
                let drawn: Vec<(u32, i32)> = world
                    .chunks
                    .chunks_over(VolumeId(0), c.face, c.i, c.j)
                    .into_iter()
                    .filter(|n| n.2)
                    .map(|n| (n.0.level, n.0.cz))
                    .collect();
                eprintln!(
                    "far ray {k}: traced {:?} (level {level}), cpu {} at layer {} from layer {}; drawn over it {drawn:?}",
                    hits[k], cpu.distance, cpu.cell.k, c.k
                );
            }
        }
    }
    eprintln!("far rays: compared {compared}, wrong {wrong}");
    assert!(
        compared > 300 && wrong * 20 <= compared,
        "{wrong} of {compared}"
    );

    // 4. From 25 km over the ground about the eye (outside the volume):
    // down onto the ground (drawn at level 0 there), up into nothing.
    let mut high = Vec::new();
    for _ in 0..64 {
        let a = unit();
        let p = planet.surface_point(ground + (a - up * a.dot(up)) * 3.0, 0.0);
        high.push((p, planet.surface_point(p, 25_000.0)));
    }
    let stream: Vec<SceneRay> = high
        .iter()
        .flat_map(|&(p, o)| {
            let down = (p - o).normalize();
            [
                common::scene_ray(frame, o, down, 1e30, 0.0),
                common::scene_ray(frame, o, -down, 1e30, 0.0),
            ]
        })
        .collect();
    let (hits, _) = common::trace_stream(
        gpu,
        &world.chunks,
        RayClass::Nearest,
        frame,
        &stream,
        None,
        0,
    );
    let mut high_wrong = 0;
    for (k, &(p, o)) in high.iter().enumerate() {
        let down = terrain_hit(&hits[2 * k]);
        let expected = (p - o).length();
        if !down.is_some_and(|(t, inside)| !inside && (t - expected).abs() <= 0.02)
            || hits[2 * k + 1].t != 0.0
        {
            high_wrong += 1;
            eprintln!(
                "high ray {k}: traced {:?} / {:?}, ground at {expected}",
                hits[2 * k],
                hits[2 * k + 1]
            );
        }
    }
    assert_eq!(high_wrong, 0, "rays from outside the volume");

    // 5. Lengths, rows already holding hits and rows past the count, on
    // probe rays that hit near the eye.
    let picked: Vec<_> = near_hits.iter().take(512).collect();
    let mut stream = Vec::new();
    let mut before = Vec::new();
    for (n, &&(o, d, cpu)) in picked.iter().enumerate() {
        let t = cpu.distance as f32;
        // Short of the hit, past it; a nearer hit there, a farther one.
        let (t_max, row) = match n % 4 {
            0 => (0.5 * t, SceneHit::default()),
            1 => (1.5 * t, SceneHit::default()),
            2 => (
                1e30,
                SceneHit {
                    t: 0.5 * t,
                    a: 15 << 28,
                    b: 7,
                    c: 9,
                },
            ),
            _ => (
                1e30,
                SceneHit {
                    t: 2.0 * t,
                    a: 15 << 28,
                    b: 7,
                    c: 9,
                },
            ),
        };
        stream.push(common::scene_ray(frame, o, d, t_max, 0.0));
        before.push(row);
    }
    // Rows past the header's count: untouched.
    let past = 64;
    for _ in 0..past {
        stream.push(common::scene_ray(frame, eye, -up, 1e30, 0.0));
        before.push(SceneHit {
            t: 7.0,
            a: 0x1234_5678,
            b: 1,
            c: 2,
        });
    }
    let (hits, surfaces) = common::trace_stream(
        gpu,
        &world.chunks,
        RayClass::Nearest,
        frame,
        &stream,
        Some(&before),
        past,
    );
    let mut kept_wrong = 0;
    for (n, &&(_, _, cpu)) in picked.iter().enumerate() {
        let h = &hits[n];
        let ok = match n % 4 {
            0 => h.t == 0.0,
            1 | 3 => terrain_hit(h)
                .is_some_and(|(t, _)| (t - cpu.distance).abs() <= 2e-3 + cpu.distance * 1e-5),
            _ => *h == before[n],
        };
        kept_wrong += usize::from(!ok);
    }
    let n = picked.len();
    let past_touched = (n..n + past)
        .filter(|&k| hits[k] != before[k] || surfaces[k] != SceneSurface::default())
        .count();
    eprintln!(
        "lengths and rows: {kept_wrong} of {n} wrong, {past_touched} rows past the count touched"
    );
    assert!(n > 400 && kept_wrong * 100 <= n, "{kept_wrong} of {n}");
    assert_eq!(past_touched, 0);
}

/// Occlusion streams agree with CPU casts: from hits near the eye (a
/// twentieth of a cell off their face) toward the sun (to 60 m) and toward
/// points of the air within 8 m, a ray is blocked exactly where the CPU's
/// meets solid before its end; rows already blocked stay so.
#[test]
fn terrain_traces_occlusion_streams() {
    let Some(gpu) = gpu() else { return };
    let (planet, eye, view, up, _) = ball_ground();
    let mut world = ray_world(gpu, &planet);
    world.settle(gpu, &planet, &view);
    let frame = eye - DVec3::new(1.25, 0.5, -2.0);
    let side = up.any_orthonormal_vector();
    // A low sun across the ball: its shadow on the ground in view.
    let sun = (up * 0.45 + side * 0.9).normalize();
    let voxel = planet.grid().voxel_size();
    let mut unit = units(0x9e37_79b9_7f4a_7c15);
    // Starts: the ground's and the ball's faces in view near the eye.
    let view_proj = view.view_proj(&planet);
    let mut starts = Vec::new();
    for y in (0..180).step_by(3) {
        for x in (0..320).step_by(3) {
            let d = view.pixel_dir(view_proj, x, y);
            if let Some(cpu) = planet.raycast(eye, d, 15.0) {
                starts.push(eye + d * cpu.distance + cpu.normal * (0.05 * voxel));
            }
        }
    }
    let mut rays = Vec::new();
    for &s in &starts {
        rays.push((s, sun, 60.0));
        // A point of the air within 8 m.
        let target = s + unit() * (1.0 + 7.0 * (unit().x * 0.5 + 0.5));
        if !planet.solid(planet.grid().locate(target).0) {
            let to = target - s;
            rays.push((s, to.normalize(), to.length()));
        }
    }
    let stream: Vec<SceneRay> = rays
        .iter()
        .map(|&(o, d, t)| common::scene_ray(frame, o, d, t as f32, 0.0))
        .collect();
    // Every eighth row arrives blocked (by another tracer).
    let blocked_row = SceneHit {
        t: 1.0,
        a: 0,
        b: 1023 | (1023 << 10) | (1023 << 20),
        c: 0,
    };
    let before: Vec<SceneHit> = (0..rays.len())
        .map(|k| {
            if k % 8 == 7 {
                blocked_row
            } else {
                SceneHit::default()
            }
        })
        .collect();
    let (hits, _) = common::trace_stream(
        gpu,
        &world.chunks,
        RayClass::Occlusion,
        frame,
        &stream,
        Some(&before),
        0,
    );
    let (mut compared, mut shadowed, mut wrong, mut lost) = (0, 0, 0, 0);
    for (k, &(o, d, t_max)) in rays.iter().enumerate() {
        if k % 8 == 7 {
            lost += usize::from(hits[k] != blocked_row);
            continue;
        }
        let cpu = planet.raycast(o, d, t_max);
        // Past level 0's reach the cut is coarser than the CPU's cells.
        if cpu.is_some_and(|c| !near_in_view(&view, &planet, o, d, c.distance)) {
            continue;
        }
        compared += 1;
        shadowed += usize::from(cpu.is_some());
        let traced_blocked = hits[k].t > 0.0 && hits[k].through() == [0.0; 3];
        let traced_free = hits[k] == SceneHit::default();
        if (cpu.is_some() && !traced_blocked) || (cpu.is_none() && !traced_free) {
            wrong += 1;
            if wrong < 8 {
                eprintln!(
                    "ray {k}: traced {:?}, cpu {:?}",
                    hits[k],
                    cpu.map(|c| c.distance)
                );
            }
        }
    }
    eprintln!(
        "occlusion rays: compared {compared}, shadowed {shadowed}, wrong {wrong}, blocked rows lost {lost}"
    );
    assert!(
        compared > 1500 && shadowed > 200,
        "{compared} compared, {shadowed} shadowed"
    );
    assert!(wrong * 50 <= compared, "{wrong} of {compared}");
    assert_eq!(lost, 0);
}

/// A volume authored in coordinate space 3 answers the rays of every space
/// and of its own, row for row the same, and no ray of another space (the
/// world's, or one with nothing in it), nearest-hit or occlusion: the same
/// place in another space holds nothing of it.
#[test]
fn a_volume_answers_only_rays_of_its_space() {
    use helio_frame_api::rays::stream::{space_of_slot, SPACE_ANY};
    let Some(gpu) = gpu() else { return };
    let (planet, eye, view, _, _) = ball_ground();
    let mut world = ray_world(gpu, &planet);
    let volume = world.chunks.primary().expect("one volume");
    world.chunks.set_space(volume, 3);
    world.settle(gpu, &planet, &view);
    let frame = eye - DVec3::new(1.25, 0.5, -2.0);
    let view_proj = view.view_proj(&planet);
    let mut base = Vec::new();
    for y in (0..180).step_by(6) {
        for x in (0..320).step_by(6) {
            let d = view.pixel_dir(view_proj, x, y);
            if planet.raycast(eye, d, 15.0).is_some() {
                base.push(common::scene_ray(frame, eye, d, 15.0, 0.0));
            }
        }
    }
    assert!(base.len() > 200, "{} rays meet the ground", base.len());
    let in_space =
        |space: u32| -> Vec<SceneRay> { base.iter().map(|r| r.in_space(space)).collect() };
    for class in [RayClass::Nearest, RayClass::Occlusion] {
        let trace = |rays: &[SceneRay]| {
            common::trace_stream(gpu, &world.chunks, class, frame, rays, None, 0).0
        };
        let any = trace(&in_space(SPACE_ANY));
        let own = trace(&in_space(space_of_slot(3)));
        let met = any.iter().filter(|h| h.t > 0.0).count();
        assert!(
            met * 10 >= base.len() * 9,
            "{class:?}: {met} of {}",
            base.len()
        );
        assert_eq!(
            any, own,
            "{class:?}: its own space sees what every space does"
        );
        for other in [space_of_slot(0), space_of_slot(5)] {
            let hits = trace(&in_space(other));
            assert!(
                hits.iter().all(|h| *h == SceneHit::default()),
                "{class:?}: space {other} sees the volume of space 3"
            );
        }
    }
}

/// A plane (one affine frame: rays are placed by its anchor) traces ray
/// streams from about the eye as the CPU casts them.
#[test]
fn a_plane_traces_stream_rays() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe {
        shape: Shape::Plane,
        plane_size_m: 3_000.0,
        ..Default::default()
    })
    .unwrap();
    let eye = planet.surface_point(DVec3::new(12.3, 0.0, -45.6), 1.7);
    let view = View::new(eye, Vec3::new(0.8, -0.35, 0.6), [320, 180]);
    let mut world = ray_world(gpu, &planet);
    world.settle(gpu, &planet, &view);
    let frame = eye + DVec3::new(-2.5, 0.75, 1.0);
    let mut unit = units(0x51ed_270b_2c1b_3c6d);
    let mut rays = Vec::new();
    while rays.len() < 20_000 {
        let o = eye + unit() * 4.0;
        if !planet.solid(planet.grid().locate(o).0) {
            rays.push((o, unit()));
        }
    }
    let stream: Vec<SceneRay> = rays
        .iter()
        .map(|&(o, d)| common::scene_ray(frame, o, d, 1e30, 0.0))
        .collect();
    let (hits, _) = common::trace_stream(
        gpu,
        &world.chunks,
        RayClass::Nearest,
        frame,
        &stream,
        None,
        0,
    );
    let (mut compared, mut wrong) = (0, 0);
    for (k, &(o, d)) in rays.iter().enumerate() {
        let Some(cpu) = planet.raycast(o, d, 60.0) else {
            continue;
        };
        if !near_in_view(&view, &planet, o, d, cpu.distance) {
            continue;
        }
        compared += 1;
        if !terrain_hit(&hits[k]).is_some_and(|(t, inside)| {
            !inside && (t - cpu.distance).abs() <= 2e-3 + cpu.distance * 1e-5
        }) {
            wrong += 1;
        }
    }
    eprintln!("plane rays: compared {compared}, wrong {wrong}");
    assert!(
        compared > 1000 && wrong * 100 <= compared,
        "{wrong} of {compared}"
    );
}

/// A `Full` stream's hit is the G-buffer's texel: every second pixel of a
/// view of the ground and a glowing, translucent brick ball, its hit within
/// 6 m (cells three pixels wide and more: the materials' boundaries are not
/// jittered there), traced as a ray from the eye through the pixel's centre
/// with the pixel's footprint (its width at the origin 0, its spread the
/// step to the next pixel's ray), decodes to the surface the publish wrote
/// there, field by field within the two encodings' precision: base colour,
/// metalness, roughness, reflectance, emission, flags, what it lets
/// through (on the ball's and a block of leaves' two-sided cells, their
/// transmission share of their colour, in the texel as in the hit, not
/// their whole colour), the shading normal. A block of leaves stands
/// beside the ball. The texel's occlusion is the hit's times
/// the screen's sky samples (a hit has none): at most the hit's.
#[test]
fn full_stream_surfaces_are_the_gbuffer_texels() {
    use helio_frame_api::rays::stream::{unpack_radiance, unpack_unit};
    let Some(gpu) = gpu() else { return };
    let (mut planet, eye, view, up, ground) = ball_ground();
    // And a block of leaves beside the ball (thin two-sided foliage).
    let side = up.any_orthonormal_vector();
    planet
        .apply(Brush {
            center: (ground + up.cross(side) * 1.5 - side * 1.6 + up * 0.6).to_array(),
            radius: 0.6,
            shape: BrushShape::Cube,
            op: BrushOp::Add,
            material: material::LEAVES,
            height: 0.0,
        })
        .unwrap();
    let mut appearance = planet.field().appearance();
    appearance.materials[material::BRICK as usize].emission = 3.0;
    appearance.materials[material::BRICK as usize].transmission = 0.4;
    let mut world = ray_world(gpu, &planet);
    world.chunks.set_appearance(VolumeId(0), &appearance);
    let hits = world.settle(gpu, &planet, &view);
    let [w, h] = view.size;
    let texture = |format: wgpu::TextureFormat| {
        gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gbuffer"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let gbuffer: Vec<wgpu::Texture> = helio_frame_api::gbuffer::TARGET_FORMATS
        .iter()
        .map(|&f| texture(f))
        .collect();
    let views: Vec<wgpu::TextureView> = gbuffer
        .iter()
        .map(|t| t.create_view(&Default::default()))
        .collect();
    let depth = texture(wgpu::TextureFormat::Depth32Float);
    let depth_view = depth.create_view(&Default::default());
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let targets: [&wgpu::TextureView; 8] = std::array::from_fn(|i| &views[i]);
    world.chunks.publish_scene(
        &mut helio_core::CommandRecorder::from_encoder(&mut encoder),
        targets,
        &depth_view,
        None,
    );
    gpu.queue.submit([encoder.finish()]);
    let albedo = read_rgba8(gpu, &gbuffer[0]);
    let orm = read_rgba8(gpu, &gbuffer[2]);
    let halves =
        |t: usize| -> [Vec<f32>; 4] { std::array::from_fn(|c| read_rgba16(gpu, &gbuffer[t], c)) };
    let [normal, emissive, sss, extra] = [halves(1), halves(3), halves(5), halves(6)];
    let view_proj = view.view_proj(&planet);
    let mut pixels = Vec::new();
    let mut rays = Vec::new();
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            let at = (y * w + x) as usize;
            let Some(hit) = hits[at] else { continue };
            if hit.distance > 6.0 || hit.level != 0 {
                continue;
            }
            let d = view.pixel_dir(view_proj, x, y);
            let spread = (view.pixel_dir(view_proj, x + 1, y) - d).length();
            pixels.push((at, hit.distance, hit.material));
            rays.push(common::scene_ray(eye, eye, d, 1e30, spread as f32));
        }
    }
    let (traced, surfaces) = common::trace_stream_full(gpu, &world.chunks, eye, &rays);
    let unorm = |b: u8| f32::from(b) / 255.0;
    let (mut compared, mut balls, mut wrong) = (0usize, 0usize, Vec::<&str>::new());
    let mut worst_normal = 1.0f32;
    let mut leaves = 0usize;
    for (k, &(at, distance, m)) in pixels.iter().enumerate() {
        // The same hit (a pixel along an edge may meet the next cell).
        if !terrain_hit(&traced[k]).is_some_and(|(t, _)| (t - distance).abs() < 1e-3) {
            continue;
        }
        compared += 1;
        let s = &surfaces[k];
        let base = s.base_color();
        let mut bad = Vec::new();
        let near = |a: f32, b: f32, tol: f32| (a - b).abs() <= tol;
        if (0..3).any(|c| !near(base[c], unorm(albedo[at][c]), 1.5 / 255.0)) {
            bad.push("base colour");
        }
        if !near(base[3], unorm(orm[at][2]), 1.5 / 255.0) {
            bad.push("metallic");
        }
        if !near(s.roughness(), unorm(orm[at][1]), 1.5 / 255.0) {
            bad.push("roughness");
        }
        let f0 = s.specular_f0();
        let texel_f0 = [normal[3][at], unorm(orm[at][3]), emissive[3][at]];
        if (0..3).any(|c| !near(f0[c], texel_f0[c], 1.5 / 255.0)) {
            bad.push("reflectance");
        }
        let glow = unpack_radiance(s.emissive);
        if (0..3).any(|c| !near(glow[c], emissive[c][at], emissive[c][at] / 128.0 + 1e-3)) {
            bad.push("emission");
        }
        if s.flags() != extra[3][at].round() as u32 {
            bad.push("flags");
        }
        let through = unpack_radiance(s.subsurface);
        if (0..3).any(|c| !near(through[c], sss[c][at], sss[c][at] / 128.0 + 2e-3)) {
            bad.push("let through");
        }
        // A two-sided cell lets through its share of its colour, the
        // texel's as the hit's (not its whole colour).
        let share = appearance.materials[m as usize]
            .transmission
            .clamp(0.0, 1.0);
        if share > 0.0
            && (s.flags() == 0
                || (0..3).any(|c| !near(sss[c][at], share * unorm(albedo[at][c]), 2.0 / 255.0)))
        {
            bad.push("share let through");
        }
        leaves += usize::from(m == material::LEAVES);
        let n = unpack_unit(s.normal);
        let texel_n = glam::Vec3::new(normal[0][at], normal[1][at], normal[2][at]).normalize();
        let cos = glam::Vec3::from_array(n).dot(texel_n);
        worst_normal = worst_normal.min(cos);
        if cos < 0.999 {
            bad.push("normal");
        }
        if s.ao() < unorm(orm[at][0]) - 1.5 / 255.0 {
            bad.push("occlusion");
        }
        balls += usize::from(s.flags() != 0);
        if !bad.is_empty() && wrong.len() < 8 {
            eprintln!("pixel {at}: {bad:?}; hit {s:?}");
        }
        if !bad.is_empty() {
            wrong.push(bad[0]);
        }
    }
    eprintln!(
        "full surfaces: {compared} compared ({balls} on the ball), {} differ {:?}, worst normal cosine {worst_normal}",
        wrong.len(),
        wrong.iter().take(16).collect::<Vec<_>>()
    );
    eprintln!("{leaves} on the leaves");
    assert!(
        compared > 2000 && balls > 100 && leaves > 50,
        "{compared} compared, {balls} two-sided, {leaves} leaves"
    );
    assert!(
        wrong.len() * 200 <= compared,
        "{} of {compared} differ",
        wrong.len()
    );
}

/// One timed pass of `stream_rays_cost`: a stream's trace or its surfaces.
struct TimedPass {
    name: &'static str,
    stream: usize,
    surface: bool,
}

/// The passes of `timed` over the streams of `traced` through `world`'s
/// last cut, pass `n` timed into `queries` at `first + 2 n`
/// (`stream_rays_cost`). Each stream's hits are cleared before its first
/// pass.
fn record_streams(
    gpu: &Gpu,
    world: &ChunkWorld,
    recorder: &mut helio_core::CommandRecorder<'_>,
    traced: &[(
        helio_frame_api::rays::stream::StreamBuffers,
        helio_pass_voxel_planet::chunks::rays::StreamTracer,
    )],
    timed: &[TimedPass],
    queries: &wgpu::QuerySet,
    first: u32,
) {
    let cut = world.ray_cut().expect("a cut");
    let mut cleared = vec![false; traced.len()];
    for (n, pass) in timed.iter().enumerate() {
        let (buffers, tracer) = &traced[pass.stream];
        if !std::mem::replace(&mut cleared[pass.stream], true) {
            buffers.clear(recorder);
        }
        let bindings = helio_pass_voxel_planet::chunks::rays::StreamBindings {
            header: &buffers.header,
            rays: &buffers.rays,
            hits: &buffers.hits,
            surfaces: buffers.surfaces.as_ref(),
        };
        let at = wgpu::ComputePassTimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: Some(first + 2 * n as u32),
            end_of_pass_write_index: Some(first + 2 * n as u32 + 1),
        };
        if pass.surface {
            tracer.surface(&gpu.device, recorder, &cut, bindings, Some(at));
        } else {
            tracer.trace(&gpu.device, recorder, &cut, bindings, Some(at));
        }
    }
}

/// What the terrain's ray stream kernels cost on GI's rays: a frame of
/// probe updates (680 probes on each of the two finest levels, 340 on each
/// of the four coarser: 2720 probes, 64 rays each, 174 080 rays, each
/// started behind its probe as GI starts them, not in runs) about a ball on
/// the ground, and the shadow rays toward the sun from their hits (a
/// millimetre off the face, as GI's): the nearest walk with its entry, the
/// entry alone (the same rays a ten-thousandth of a metre long), the
/// `Diffuse` and the `Full` surfaces, and the sun's rays walked whole (no width) and answered at
/// their footprint (as wide as the probe ray's footprint at the hit:
/// walked two texels, the pages past that). Frames are pipelined (two in
/// flight, none waited for alone: a synchronised harness downclocks), the
/// streams recorded before the draw on odd frames and after it on even
/// ones; GPU timestamps, medians in milliseconds. A report: ignored.
#[test]
#[ignore]
fn stream_rays_cost() {
    use helio_frame_api::rays::stream::{
        unpack_unit, Stream, StreamBuffers, SurfaceDetail, TracerId, MASK_ALL,
    };
    use helio_pass_voxel_planet::chunks::rays::StreamTracer;
    let Some(gpu) = gpu() else { return };
    if !gpu
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        eprintln!("no timestamps on this device");
        return;
    }
    let (planet, eye, view, _, _) = ball_ground();
    let mut next = xorshift(0x6a09_e667_f3bc_c908);
    let mut pick = |n: u64| (next() % n) as i64;
    let mut probes = Vec::new();
    for level in 0..6u32 {
        let count = if level < 2 { 680 } else { 340 };
        let mut taken = std::collections::BTreeSet::new();
        while taken.len() < count {
            taken.insert(glam::I64Vec3::new(pick(24), pick(24), pick(24)).to_array());
        }
        probes.extend(
            taken
                .into_iter()
                .map(|at| probe_place(eye, level, glam::I64Vec3::from_array(at))),
        );
    }
    let mut unit = units(0xbb67_ae85_84ca_a73b);
    let dirs: Vec<DVec3> = (0..probes.len() * 64).map(|_| unit()).collect();
    let mut world = ray_world(gpu, &planet);
    world.sun_occlusion = true;
    world.settle(gpu, &planet, &view);
    world.settle_sun_pages(gpu, &planet, &view);
    world.chunks.take_timings();
    // GI's offset behind the probe (`gi_ray_offset` near the origin).
    let behind = 0.002;
    let probe_rays: Vec<SceneRay> = (0..dirs.len())
        .map(|k| common::scene_ray(eye, probes[k / 64] - dirs[k] * behind, dirs[k], 1e30, 0.0))
        .collect();
    // The shadow rays: from the probe rays' hits toward the sun.
    let (hits, surfaces) = common::trace_stream(
        gpu,
        &world.chunks,
        RayClass::Nearest,
        eye,
        &probe_rays,
        None,
        0,
    );
    let sun = world.sun.as_dvec3();
    let mut shadow = Vec::new();
    let mut wide = Vec::new();
    for (k, ray) in probe_rays.iter().enumerate() {
        let Some((t, false)) = terrain_hit(&hits[k]) else {
            continue;
        };
        let face = DVec3::from_array(unpack_unit(surfaces[k].face).map(f64::from));
        if face.dot(sun) <= 0.0 {
            continue;
        }
        let at = eye + DVec3::from_array(ray.origin.map(f64::from)) + dirs[k] * t;
        let from = at + face * behind;
        shadow.push(common::scene_ray(eye, from, sun, 1e30, 0.0));
        let mut w = common::scene_ray(eye, from, sun, 1e30, 0.0);
        w.cone = SceneRay::cone((0.443 * t) as f32, 0.0, MASK_ALL);
        wide.push(w);
    }
    let entry: Vec<SceneRay> = probe_rays
        .iter()
        .map(|r| SceneRay { t_max: 1e-4, ..*r })
        .collect();
    // (rays, class, detail).
    let streams: Vec<(&[SceneRay], RayClass, SurfaceDetail)> = vec![
        (&probe_rays, RayClass::Nearest, SurfaceDetail::Diffuse),
        (&probe_rays, RayClass::Nearest, SurfaceDetail::Full),
        (&entry, RayClass::Nearest, SurfaceDetail::Diffuse),
        (&shadow, RayClass::Occlusion, SurfaceDetail::Full),
        (&wide, RayClass::Occlusion, SurfaceDetail::Full),
    ];
    let timed = [
        TimedPass {
            name: "nearest (walk and entry)",
            stream: 0,
            surface: false,
        },
        TimedPass {
            name: "diffuse surfaces pass",
            stream: 0,
            surface: true,
        },
        TimedPass {
            name: "nearest (a full stream's)",
            stream: 1,
            surface: false,
        },
        TimedPass {
            name: "full surfaces pass",
            stream: 1,
            surface: true,
        },
        TimedPass {
            name: "entry",
            stream: 2,
            surface: false,
        },
        TimedPass {
            name: "sun rays walked",
            stream: 3,
            surface: false,
        },
        TimedPass {
            name: "sun rays at their footprint",
            stream: 4,
            surface: false,
        },
    ];
    let traced: Vec<(StreamBuffers, StreamTracer)> = streams
        .iter()
        .enumerate()
        .map(|(n, &(rays, class, detail))| {
            let name: &'static str = Box::leak(format!("cost{n}").into_boxed_str());
            let stream = Stream::new(name, class, rays.len() as u32);
            let buffers = StreamBuffers::new(&gpu.device, &stream);
            gpu.queue
                .write_buffer(&buffers.rays, 0, bytemuck::cast_slice(rays));
            buffers.set_count(&gpu.queue, rays.len() as u32);
            let tracer = StreamTracer::new(&gpu.device, class, TracerId(common::VOXEL_TRACER))
                .with_detail(detail);
            tracer.write_frame(eye, |buffer, bytes| {
                gpu.queue.write_buffer(buffer, 0, bytes)
            });
            (buffers, tracer)
        })
        .collect();
    const FRAMES: usize = 160;
    let stamps = 2 * timed.len() as u32;
    let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("stream ray timing"),
        ty: wgpu::QueryType::Timestamp,
        count: stamps * FRAMES as u32,
    });
    let stride = (u64::from(stamps) * 8).next_multiple_of(256);
    let resolved = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: FRAMES as u64 * stride,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let config = view.config();
    let tan_y = f64::from((FOV_Y * 0.5).tan());
    let tan_x = tan_y * f64::from(view.size[0]) / f64::from(view.size[1]);
    let cone = residency::View {
        forward: view.forward.as_dvec3(),
        cos_half: (tan_x * tan_x + tan_y * tan_y).sqrt().atan().cos(),
    };
    let mut in_flight = std::collections::VecDeque::new();
    for f in 0..FRAMES {
        let view_proj = view.view_proj(&planet);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut recorder = helio_core::CommandRecorder::from_encoder(&mut encoder);
        let before = f % 2 == 1;
        let first = f as u32 * stamps;
        if before {
            record_streams(
                gpu,
                &world.chunks,
                &mut recorder,
                &traced,
                &timed,
                &queries,
                first,
            );
        }
        world.chunks.update(
            &mut recorder,
            view.eye,
            DVec3::ZERO,
            Some(cone),
            world.frame,
            &config,
        );
        world
            .chunks
            .draw_sun_occlusion(&mut recorder, world.frame, view.eye, world.sun);
        world.chunks.draw(
            &mut recorder,
            world.frame,
            view.eye,
            view_proj,
            view_proj,
            [glam::Vec2::ZERO; 2],
            view.near,
            world.sun,
            view.size,
            None,
            None,
        );
        if !before {
            record_streams(
                gpu,
                &world.chunks,
                &mut recorder,
                &traced,
                &timed,
                &queries,
                first,
            );
        }
        drop(recorder);
        encoder.resolve_query_set(
            &queries,
            first..first + stamps,
            &resolved,
            f as u64 * stride,
        );
        in_flight.push_back(gpu.queue.submit([encoder.finish()]));
        world.chunks.submitted();
        world.frame += 1;
        // Two frames in flight.
        if in_flight.len() > 2 {
            let index = in_flight.pop_front().unwrap();
            gpu.device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(index),
                    timeout: None,
                })
                .unwrap();
        }
    }
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let words: Vec<u64> =
        bytemuck::cast_slice(&common::read_buffer(gpu, &resolved, resolved.size())).to_vec();
    let period = f64::from(gpu.queue.get_timestamp_period()) * 1e-6;
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let per_frame = (stride / 8) as usize;
    // The first frames warm up.
    let stage = |n: usize, before: bool| {
        median(
            (16..FRAMES)
                .filter(|f| (f % 2 == 1) == before)
                .map(|f| {
                    let at = f * per_frame + 2 * n;
                    (words[at + 1] - words[at]) as f64 * period
                })
                .collect(),
        )
    };
    eprintln!(
        "{} probe rays a frame, {} sun rays from their hits",
        probe_rays.len(),
        shadow.len()
    );
    for (n, pass) in timed.iter().enumerate() {
        eprintln!(
            "stream {}: {:.3} ms before the draw, {:.3} ms after it",
            pass.name,
            stage(n, true),
            stage(n, false)
        );
    }
}

/// Ray streams at an Earth's scale from orbit: from the eye 300 km up (out
/// of the volume, entering it), every second pixel's ray hits where the
/// camera's own trace drew it (the same walk, from the ceiling ball in);
/// from a kilometre (or four of the drawn level's cells) over those hits,
/// straight down, at the drawn level's cells (kilometres: the rays start
/// hundreds of kilometres from the eye).
#[test]
fn stream_rays_from_orbit() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let grid = *planet.grid();
    let dir = grid.direction(
        2,
        f64::from(grid.cells()) * 0.3,
        f64::from(grid.cells()) * 0.7,
    );
    let eye = dir * (grid.radius() + 300_000.0);
    let view = View::new(
        eye,
        (-dir + dir.any_orthonormal_vector() * 0.5)
            .normalize()
            .as_vec3(),
        [256, 144],
    );
    let mut world = ray_world(gpu, &planet);
    let drawn = world.settle(gpu, &planet, &view);
    let view_proj = view.view_proj(&planet);
    let frame = eye + DVec3::new(40.0, -12.5, 7.25);
    let mut pixels = Vec::new();
    for y in (0..view.size[1]).step_by(2) {
        for x in (0..view.size[0]).step_by(2) {
            if let Some(h) = drawn[(y * view.size[0] + x) as usize] {
                pixels.push((view.pixel_dir(view_proj, x, y), h));
            }
        }
    }
    // Over the hit by a kilometre, or four of its level's cells (a coarse
    // cell's side reaches a cell over the face it was hit on).
    let lift = |level: u32| (4.0 * grid.voxel_size() * f64::from(1u32 << level)).max(1000.0);
    let mut stream = Vec::new();
    for &(d, h) in &pixels {
        stream.push(common::scene_ray(frame, eye, d, 1e30, 0.0));
        let at = eye + d * h.distance;
        let up = grid.up(at);
        stream.push(common::scene_ray(
            frame,
            at + up * lift(h.level),
            -up,
            1e30,
            0.0,
        ));
    }
    let (hits, _) = common::trace_stream(
        gpu,
        &world.chunks,
        RayClass::Nearest,
        frame,
        &stream,
        None,
        0,
    );
    let (mut from_eye_wrong, mut down_wrong) = (0, 0);
    for (k, &(_, h)) in pixels.iter().enumerate() {
        let cell = grid.voxel_size() * f64::from(1u32 << h.level);
        let eye_hit = terrain_hit(&hits[2 * k]);
        if !eye_hit.is_some_and(|(t, inside)| {
            !inside && (t - h.distance).abs() <= 1e-3 * cell + 1e-6 * h.distance
        }) {
            from_eye_wrong += 1;
            if from_eye_wrong < 8 {
                eprintln!(
                    "orbit ray {k}: traced {:?}, drawn at {} (level {})",
                    hits[2 * k],
                    h.distance,
                    h.level
                );
            }
        }
        let down = terrain_hit(&hits[2 * k + 1]);
        if !down.is_some_and(|(t, inside)| !inside && (t - lift(h.level)).abs() <= 3.0 * cell) {
            down_wrong += 1;
            if down_wrong < 8 {
                eprintln!(
                    "down ray {k}: traced {:?}, level {} cells of {cell} m",
                    hits[2 * k + 1],
                    h.level
                );
            }
        }
    }
    eprintln!(
        "orbit: {} pixels, from the eye {from_eye_wrong} wrong, from a kilometre up {down_wrong} wrong",
        pixels.len()
    );
    assert!(pixels.len() > 2000, "{} pixels", pixels.len());
    assert!(
        from_eye_wrong * 200 <= pixels.len(),
        "{from_eye_wrong} from the eye"
    );
    assert!(
        down_wrong * 100 <= pixels.len(),
        "{down_wrong} from a kilometre up"
    );
}

/// Level `level`'s probe `at` (0..24 an axis) of GI's probe clipmap about
/// `eye` as GI lays it out (24 a side, each level twice as far apart as the
/// one inside it, `PROBE_SPACING` the finest): a set of ray origins in open
/// air and in rock, as GI's probe rays have.
fn probe_place(eye: DVec3, level: u32, at: glam::I64Vec3) -> DVec3 {
    let spacing = PROBE_SPACING * f64::from(1u32 << level);
    let first = (eye / spacing).floor().as_i64vec3() - 11;
    (first + at).as_dvec3() * spacing
}

/// Occlusion rays leaving the ground as wide as a footprint (the ray-stream
/// contract: a consumer's ray leaves a hit as wide as the footprint that
/// found it, offset off its face by the consumer) answer as the exact walk
/// of a zero-width ray: in a forest with the sun 50 degrees up, over ground
/// points under open sky (the zero-width walk lit at the point and at eight
/// points half a metre about it across the sun) rays of width 0, 1 cm,
/// 10 cm and 1 m reach the sun, and under cover (all nine walks shaded: a
/// solid roof's shadow, the canopies' solid parts) they are stopped.
/// Walked rays (narrower than the sun pages' finest texel) answer exactly;
/// wider ones the pages answer past a walked band (`paged_occlusion`), with
/// at most one in fifty off (a page texel is a point sample): 0 at 10 cm, 1
/// of 164 open and 3 of 159 covered at 1 m (read at the width's own texels,
/// before half-width reads, 10 cm rays had 1 and 1 off).
#[test]
fn wide_occlusion_rays_leave_the_ground() {
    use helio_frame_api::rays::stream::{unpack_unit, RayClass, MASK_ALL};
    let Some(gpu) = gpu() else { return };
    let (mut planet, dir) = forest_planet();
    let eye = planet.surface_point(dir, 1.7);
    let up = eye.normalize();
    let side = up.any_orthonormal_vector();
    let fwd = up.cross(side);
    // A solid roof, 6 m across and 5 m up, toward the sun: a real occluder
    // whose shadow's core is wider than the widest ray (the forest's
    // canopies have gaps a metre-wide ray may rightly see light through).
    planet
        .apply(Brush {
            center: (planet.surface_point(dir, 5.0) + side * 6.0).to_array(),
            radius: 3.0,
            shape: BrushShape::Cube,
            op: BrushOp::Add,
            material: material::BRICK,
            height: 0.0,
        })
        .unwrap();
    let view = View::new(eye, (fwd - up * 0.25).normalize().as_vec3(), [320, 180]);
    let mut world = ray_world(gpu, &planet);
    world.sun_occlusion = true;
    let elevation = 50f64.to_radians();
    world.sun = (up * elevation.sin() + side * elevation.cos())
        .normalize()
        .as_vec3();
    world.settle(gpu, &planet, &view);
    world.settle_sun_pages(gpu, &planet, &view);
    let pages = world.chunks.read_sun_occlusion().expect("drawn");
    let ep = &pages.frame.epochs[0];
    let sun = -DVec3::new(
        f64::from(ep.forward[0]),
        f64::from(ep.forward[1]),
        f64::from(ep.forward[2]),
    );
    // The ground under a half-metre grid 20 m about the eye: rays down.
    let mut starts = Vec::new();
    for a in -40..=40 {
        for b in -40..=40 {
            starts.push(eye + side * (f64::from(a) * 0.5) + fwd * (f64::from(b) * 0.5) + up * 1.0);
        }
    }
    let down: Vec<SceneRay> = starts
        .iter()
        .map(|&o| common::scene_ray(eye, o, -up, 20.0, 0.0))
        .collect();
    let (hits, surfaces) =
        common::trace_stream(gpu, &world.chunks, RayClass::Nearest, eye, &down, None, 0);
    // Each ground point off its face as the reference offsets a ray leaving
    // a hit (2 mm, f32 ulps, a hundredth of the widest footprint).
    let mut ground = Vec::new();
    for (k, &o) in starts.iter().enumerate() {
        let Some((t, false)) = terrain_hit(&hits[k]) else {
            continue;
        };
        let face = DVec3::from_array(unpack_unit(surfaces[k].face).map(f64::from));
        if face.dot(up) < 0.5 {
            continue;
        }
        ground.push(o - up * t + face * (0.002 + 0.01));
    }
    let across = sun.cross(up).normalize();
    let across2 = sun.cross(across).normalize();
    let trace = |rays: &[SceneRay]| {
        common::trace_stream(gpu, &world.chunks, RayClass::Occlusion, eye, rays, None, 0).0
    };
    let ray = |p: DVec3, width: f32| {
        let mut r = common::scene_ray(eye, p, sun, 1e30, 0.0);
        r.cone = SceneRay::cone(width, 0.0, MASK_ALL);
        r
    };
    // The exact walk at the point and half a metre (the widest ray's
    // radius) either side across the sun.
    let mut offsets = vec![DVec3::ZERO];
    for k in 0..8 {
        let angle = f64::from(k) * std::f64::consts::FRAC_PI_4;
        offsets.push((across * angle.cos() + across2 * angle.sin()) * 0.5);
    }
    let exact: Vec<SceneRay> = ground
        .iter()
        .flat_map(|&p| offsets.iter().map(move |&o| (p, o)))
        .map(|(p, o)| ray(p + o, 0.0))
        .collect();
    let exact = trace(&exact);
    let lit = |h: &SceneHit| h.through()[0] >= 0.5;
    let (mut open, mut covered) = (Vec::new(), Vec::new());
    for (k, &p) in ground.iter().enumerate() {
        let walks = &exact[k * offsets.len()..(k + 1) * offsets.len()];
        if walks.iter().all(lit) {
            open.push(p);
        } else if !walks.iter().any(lit) {
            covered.push(p);
        }
    }
    eprintln!(
        "{} ground points: {} under open sky, {} under a canopy",
        ground.len(),
        open.len(),
        covered.len()
    );
    assert!(open.len() >= 100 && covered.len() >= 50);
    let finest = pages.frame.info[0];
    let mut answers = Vec::new();
    for width in [0.0f32, 0.01, 0.1, 1.0] {
        let stopped = trace(&open.iter().map(|&p| ray(p, width)).collect::<Vec<_>>())
            .iter()
            .filter(|h| !lit(h))
            .count();
        let passed = trace(&covered.iter().map(|&p| ray(p, width)).collect::<Vec<_>>())
            .iter()
            .filter(|h| lit(h))
            .count();
        eprintln!(
            "width {width} m (the pages' finest texel {finest} m): {stopped} of {} open \
             points stopped, {passed} of {} covered let through",
            open.len(),
            covered.len()
        );
        if width < finest {
            // Walked all the way: the exact answer.
            assert_eq!(stopped + passed, 0, "width {width}: walked rays differ");
        }
        answers.push((width, stopped, passed));
    }
    // Answered by the pages past their walked band: a page texel is a point
    // sample of the occluders' depth over a texel, so a receiver within a
    // texel of a shadow's edge can read either side (sun_pages_under_a_canopy
    // allows one in twelve over GI-shaped hits); over clear ground and under
    // a canopy at least half a metre across, at most one in fifty.
    for (width, stopped, passed) in answers {
        assert!(
            stopped * 50 <= open.len() && passed * 50 <= covered.len(),
            "width {width}: {stopped} open points stopped, {passed} covered let through"
        );
    }
}

/// What a world costs from nothing to its settled view, as a host that has
/// just opened a level pays it (Earth with its features, the engine's
/// capacity and residency settings, 1280x720 from the ground): making the
/// world (on the frame thread), the wait for its pipelines, the first chunk
/// drawn, the settled view, and the longest frame's CPU on the way. A
/// measurement: `--ignored --nocapture load_in`.
#[test]
#[ignore]
fn load_in_from_a_new_world() {
    let Some(gpu) = gpu() else { return };
    let started = std::time::Instant::now();
    let planet = Planet::new(PlanetRecipe {
        terrain: helio_pass_voxel_planet::layers::TerrainLayers::earth().source(7),
        features: helio_pass_voxel_planet::feature_recipe::FeatureRecipe::earth().grow(0.1, 7),
        ..PlanetRecipe::default()
    })
    .unwrap();
    let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
    eprintln!("LOADIN the CPU world built in {:.0} ms", ms(started));
    let view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [1280, 720]);
    let config = ResidencyConfig::default();
    // As a host makes it: the world and its volume in one frame's prepare,
    // their pipelines compiling off the frame while frames go on.
    let made = std::time::Instant::now();
    let mut chunks = ChunkWorld::with_capacity(&gpu.device, &gpu.queue, Capacity::default());
    chunks.add_volume(
        0,
        &planet,
        helio_pass_voxel_planet::VolumeTransform::IDENTITY,
    );
    let make = ms(made);
    let mut world = World {
        chunks,
        frame: 0,
        sun: Vec3::new(0.3, 0.8, 0.4).normalize(),
        scene_depth: None,
        lights: None,
        sun_occlusion: false,
        jitter: None,
        last: None,
    };
    let (mut waiting_frames, mut longest_waiting) = (0u64, 0.0f64);
    let (mut ready, mut first, mut longest, mut cpu) = (None, None, (0.0f64, 0u64), 0.0f64);
    let mut generated = Vec::new();
    loop {
        let frame = std::time::Instant::now();
        let stats = world.frame_with(gpu, &planet, &view, &config);
        let took = ms(frame);
        if took > longest.0 {
            longest = (took, world.frame);
        }
        cpu = cpu.max(stats.cpu_ms.iter().sum());
        if ready.is_none() {
            if world.chunks.pipelines_ready() {
                ready = Some((ms(made), world.frame));
            } else {
                // Frames while the pipelines compile: what a host's frame
                // thread (and whatever else shares its device) pays.
                waiting_frames += 1;
                longest_waiting = longest_waiting.max(took);
                longest = (0.0, 0);
            }
        }
        let count = world.chunks.chunk_count();
        if first.is_none() && count > 0 {
            first = Some((ms(made), world.frame));
        }
        if world.frame.is_multiple_of(30) {
            generated.push((ms(made) as u64, count));
        }
        if world.chunks.settled() {
            break;
        }
        if ready.is_none() {
            std::thread::sleep(std::time::Duration::from_millis(4));
        }
        assert!(made.elapsed().as_secs() < 300, "the view did not settle");
    }
    let (ready, first) = (ready.unwrap(), first.unwrap());
    eprintln!(
        "LOADIN world made in {make:.1} ms (frame thread); pipelines ready at {:.0} ms (frame {}); first chunk at {:.0} ms (frame {}); settled at {:.0} ms (frame {}), {} chunks; longest frame {:.1} ms (frame {}); longest update CPU {cpu:.2} ms",
        ready.0,
        ready.1,
        first.0,
        first.1,
        ms(made),
        world.frame,
        world.chunks.chunk_count(),
        longest.0,
        longest.1,
    );
    eprintln!(
        "LOADIN while the pipelines compiled: {waiting_frames} frames, the longest {longest_waiting:.1} ms"
    );
    eprintln!("LOADIN chunks resident by time (ms, chunks): {generated:?}");
}

/// A planner whose thread ends (a panic in it) is started again: the world
/// hands it every volume as new and the view streams back in to the same
/// hits. It used to keep the cut it had and never stream again.
#[test]
fn a_planner_that_dies_is_started_again_and_the_view_comes_back() {
    let Some(gpu) = gpu() else { return };
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let view = ground_view(&planet, land(&planet, 4, 0.37, 0.61), [320, 180]);
    let mut world = World::new(gpu, &planet);
    let before = world.settle(gpu, &planet, &view);
    let chunks = world.chunks.chunk_count();
    assert!(chunks > 0 && world.chunks.planner_restarts() == 0);

    world.chunks.fail_planner();
    let after = world.settle(gpu, &planet, &view);
    assert_eq!(world.chunks.planner_restarts(), 1);
    assert!(!world.chunks.planner_stopped());
    let hit = before.iter().filter(|h| h.is_some()).count();
    let same = before
        .iter()
        .zip(&after)
        .filter(|(a, b)| a.is_some() && a.map(|h| h.cell) == b.map(|h| h.cell))
        .count();
    assert!(hit > 1000, "{hit} pixels hit");
    assert!(
        same * 1000 >= hit * 995,
        "{same} of {hit} hits came back the same"
    );
    // And it goes on streaming: another view settles too.
    let turned = View::new(view.eye, -view.forward, view.size);
    world.settle(gpu, &planet, &turned);
    assert!(world.chunks.chunk_count() > 0);

    // A planner that keeps dying is given up on, and said so.
    for _ in 0..4 {
        world.chunks.fail_planner();
        world.frame(gpu, &planet, &view, None);
    }
    assert!(world.chunks.planner_stopped());
    assert_eq!(world.chunks.planner_restarts(), 3);
}
