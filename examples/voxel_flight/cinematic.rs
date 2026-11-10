//! `HELIO_VOXEL_FLIGHT_CINEMATIC=<name|script.json>`: a scripted, offline-
//! rendered camera flight. Every frame is drawn on the planet's streaming
//! until nothing is left to bring in, then for `_CONVERGE` frames more at the
//! same view (so the temporal upscaler's and the denoisers' history is the
//! frame's own), then saved as `<OUTPUT>/frame_NNNN.png` on a worker thread:
//! smooth at any frame rate, however long the GPU takes.
//!
//! cargo run -p helio-voxel-examples --release --bin voxel_flight -- OUTPUT 3840 2160 native
//!
//! Render larger than you will show (3840x2160 for 1080p): no frame is
//! upscaled or anti-aliased (`HELIO_VOXEL_FLIGHT_NO_AA`, on by default here:
//! no temporal upscaler, so no history to wait for), and the downscale
//! supersamples.
//!
//! A script is JSON (see `cinematic/*.json`; `planet_descent` is the default
//! name): `fps`, `seconds`, `sun_elevation_deg`, `anchors`, `segments`
//! (or `keys` alone: one track segment) and `events`.
//!
//! * A **track** segment flies a ground track: keys of `t`, `altitude_m`
//!   (over the ground, features included), `distance_m` (along the track
//!   from the spawn), `heading_deg` (left of travel), `pitch_deg` (over the
//!   horizon) and `sun_elevation_deg`; a key leaves out what it does not
//!   change. Curves are Catmull-Rom; altitude is interpolated in log space,
//!   so a descent from orbit to the ground eases like one. Script-level
//!   `"track": "mountain"` ends the track at the summit nearest the spawn,
//!   `track_length_m` away; `"track": "anchor:<name>"` ends it at an anchor
//!   (whose `near` is then looked for from the spawn).
//! * A **free** segment (`"mode": "free"`) starts exactly where the previous
//!   one ended, in position, look and speed, then moves through keys placed
//!   relative to **anchors**: `{"t", "around": name, "radius_m",
//!   "angle_deg", "height_m", "look_height_m"}` (a point on a circle round
//!   the anchor, looking at it; angle 0 is behind it along the track) or
//!   `{"t", "at": name, "right_m", "forward_m", "height_m", "look":
//!   {"right_m", "forward_m", "height_m"}}` (or `"look": {"anchor": name,
//!   "height_m"}`). Heights are over the *ground*
//!   (features passed through); the eye is kept `min_clearance_m` (0.6)
//!   over it.
//! * `anchors` (in order): `{"name", "kind": "clearing"|"feature",
//!   "near": {"distance_m", "side_m"} | {"anchor": name}, "radius_m"
//!   (clearing), "material" (feature, e.g. "Wood"), "skip_within_m"}`: the
//!   nearest open ground, or the nearest tree trunk.
//! * `events`: `{"t", "dig": {"radius_m", "range_m"}}`.
//!
//! Order and resuming: a frame every `_SPARSE` s (10) of the film is drawn
//! first, then the rest in order; frames already on disk are skipped, and a
//! frame is written whole or not at all, so a stopped render resumes where
//! it was (`_FRAMES=a-b` limits the range).
//!
//! Environment: `_CLOCK=s`, `_CLOCK_RATE=r` set the frame clock (frozen at
//! `s` seconds by default, so clouds, wind and the like repeat from run to
//! run); `_FPS`, `_SECONDS` override the script's; `_FRAMES=a-b`
//! renders only those; `_SUBFRAMES=n` (3) frames along the last 1/fps before
//! each saved one; `_SETTLE=n` (600) the most frames streaming may take;
//! `_CONVERGE=n` (24) frames at the final view; `_CHECK=1` reports where the
//! camera is inside solid voxels and stops. Streaming runs at offline
//! budgets and a finer level of detail (see `offline_defaults`).
//!
//! Encode, for example (downscaled from a 4K render):
//! `ffmpeg -framerate 24 -i frame_%04d.png -vf scale=1920:1080:flags=lanczos -c:v libx264 -crf 16 -pix_fmt yuv420p out.mp4`

use super::Flight;
use glam::{DQuat, DVec3, Vec3};
use helio_pass_voxel_planet::chunks::pass::ChunkPass;
use helio_pass_voxel_planet::{Brush, BrushOp, BrushShape, Cell, Matter};
use serde_json::Value;
use std::sync::Arc;

const BUILTIN: &[(&str, &str)] = &[
    (
        "planet_descent",
        include_str!("cinematic/planet_descent.json"),
    ),
    ("showcase", include_str!("cinematic/showcase.json")),
    ("crater", include_str!("cinematic/crater.json")),
];

/// Streaming for a render that is not in a hurry: more generation a frame,
/// and no coarsening for the eye's motion or for what is just off screen (a
/// turn does not show detail arriving). The level of detail itself stays at
/// the engine's cells of a pixel or less: rendered 2x over and downscaled,
/// they are half a pixel of what is shown (finer than that, the streaming
/// never settles). What the environment sets is left alone.
pub fn offline_defaults() {
    for (name, value) in [
        ("HELIO_VOXEL_CHUNKS_BUDGET", "1024"),
        ("HELIO_VOXEL_CHUNKS_GENERATE_MS", "40"),
        ("HELIO_VOXEL_CHUNKS_FRAME_MS", "120"),
        ("HELIO_VOXEL_OFFSCREEN_PIXELS", "1"),
        ("HELIO_VOXEL_MOTION_MAX", "1"),
        // Drawn larger than shown and downscaled: no temporal upscaler, no
        // anti-aliasing, every frame on its own.
        ("HELIO_VOXEL_FLIGHT_NO_AA", "1"),
    ] {
        if std::env::var_os(name).is_none() {
            std::env::set_var(name, value);
        }
    }
}

// ----------------------------------------------------------------- script

#[derive(Clone, Copy, Debug)]
struct Key {
    t: f64,
    altitude: f64,
    distance: f64,
    heading: f64,
    pitch: f64,
}

#[derive(Clone, Debug)]
enum Spec {
    Around {
        anchor: String,
        radius: f64,
        angle: f64,
        height: f64,
        look_height: f64,
    },
    At {
        anchor: String,
        right: f64,
        forward: f64,
        height: f64,
        look: [f64; 3],
        /// Look at this anchor (`look`'s height over it) instead.
        look_anchor: Option<String>,
    },
}

struct FreeKey {
    t: f64,
    spec: Spec,
}

enum SegmentSpec {
    Track(Vec<Key>),
    Free(Vec<FreeKey>),
}

struct AnchorSpec {
    name: String,
    clearing: bool,
    radius: f64,
    material: Option<u32>,
    /// Candidates closer than this to where it is looked for are passed over.
    skip: f64,
    near_anchor: Option<String>,
    near: (f64, f64),
}

struct Dig {
    t: f64,
    radius: f64,
    range: f64,
}

struct Script {
    name: String,
    fps: f64,
    seconds: f64,
    sun_elevation: Option<f64>,
    sun_keys: Vec<(f64, f64)>,
    to_mountain: bool,
    /// `"track": "anchor:<name>"`: the track ends at that anchor.
    to_anchor: Option<String>,
    track_length: f64,
    min_clearance: f64,
    anchors: Vec<AnchorSpec>,
    segments: Vec<SegmentSpec>,
    digs: Vec<Dig>,
}

fn number(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn load(spec: &str) -> Script {
    let text = match BUILTIN.iter().find(|(name, _)| *name == spec) {
        Some((_, text)) => (*text).to_string(),
        None => std::fs::read_to_string(spec).unwrap_or_else(|e| {
            panic!(
                "cinematic script {spec}: not a builtin ({}) or a file: {e}",
                BUILTIN.iter().map(|b| b.0).collect::<Vec<_>>().join(", ")
            )
        }),
    };
    let json: Value = serde_json::from_str(&text).expect("cinematic script: valid JSON");
    let env = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok());

    let mut sun_keys: Vec<(f64, f64)> = Vec::new();
    let mut segment_values: Vec<Value> = json["segments"].as_array().cloned().unwrap_or_default();
    if segment_values.is_empty() {
        segment_values.push(serde_json::json!({ "mode": "track", "keys": json["keys"] }));
    }
    let mut segments = Vec::new();
    let mut last_t = f64::NEG_INFINITY;
    for seg in &segment_values {
        let keys = seg["keys"].as_array().expect("a segment's `keys`");
        let free = seg["mode"].as_str() == Some("free");
        let mut track: Vec<Key> = Vec::new();
        let mut frees: Vec<FreeKey> = Vec::new();
        for k in keys {
            let t = number(k, "t").expect("a key's `t`");
            assert!(t > last_t, "cinematic script: key times must increase ({t})");
            last_t = t;
            if let Some(sun) = number(k, "sun_elevation_deg") {
                sun_keys.push((t, sun));
            }
            if free {
                let spec = if let Some(anchor) = k["around"].as_str() {
                    Spec::Around {
                        anchor: anchor.to_string(),
                        radius: number(k, "radius_m").unwrap_or(6.0),
                        angle: number(k, "angle_deg").unwrap_or(0.0),
                        height: number(k, "height_m").unwrap_or(1.8),
                        look_height: number(k, "look_height_m").unwrap_or(2.5),
                    }
                } else {
                    let look = &k["look"];
                    Spec::At {
                        anchor: k["at"].as_str().expect("`around` or `at`").to_string(),
                        right: number(k, "right_m").unwrap_or(0.0),
                        forward: number(k, "forward_m").unwrap_or(0.0),
                        height: number(k, "height_m").unwrap_or(1.8),
                        look: [
                            number(look, "right_m").unwrap_or(0.0),
                            number(look, "forward_m").unwrap_or(10.0),
                            number(look, "height_m").unwrap_or(2.0),
                        ],
                        look_anchor: look["anchor"].as_str().map(str::to_string),
                    }
                };
                frees.push(FreeKey { t, spec });
            } else {
                let prev = track.last().copied();
                track.push(Key {
                    t,
                    altitude: number(k, "altitude_m")
                        .or(prev.map(|p| p.altitude))
                        .expect("a key's `altitude_m`"),
                    distance: number(k, "distance_m")
                        .or(prev.map(|p| p.distance))
                        .unwrap_or(0.0),
                    heading: number(k, "heading_deg")
                        .or(prev.map(|p| p.heading))
                        .unwrap_or(0.0),
                    pitch: number(k, "pitch_deg")
                        .or(prev.map(|p| p.pitch))
                        .unwrap_or(-10.0),
                });
            }
        }
        segments.push(if free {
            SegmentSpec::Free(frees)
        } else {
            assert!(track.len() >= 2, "a track segment needs two keys");
            SegmentSpec::Track(track)
        });
    }
    let seconds_of_keys = last_t;

    let anchors = json["anchors"]
        .as_array()
        .map(|list| {
            list.iter()
                .map(|a| {
                    let near = &a["near"];
                    AnchorSpec {
                        name: a["name"].as_str().expect("an anchor's `name`").to_string(),
                        clearing: a["kind"].as_str() == Some("clearing"),
                        radius: number(a, "radius_m").unwrap_or(12.0),
                        material: a["material"].as_str().map(|m| {
                            helio_pass_voxel_planet::terrain::material::from_name(m)
                                .unwrap_or_else(|| panic!("unknown material {m}"))
                        }),
                        skip: number(a, "skip_within_m").unwrap_or(0.0),
                        near_anchor: near["anchor"].as_str().map(str::to_string),
                        near: (
                            number(near, "distance_m").unwrap_or(0.0),
                            number(near, "side_m").unwrap_or(0.0),
                        ),
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let digs = json["events"]
        .as_array()
        .map(|events| {
            events
                .iter()
                .filter_map(|e| {
                    let dig = e.get("dig")?;
                    Some(Dig {
                        t: number(e, "t")?,
                        radius: number(dig, "radius_m").unwrap_or(30.0),
                        range: number(dig, "range_m").unwrap_or(1000.0),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let track_length = number(&json, "track_length_m").unwrap_or_else(|| {
        segments
            .iter()
            .find_map(|s| match s {
                SegmentSpec::Track(keys) => keys.last().map(|k| k.distance),
                SegmentSpec::Free(_) => None,
            })
            .unwrap_or(0.0)
    });
    Script {
        name: json["name"].as_str().unwrap_or(spec).to_string(),
        fps: env("HELIO_VOXEL_FLIGHT_CINEMATIC_FPS")
            .or(number(&json, "fps"))
            .unwrap_or(24.0),
        seconds: env("HELIO_VOXEL_FLIGHT_CINEMATIC_SECONDS")
            .or(number(&json, "seconds"))
            .unwrap_or(seconds_of_keys),
        sun_elevation: number(&json, "sun_elevation_deg"),
        sun_keys,
        to_mountain: json["track"].as_str() == Some("mountain"),
        to_anchor: json["track"]
            .as_str()
            .and_then(|t| t.strip_prefix("anchor:"))
            .map(str::to_string),
        track_length,
        min_clearance: number(&json, "min_clearance_m").unwrap_or(0.6),
        anchors,
        segments,
        digs,
    }
}

// ----------------------------------------------------------------- curves

/// Catmull-Rom through four values at parameter `u` of the middle two.
fn catmull(p0: f64, p1: f64, p2: f64, p3: f64, u: f64) -> f64 {
    0.5 * (2.0 * p1
        + (p2 - p0) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u * u
        + (3.0 * p1 - p0 - 3.0 * p2 + p3) * u * u * u)
}

/// The track keys' value at `t`, by `get` of each key.
fn curve(keys: &[Key], t: f64, get: impl Fn(&Key) -> f64) -> f64 {
    let t = t.clamp(keys[0].t, keys.last().unwrap().t);
    let i = keys
        .windows(2)
        .position(|w| t <= w[1].t)
        .unwrap_or(keys.len() - 2);
    let (a, b) = (&keys[i], &keys[i + 1]);
    let u = (t - a.t) / (b.t - a.t);
    let before = get(&keys[i.saturating_sub(1)]);
    let after = get(&keys[(i + 2).min(keys.len() - 1)]);
    catmull(before, get(a), get(b), after, u)
}

/// A point of a free segment's path: time, eye and look (offsets from the
/// segment's base, so the numbers stay small at a planet's scale).
type Point = (f64, DVec3, DVec3);

/// The path through `points` at `t`. `points[0]` only shapes the start (the
/// previous segment's motion just before it); the path begins at `points[1]`.
fn spline(points: &[Point], t: f64) -> (DVec3, DVec3) {
    let t = t.clamp(points[1].0, points.last().unwrap().0);
    let i = (1..points.len() - 1)
        .find(|&i| t <= points[i + 1].0)
        .unwrap_or(points.len() - 2);
    let (a, b) = (&points[i], &points[i + 1]);
    let u = (t - a.0) / (b.0 - a.0);
    let before = &points[i - 1];
    let after = &points[(i + 2).min(points.len() - 1)];
    let along = |get: &dyn Fn(&Point) -> DVec3| {
        DVec3::new(
            catmull(get(before).x, get(a).x, get(b).x, get(after).x, u),
            catmull(get(before).y, get(a).y, get(b).y, get(after).y, u),
            catmull(get(before).z, get(a).z, get(b).z, get(after).z, u),
        )
    };
    (along(&|p| p.1), along(&|p| p.2))
}

// ------------------------------------------------------------ the timeline

/// The ground track a script's distances are along: `origin + ahead * d`.
struct Track {
    origin: DVec3,
    ahead: DVec3,
}

struct Seg {
    start: f64,
    end: f64,
    data: SegData,
}

enum SegData {
    Track(Vec<Key>),
    Free { base: DVec3, points: Vec<Point> },
}

struct Timeline {
    track: Track,
    segs: Vec<Seg>,
    min_clearance: f64,
    /// How far up the eye is carried to stay out of voxels, by time (free
    /// segments only): what each moment needs, widened and smoothed so the
    /// camera rises over a tree and settles back like a crane would.
    lift: Vec<(f64, f64)>,
}

/// The local frame at a point on the planet: up, the track's direction along
/// the ground, and right.
fn frame(at: DVec3, ahead: DVec3) -> (DVec3, DVec3, DVec3) {
    let up = at.normalize();
    let forward = (ahead - up * ahead.dot(up))
        .try_normalize()
        .unwrap_or_else(|| up.any_orthonormal_vector());
    (up, forward, forward.cross(up))
}

/// The ground point under `p` (features passed through).
fn ground(flight: &Flight, p: DVec3) -> DVec3 {
    flight.planet.surface_point_of(p, 0.0, Matter::Ground)
}

fn find_anchor(flight: &Flight, spec: &AnchorSpec, from: DVec3) -> DVec3 {
    let planet = &flight.planet;
    let grid = *planet.grid();
    let (base, _) = grid.locate(from);
    let step = (2.0 / grid.voxel_size()).round().max(1.0) as i32;
    let reach = (spec.radius / 2.0).ceil() as i32;
    let found = (0..160).find_map(|ring: i32| {
        (-ring..=ring)
            .flat_map(|a| (-ring..=ring).map(move |b| (a, b)))
            .filter(|(a, b)| a.abs().max(b.abs()) == ring)
            .find_map(|(a, b)| {
                let (i, j) = (base.i + a * step, base.j + b * step);
                let top = planet.column_top(base.face, i, j, 0);
                let feature = |i: i32, j: i32, k: i32| {
                    planet.features_at(0, base.face, i, j, k).flatten()
                };
                if spec.clearing {
                    // Nothing grows within `radius` of the cell.
                    let open = (-reach..=reach).all(|x| {
                        (-reach..=reach).all(|y| {
                            let (ci, cj) = (i + x * step, j + y * step);
                            let t = planet.column_top(base.face, ci, cj, 0);
                            // Cells, not metres: every 0.6 m up to 16 m (at 0.1 m).
                            (1..=160)
                                .step_by(6)
                                .all(|dk| feature(ci, cj, t + dk).is_none())
                        })
                    });
                    let c = grid.cell_center(Cell::new(base.face, i, j, top + 1));
                    (open && (c - from).length() >= spec.skip).then_some(c)
                } else {
                    [1, 3, 6].into_iter().find_map(|dk| {
                        let m = feature(i, j, top + dk)?;
                        let c = grid.cell_center(Cell::new(base.face, i, j, top + 1));
                        (spec.material.map_or(true, |want| want == m)
                            && (c - from).length() >= spec.skip)
                            .then_some(c)
                    })
                }
            })
    });
    let found = found.unwrap_or_else(|| panic!("anchor {}: nothing found near", spec.name));
    ground(flight, found)
}

impl Timeline {
    /// The eye and look of the segment at `t`, before the ground keeps the eye
    /// out of the ground.
    fn raw(&self, flight: &Flight, i: usize, t: f64) -> (DVec3, DVec3) {
        let seg = &self.segs[i];
        match &seg.data {
            SegData::Track(keys) => {
                let (eye, forward) = track_pose(flight, &self.track, keys, t);
                (eye, eye + forward.as_dvec3() * 100.0)
            }
            SegData::Free { base, points } => {
                let (eye, look) = spline(points, t);
                (*base + eye, *base + look)
            }
        }
    }

    /// The eye and look direction at `t`.
    fn pose(&self, flight: &Flight, t: f64) -> (DVec3, Vec3) {
        self.pose_with(flight, t, true)
    }

    /// The lift at `t`.
    fn lift_at(&self, t: f64) -> f64 {
        let l = &self.lift;
        if l.is_empty() || t <= l[0].0 || t >= l.last().unwrap().0 {
            return 0.0;
        }
        let i = l.partition_point(|p| p.0 <= t).saturating_sub(1);
        let (a, b) = (l[i], l[(i + 1).min(l.len() - 1)]);
        let u = ((t - a.0) / (b.0 - a.0).max(1e-9)).clamp(0.0, 1.0);
        a.1 + (b.1 - a.1) * u
    }

    fn pose_with(&self, flight: &Flight, t: f64, lifted: bool) -> (DVec3, Vec3) {
        let i = self
            .segs
            .iter()
            .position(|s| t <= s.end)
            .unwrap_or(self.segs.len() - 1);
        let t = t.clamp(self.segs[0].start, self.segs.last().unwrap().end);
        let (mut eye, look) = self.raw(flight, i, t);
        if matches!(self.segs[i].data, SegData::Free { .. }) {
            let floor = ground(flight, eye).length() + self.min_clearance;
            if eye.length() < floor {
                eye = eye.normalize() * floor;
            }
        }
        if lifted {
            eye += eye.normalize() * self.lift_at(t);
        }
        let forward = (look - eye).try_normalize().unwrap_or(DVec3::X);
        (eye, forward.as_vec3())
    }

    /// Fills [`Self::lift`]: for each moment of the free segments, the least
    /// lift that leaves the eye, and a ball round it, in air; widened and
    /// smoothed so the camera rises over a tree and settles back, then raised
    /// wherever smoothing left it in a solid, until nowhere is.
    fn plan_lift(&mut self, flight: &Flight) {
        let free: Vec<(f64, f64)> = self
            .segs
            .iter()
            .filter(|s| matches!(s.data, SegData::Free { .. }))
            .map(|s| (s.start, s.end))
            .collect();
        let step = 1.0 / 48.0;
        let blocked = |eye: DVec3, up: DVec3| {
            let side = up.any_orthonormal_vector();
            let other = up.cross(side);
            [DVec3::ZERO, side, -side, other, -other, up, -up]
                .iter()
                .any(|d| {
                    let (cell, _) = flight.planet.grid().locate(eye + *d * 0.9);
                    flight.planet.solid(cell)
                })
        };
        // The unlifted eye at each moment, and the least lift it needs alone.
        let mut eyes: Vec<(f64, DVec3)> = Vec::new();
        let mut raw: Vec<f64> = Vec::new();
        for (a, b) in free {
            let n = ((b - a) / step).ceil() as usize;
            for k in 0..=n {
                let t = (a + k as f64 * step).min(b);
                let (eye, _) = self.pose_with(flight, t, false);
                let up = eye.normalize();
                let mut lift = 0.0;
                while lift < 60.0 && blocked(eye + up * lift, up) {
                    lift += 0.25;
                }
                eyes.push((t, eye));
                raw.push(lift);
            }
        }
        if eyes.is_empty() {
            return;
        }
        // Widen (max over +-1.6 s), then smooth (two box passes of +-0.8 s,
        // which reach +-1.6 s: never below what a moment needs).
        let window = (0.8 / step) as isize;
        let pass = |v: &[f64], widen: bool| -> Vec<f64> {
            (0..v.len() as isize)
                .map(|i| {
                    let lo = (i - window).max(0) as usize;
                    let hi = ((i + window) as usize).min(v.len() - 1);
                    let w = &v[lo..=hi];
                    if widen {
                        w.iter().copied().fold(0.0, f64::max)
                    } else {
                        w.iter().sum::<f64>() / w.len() as f64
                    }
                })
                .collect()
        };
        let mut smooth = raw.clone();
        for round in 0..80 {
            smooth = pass(&pass(&pass(&pass(&raw, true), true), false), false);
            // A smoothed lift can carry the eye into another solid: raise
            // what is needed there, and smooth again.
            let mut raised = false;
            for (i, (_, eye)) in eyes.iter().enumerate() {
                let up = eye.normalize();
                if blocked(*eye + up * smooth[i], up) {
                    raw[i] = smooth[i] + 0.5;
                    raised = true;
                }
            }
            if !raised {
                eprintln!("CINEMATIC lift: settled after {round} rounds");
                break;
            }
        }
        let top = smooth.iter().copied().fold(0.0, f64::max);
        eprintln!("CINEMATIC lift: at most {top:.1} m over the planned path");
        self.lift = eyes.iter().map(|e| e.0).zip(smooth).collect();
    }
}

/// A track segment's eye and look at `t`.
fn track_pose(flight: &Flight, track: &Track, keys: &[Key], t: f64) -> (DVec3, Vec3) {
    let distance = curve(keys, t, |k| k.distance);
    let altitude = (curve(keys, t, |k| (k.altitude + 1.0).ln()).exp() - 1.0).max(1.0);
    let yaw = curve(keys, t, |k| k.heading).to_radians();
    let pitch = curve(keys, t, |k| k.pitch);

    let ahead = track.ahead;
    let at = |d: f64| track.origin + ahead * d;
    let planet = &flight.planet;
    let dir = at(distance).normalize();

    // The ground under the track, smoothed over a span that grows with
    // altitude, so a low pass rises over a mountain instead of jolting.
    let spread = (altitude * 1.5).clamp(20.0, 8000.0);
    let taps = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mean = taps
        .iter()
        .map(|k| {
            planet
                .surface_point(at(distance + k * spread * 0.5), 0.0)
                .length()
        })
        .sum::<f64>()
        / taps.len() as f64;
    let here = planet.surface_point(dir, 0.0).length();
    let radius = (mean + altitude).max(here + 3.0 + 0.35 * altitude.min(200.0));
    let eye = dir * radius;

    // Direction of travel at the eye, turned `yaw` about up, tipped `pitch`.
    let up = dir;
    let step = (at(distance + 1000.0).normalize() - dir).normalize();
    let level = (step - up * step.dot(up)).try_normalize().unwrap_or(ahead);
    let level = DQuat::from_axis_angle(up, yaw) * level;
    let p = pitch.to_radians();
    let forward = (level * p.cos() + up * p.sin()).normalize().as_vec3();
    (eye, forward)
}

/// Builds the timeline: resolves anchors, then each segment in turn (a free
/// one starts from the pose the previous one ends in).
fn build(flight: &Flight, script: &Script, ground_point: DVec3, heading: f64) -> Timeline {
    let planet = &flight.planet;
    let mut track = if script.to_mountain {
        let (dir, height) = super::highest_near(planet, 2, 0.47, 0.53, 0.15);
        let peak = planet.surface_point(dir, 0.0);
        let ahead = super::tangent(peak, heading).as_dvec3();
        eprintln!("CINEMATIC track ends at the summit ({height:.0} m)");
        Track {
            origin: peak - ahead * script.track_length,
            ahead,
        }
    } else {
        Track {
            origin: ground_point,
            ahead: super::tangent(ground_point, heading).as_dvec3(),
        }
    };

    let mut anchors: Vec<(String, DVec3)> = Vec::new();
    for spec in &script.anchors {
        let from = match &spec.near_anchor {
            Some(name) => {
                anchors
                    .iter()
                    .find(|a| &a.0 == name)
                    .unwrap_or_else(|| panic!("anchor {name} is not defined before use"))
                    .1
            }
            None => {
                let on_track = track.origin + track.ahead * spec.near.0;
                let (_, _, right) = frame(on_track, track.ahead);
                on_track + right * spec.near.1
            }
        };
        let point = find_anchor(flight, spec, from);
        eprintln!(
            "CINEMATIC anchor {}: {:.1} m from where it was looked for",
            spec.name,
            (point - from).length()
        );
        // What stands over it, a mark each half metre up from the ground.
        let up = point.normalize();
        let profile: String = (0..48)
            .map(|k| {
                let at = point + up * (f64::from(k) * 0.5 + 0.05);
                let (cell, _) = planet.grid().locate(at);
                if planet.solid(cell) { '#' } else { '.' }
            })
            .collect();
        eprintln!("CINEMATIC anchor {} column (0-24 m): {profile}", spec.name);
        anchors.push((spec.name.clone(), point));
    }
    // A sun for the film's ground (`HELIO_VOXEL_FLIGHT_SUN=x,y,z` makes the
    // scene's light and the sky's sun; the terrain's follows it): `azimuth`
    // degrees round from the direction of travel, `elevation` over the
    // horizon, at the first anchor.
    if let Some((_, at)) = anchors.first() {
        let (up, forward, right) = frame(*at, track.ahead);
        for (az, el) in [(0.0f64, 40.0f64), (60.0, 40.0), (90.0, 40.0), (-60.0, 40.0), (135.0, 40.0)] {
            let (a, e) = (az.to_radians(), el.to_radians());
            let sun = (forward * a.cos() + right * a.sin()) * e.cos() + up * e.sin();
            eprintln!(
                "CINEMATIC sun azimuth {az} elevation {el}: HELIO_VOXEL_FLIGHT_SUN={:.5},{:.5},{:.5}",
                sun.x, sun.y, sun.z
            );
        }
    }
    // The track ends at an anchor: its origin is `track_length_m` back.
    if let Some(name) = &script.to_anchor {
        let point = anchors
            .iter()
            .find(|a| &a.0 == name)
            .unwrap_or_else(|| panic!("track anchor {name} is not defined"))
            .1;
        let ahead = super::tangent(point, heading).as_dvec3();
        track = Track {
            origin: point - ahead * script.track_length,
            ahead,
        };
    }
    let anchor = |name: &str| {
        anchors
            .iter()
            .find(|a| a.0 == name)
            .unwrap_or_else(|| panic!("unknown anchor {name}"))
            .1
    };

    let mut timeline = Timeline {
        track,
        segs: Vec::new(),
        min_clearance: script.min_clearance,
        lift: Vec::new(),
    };
    let mut start = 0.0;
    for spec in &script.segments {
        match spec {
            SegmentSpec::Track(keys) => {
                let end = keys.last().unwrap().t;
                timeline.segs.push(Seg {
                    start: keys[0].t,
                    end,
                    data: SegData::Track(keys.clone()),
                });
                // The next segment starts where this one ends.
                start = end;
            }
            SegmentSpec::Free(keys) => {
                // Where the last segment ends and how it was moving.
                let i = timeline.segs.len() - 1;
                let lead_t = (start - 0.3).max(timeline.segs[i].start);
                let (eye0, look0) = timeline.raw(flight, i, start);
                let (lead_eye, lead_look) = timeline.raw(flight, i, lead_t);
                let base = eye0;
                let mut points: Vec<Point> = vec![
                    (lead_t, lead_eye - base, lead_look - base),
                    (start, eye0 - base, look0 - base),
                ];
                for k in keys {
                    let (eye, look) = match &k.spec {
                        Spec::Around {
                            anchor: name,
                            radius,
                            angle,
                            height,
                            look_height,
                        } => {
                            let a = anchor(name);
                            let (up, forward, right) = frame(a, timeline.track.ahead);
                            let a_rad = angle.to_radians();
                            let flat = a + (-forward * a_rad.cos() + right * a_rad.sin()) * *radius;
                            let g = ground(flight, flat);
                            (g + g.normalize() * *height, a + up * *look_height)
                        }
                        Spec::At {
                            anchor: name,
                            right: r,
                            forward: f,
                            height,
                            look,
                            look_anchor,
                        } => {
                            let a = anchor(name);
                            let (up, forward, right) = frame(a, timeline.track.ahead);
                            let g = ground(flight, a + forward * *f + right * *r);
                            let target = match look_anchor {
                                Some(other) => anchor(other),
                                None => ground(flight, a + forward * look[1] + right * look[0]),
                            };
                            (g + g.normalize() * *height, target + up * look[2])
                        }
                    };
                    points.push((k.t, eye - base, look - base));
                }
                let end = keys.last().unwrap().t;
                timeline.segs.push(Seg {
                    start,
                    end,
                    data: SegData::Free { base, points },
                });
                start = end;
            }
        }
    }
    timeline.plan_lift(flight);
    timeline
}

// ------------------------------------------------------------------- run

fn settle(flight: &mut Flight, eye: DVec3, forward: Vec3, max: u32, converge: u32) -> u32 {
    let mut frames = 0;
    for n in 1..=max {
        flight.draw(eye, forward);
        let _ = flight.device.poll(wgpu::PollType::wait_indefinitely());
        frames = n;
        if n >= 2
            && flight
                .renderer
                .find_pass::<ChunkPass>()
                .is_some_and(ChunkPass::settled)
        {
            break;
        }
    }
    // The view's own history, for the upscaler and the denoisers.
    for _ in 0..converge {
        flight.draw(eye, forward);
    }
    if std::env::var_os("HELIO_VOXEL_FLIGHT_CINEMATIC_STATS").is_some() {
        let pages = flight
            .renderer
            .find_pass::<ChunkPass>()
            .and_then(ChunkPass::sun_page_stats);
        eprintln!("CINEMATIC sun pages after {} draws: {pages:?}", frames + converge);
    }
    frames + converge
}

fn record(flight: &mut Flight, script: &Script, timeline: &Timeline, total: u32) {
    let hz: u32 = std::env::var("HELIO_VOXEL_FLIGHT_CINEMATIC_RECORD_HZ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120);
    let per_frame = ((f64::from(hz) / script.fps).round() as u32).max(1);
    let step = 1.0 / f64::from(hz);
    flight.renderer.set_frame_clock_delta(Some(step as f32));
    flight.renderer.set_frame_delta_override(Some(step as f32));
    let started = std::time::Instant::now();
    let mut frame = 0u32;
    let mut n = 0u64;
    while frame < total {
        let t = n as f64 * step;
        let (eye, forward) = timeline.pose(flight, t);
        flight.draw(eye, forward);
        if n % u64::from(per_frame) == 0 {
            let _ = flight.device.poll(wgpu::PollType::wait_indefinitely());
            flight.capture_async(&format!("frame_{frame:04}"));
            if frame % 24 == 0 {
                eprintln!(
                    "CINEMATIC record frame {frame}/{} t {t:.2} s ({:.0} s elapsed)",
                    total - 1,
                    started.elapsed().as_secs_f64()
                );
            }
            frame += 1;
        }
        n += 1;
        // Never ahead of the wall clock.
        let due = std::time::Duration::from_secs_f64(n as f64 * step);
        if let Some(wait) = due.checked_sub(started.elapsed()) {
            std::thread::sleep(wait);
        }
    }
    flight.finish_captures();
    eprintln!("CINEMATIC record done: {total} frames");
}

pub fn run(flight: &mut Flight, spec: &str, ground_point: DVec3, heading: f64) {
    let script = load(spec);
    let timeline = build(flight, &script, ground_point, heading);
    let total = (script.seconds * script.fps).round() as u32;
    let dt = 1.0 / script.fps;
    let env_u32 = |name: &str, default: u32| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };

    // `_RECORD=1`: the engine as it runs live (its own streaming budgets,
    // temporal upscaler, no settling). The camera follows the script at
    // `_RECORD_HZ` draws per second of the film (120), paced to the wall
    // clock so the streaming gets the time it would live; every
    // `hz / fps`-th draw is written out.
    if std::env::var_os("HELIO_VOXEL_FLIGHT_CINEMATIC_RECORD").is_some() {
        record(flight, &script, &timeline, total);
        return;
    }

    // `_CHECK`: where the eye is inside solid voxels.
    if std::env::var_os("HELIO_VOXEL_FLIGHT_CINEMATIC_CHECK").is_some() {
        let mut inside = 0;
        for n in 0..total * 2 {
            let t = f64::from(n) * dt * 0.5;
            let (eye, _) = timeline.pose(flight, t);
            if n % 12 == 0 {
                eprintln!(
                    "CINEMATIC check: t {t:.2} s eye {:.1} m over the ground, {:.0} m from the planet's centre line",
                    eye.length() - ground(flight, eye).length(),
                    eye.length()
                );
            }
            let (cell, _) = flight.planet.grid().locate(eye);
            if flight.planet.solid(cell) {
                inside += 1;
                eprintln!(
                    "CINEMATIC check: t {t:.2} s the eye is inside solid (ground {:.1} m under)",
                    eye.length() - ground(flight, eye).length()
                );
            }
        }
        eprintln!("CINEMATIC check: {inside} samples inside solid of {}", total * 2);
        return;
    }

    let (first, last) = match std::env::var("HELIO_VOXEL_FLIGHT_CINEMATIC_FRAMES") {
        Ok(range) => {
            let (a, b) = range.split_once('-').expect("_FRAMES=a-b");
            (a.parse().unwrap(), b.parse::<u32>().unwrap().min(total - 1))
        }
        Err(_) => (0, total - 1),
    };
    // Without a temporal upscaler a frame needs no history: only the
    // streaming is waited for.
    let no_aa = std::env::var_os("HELIO_VOXEL_FLIGHT_NO_AA").is_some();
    let subframes = env_u32("HELIO_VOXEL_FLIGHT_CINEMATIC_SUBFRAMES", if no_aa { 1 } else { 3 }).max(1);
    let settle_max = env_u32("HELIO_VOXEL_FLIGHT_CINEMATIC_SETTLE", 600);
    let converge = env_u32("HELIO_VOXEL_FLIGHT_CINEMATIC_CONVERGE", if no_aa { 0 } else { 24 });
    eprintln!(
        "CINEMATIC {}: {total} frames at {} fps ({:.1} s), rendering {first}..={last} at {}x{}",
        script.name, script.fps, script.seconds, flight.size[0], flight.size[1]
    );

    // The sun's elevation over time: smooth between the keys that set it.
    let sun_at = |t: f64| -> Option<f32> {
        let keys = &script.sun_keys;
        let base = script.sun_elevation;
        let first = keys.first().copied().or(base.map(|b| (0.0, b)))?;
        let mut all = vec![(0.0, base.unwrap_or(first.1))];
        all.extend(keys.iter().copied());
        let i = all.windows(2).position(|w| t <= w[1].0);
        Some(match i {
            None => all.last().unwrap().1 as f32,
            Some(i) => {
                let (a, b) = (all[i], all[i + 1]);
                let u = ((t - a.0) / (b.0 - a.0).max(1e-9)).clamp(0.0, 1.0);
                let s = u * u * (3.0 - 2.0 * u);
                (a.1 + (b.1 - a.1) * s) as f32
            }
        })
    };

    // The frame clock (the sky's clouds and weather, foliage wind, material
    // time) follows the wall clock unless the host sets it, so two runs, or
    // two frames that took different times, have different skies. Here it
    // only moves when the script says: `_CLOCK=s` freezes it at `s` seconds
    // (a resumed or repeated render gets the same sky), `_CLOCK_RATE=r` has
    // it advance `r` seconds a second of the film (0, frozen, by default).
    let clock_start: f32 = std::env::var("HELIO_VOXEL_FLIGHT_CINEMATIC_CLOCK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let clock_rate: f64 = std::env::var("HELIO_VOXEL_FLIGHT_CINEMATIC_CLOCK_RATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    flight.renderer.set_frame_clock_delta(Some(0.0));
    // And the frame delta temporal filters read (the shadow denoiser, the
    // probes of global illumination) is a constant, not how long the last
    // draw happened to take.
    flight.renderer.set_frame_delta_override(Some(1.0 / 60.0));
    let mut clock_now = f64::from(clock_start);

    // Events before `first` still apply (resuming a render keeps its edits).
    let mut dug = vec![false; script.digs.len()];
    let started = std::time::Instant::now();
    // The order: a frame every `_SPARSE` seconds of the film first (10: the
    // whole film at a glance, early), then the rest in order. A frame is
    // the same whatever order it is drawn in (the clock and the frame delta
    // are fixed, the streaming settles); edits (`events`) need the film in
    // order. Frames already on disk are skipped (`_SKIP_EXISTING=0` redraws).
    let sparse_s: f64 = std::env::var("HELIO_VOXEL_FLIGHT_CINEMATIC_SPARSE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10.0);
    let order: Vec<u32> = if script.digs.is_empty() {
        let stride = ((sparse_s * script.fps).round() as u32).max(1);
        let mut v: Vec<u32> = (first..=last).filter(|f| f % stride == 0).collect();
        v.extend((first..=last).filter(|f| f % stride != 0));
        v
    } else {
        (0..=last).collect()
    };
    let skip_existing = std::env::var("HELIO_VOXEL_FLIGHT_CINEMATIC_SKIP_EXISTING")
        .map_or(true, |v| v != "0");
    let mut previous: Option<u32> = None;
    for &frame in &order {
        let t = f64::from(frame) * dt;
        for (i, dig) in script.digs.iter().enumerate() {
            if dug[i] || t < dig.t {
                continue;
            }
            dug[i] = true;
            let (eye, forward) = timeline.pose(flight, t);
            let dir = forward.as_dvec3();
            match flight.planet.raycast(eye, dir, dig.range) {
                Some(hit) => {
                    let at = eye + dir * hit.distance;
                    let mut planet = (*flight.planet).clone();
                    planet
                        .apply(Brush {
                            center: at.to_array(),
                            radius: dig.radius,
                            shape: BrushShape::Sphere,
                            op: BrushOp::Remove,
                            material: 0,
                            height: 0.0,
                        })
                        .unwrap();
                    flight.planet = Arc::new(planet);
                    eprintln!(
                        "CINEMATIC dig {} m at {:.0} m, t {t:.2} s",
                        dig.radius, hit.distance
                    );
                }
                None => eprintln!(
                    "CINEMATIC dig at t {t:.2} s: the view's centre meets no ground within {} m",
                    dig.range
                ),
            }
        }
        if frame < first {
            continue;
        }
        if skip_existing && flight.output.join(format!("frame_{frame:04}.png")).exists() {
            previous = None;
            continue;
        }
        flight.sun_elevation = sun_at(t);
        // The clock is where the film puts it: one draw moves it there and
        // the rest stand still.
        let want = f64::from(clock_start) + t * clock_rate;
        let jump = want - clock_now;
        if jump.abs() > 1e-6 || frame == first {
            let (eye, forward) = timeline.pose(flight, t);
            flight.renderer.set_frame_clock_delta(Some(jump as f32));
            flight.draw(eye, forward);
            flight.renderer.set_frame_clock_delta(Some(0.0));
            clock_now = want;
        }
        // The frames before this one's instant, so the upscaler and the
        // streaming see the motion; then the settled frame itself.
        for sub in 0..subframes {
            let back = f64::from(subframes - sub) / f64::from(subframes) * dt;
            let (eye, forward) = timeline.pose(flight, (t - back).max(0.0));
            flight.draw(eye, forward);
        }
        let (eye, forward) = timeline.pose(flight, t);
        // A frame that does not follow the last one starts with little of it
        // streamed or lit: it settles, and its lighting converges, longer.
        let adjacent = previous.map_or(false, |p| p + 1 == frame);
        let extra = if adjacent { converge } else { converge.max(48) };
        previous = Some(frame);
        let spent = settle(flight, eye, forward, settle_max, extra);
        let _ = flight.device.poll(wgpu::PollType::wait_indefinitely());
        flight.capture_async(&format!("frame_{frame:04}"));
        eprintln!(
            "CINEMATIC frame {frame}/{} t {t:.2} s alt {:.1} m, {spent} frames ({:.0} s elapsed)",
            total - 1,
            eye.length() - flight.planet.surface_point(eye, 0.0).length(),
            started.elapsed().as_secs_f64()
        );
    }
    flight.finish_captures();
    eprintln!(
        "CINEMATIC done: ffmpeg -framerate {} -i frame_%04d.png -vf scale=1920:1080:flags=lanczos -c:v libx264 -crf 16 -pix_fmt yuv420p out.mp4",
        script.fps
    );
}
