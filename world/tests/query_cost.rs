//! What CPU world queries cost on Earth with its features (a measurement,
//! run with `--ignored --nocapture`).

use glam::DVec3;
use helio_voxel_world::feature_recipe::FeatureRecipe;
use helio_voxel_world::layers::TerrainLayers;
use helio_voxel_world::{Brush, BrushOp, BrushShape, Cell, Matter, Planet, PlanetRecipe};
use std::time::Instant;

fn earth() -> Planet {
    let voxel = 0.1;
    Planet::new(PlanetRecipe {
        voxel_size_m: voxel,
        terrain: TerrainLayers::earth().source(7),
        features: FeatureRecipe::earth().grow(voxel, 7),
        ..Default::default()
    })
    .unwrap()
}

/// A surface point (1.7 m up) under the most feature cells around a few
/// directions: a forest.
fn forest(planet: &Planet) -> DVec3 {
    let mut best = (0, DVec3::ZERO);
    for t in 0..64 {
        let a = t as f64 * 0.37;
        let dir = DVec3::new(a.cos(), 0.8 + 0.1 * (a * 1.3).sin(), a.sin()).normalize();
        let eye = planet.surface_point_of(dir * planet.outer_radius(), 1.7, Matter::Ground);
        let all = planet.ground_height_of(eye, Matter::All);
        let ground = planet.ground_height_of(eye, Matter::Ground);
        let crowns = (ground - all).max(0.0);
        let score = (crowns * 10.0) as i32;
        if score > best.0 {
            best = (score, eye);
        }
    }
    best.1
}

fn time<T>(what: &str, n: usize, mut f: impl FnMut(usize) -> T) {
    let started = Instant::now();
    for i in 0..n {
        std::hint::black_box(f(i));
    }
    let ms = started.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "QUERY_COST {what}: {:.4} ms each ({n} in {ms:.1} ms)",
        ms / n as f64
    );
}

#[test]
#[ignore]
fn query_costs_on_earth() {
    let started = Instant::now();
    let planet = earth();
    eprintln!(
        "QUERY_COST build {:.1} ms",
        started.elapsed().as_secs_f64() * 1e3
    );
    let eye = forest(&planet);
    let up = planet.grid().up(eye);
    let (cell, _) = planet.grid().locate(eye);
    eprintln!(
        "QUERY_COST eye {eye:?} ground {:.2} m, all {:.2} m",
        planet.ground_height_of(eye, Matter::Ground),
        planet.ground_height_of(eye, Matter::All)
    );
    // A player capsule's cells (0.6 x 1.8 x 0.6 m at 0.1 m voxels), a frame.
    let capsule = |offset: i32| {
        let mut solid = 0;
        for i in -3..3 {
            for j in -3..3 {
                for k in -17..2 {
                    solid += planet.solid_of(
                        Cell::new(cell.face, cell.i + i + offset, cell.j + j, cell.k + k),
                        Matter::All,
                    ) as u32;
                }
            }
        }
        solid
    };
    time("capsule cells, All (684 cells)", 200, |i| {
        capsule(i as i32 % 50)
    });
    time("solid_of one cell, All", 20_000, |i| {
        planet.solid_of(
            Cell::new(cell.face, cell.i + (i % 200) as i32, cell.j, cell.k - 10),
            Matter::All,
        )
    });
    time("solid_of one cell, Ground", 20_000, |i| {
        planet.solid_of(
            Cell::new(cell.face, cell.i + (i % 200) as i32, cell.j, cell.k - 10),
            Matter::Ground,
        )
    });
    let forward = up.any_orthonormal_vector();
    time("raycast 100 m level through the forest, All", 200, |i| {
        let d = (forward * (i as f64 * 0.05).cos() + up.cross(forward) * (i as f64 * 0.05).sin())
            .normalize();
        planet.raycast(eye, d, 100.0)
    });
    time("raycast 30 m down-forward, All", 200, |i| {
        let d = (forward * (i as f64 * 0.05).cos() - up).normalize();
        planet.raycast(eye, d, 30.0)
    });
    time("ground_height_of, All", 200, |i| {
        planet.ground_height_of(eye + forward * i as f64, Matter::All)
    });
    time("ground_height_of, Ground", 200, |i| {
        planet.ground_height_of(eye + forward * i as f64, Matter::Ground)
    });
    time("air_clearance", 200, |i| {
        planet.air_clearance(eye + forward * i as f64)
    });
    time("surface_point_of, Ground", 200, |i| {
        planet.surface_point_of(eye + forward * i as f64, 0.5, Matter::Ground)
    });
    let high = eye + up * 3000.0;
    time("ground_height_of at 3 km, All", 50, |i| {
        planet.ground_height_of(high + forward * i as f64 * 10.0, Matter::All)
    });
}

/// A brick of cells, sampled at once, is each of its cells exactly: in a
/// forest (trunks, crowns and far cover) and through edits (a dug hole and
/// an added block).
#[test]
fn bricks_are_their_cells() {
    let mut planet = earth();
    let eye = forest(&planet);
    let up = planet.grid().up(eye);
    let side = up.any_orthonormal_vector();
    planet
        .apply(Brush {
            center: (eye - up * 2.0).to_array(),
            radius: 1.2,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
            height: 0.0,
        })
        .unwrap();
    planet
        .apply(Brush {
            center: (eye + side * 1.5).to_array(),
            radius: 0.4,
            shape: BrushShape::Cube,
            op: BrushOp::Add,
            material: 7,
            height: 0.0,
        })
        .unwrap();
    let (cell, _) = planet.grid().locate(eye);
    const N: i32 = 16;
    let (mut solid, mut features) = (0, 0);
    for (di, dj, dk) in [
        (0, 0, -1),
        (-1, 0, -1),
        (0, -1, 0),
        (0, 0, 0),
        (1, 1, -2),
        (0, 0, 1),
    ] {
        let origin = [
            (cell.i.div_euclid(N) + di) * N,
            (cell.j.div_euclid(N) + dj) * N,
            (cell.k.div_euclid(N) + dk) * N,
        ];
        let brick = planet.sample_brick(cell.face, origin, N, true);
        let bare = planet.sample_brick(cell.face, origin, N, false);
        for dj in 0..N {
            for di in 0..N {
                for dk in 0..N {
                    let index = ((dj * N + di) * N + dk) as usize;
                    let c = Cell::new(cell.face, origin[0] + di, origin[1] + dj, origin[2] + dk);
                    let cell = brick[index];
                    assert_eq!(
                        (cell.kind, cell.material),
                        planet.sample_kind(0, c.face, c.i, c.j, c.k),
                        "{c:?}"
                    );
                    assert_eq!(
                        cell.ground,
                        planet.solid_of(c, Matter::Ground),
                        "{c:?} ground"
                    );
                    assert_eq!(bare[index].kind == 1, cell.ground, "{c:?} without features");
                    solid += cell.kind;
                    features += u32::from((cell.kind == 1) != cell.ground);
                }
            }
        }
    }
    eprintln!("BRICKS {solid} solid cells, {features} of them features");
    assert!(features > 100, "the bricks hold features: {features}");
}

#[test]
#[ignore]
fn brick_cost_on_earth() {
    let planet = earth();
    let eye = forest(&planet);
    let (cell, _) = planet.grid().locate(eye);
    for (n, with_features) in [(16, true), (16, false), (32, true)] {
        time(
            &format!("sample_brick {n}^3 features {with_features}"),
            20,
            |t| {
                let origin = [
                    (cell.i.div_euclid(n) + t as i32 % 5) * n,
                    cell.j.div_euclid(n) * n,
                    (cell.k.div_euclid(n) - 1) * n,
                ];
                planet.sample_brick(cell.face, origin, n, with_features)
            },
        );
    }
}

/// What sampling bricks costs side by side: 64 bricks of a forest (cold
/// caches each time) one after another and on 8 threads, and how long one
/// cell's query (the game thread's) waits while 4 threads sample bricks. A
/// measurement, run with `--ignored --nocapture`.
#[test]
#[ignore]
fn bricks_sampled_side_by_side() {
    const N: i32 = 16;
    const THREADS: usize = 8;
    let origins = |planet: &Planet| {
        let (cell, _) = planet.grid().locate(forest(planet));
        let mut out = Vec::new();
        for dj in -4..4 {
            for di in -4..4 {
                out.push((
                    cell.face,
                    [
                        (cell.i.div_euclid(N) + di) * N,
                        (cell.j.div_euclid(N) + dj) * N,
                        (cell.k.div_euclid(N) - 1) * N,
                    ],
                ));
            }
        }
        (cell, out)
    };
    for round in 0..3 {
        let planet = earth();
        let (_, bricks) = origins(&planet);
        let started = Instant::now();
        for (face, origin) in &bricks {
            std::hint::black_box(planet.sample_brick(*face, *origin, N, true));
        }
        let one = started.elapsed().as_secs_f64() * 1e3;

        let planet = earth();
        let (_, bricks) = origins(&planet);
        let started = Instant::now();
        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let (planet, bricks) = (&planet, &bricks);
                scope.spawn(move || {
                    for (face, origin) in bricks.iter().skip(t).step_by(THREADS) {
                        std::hint::black_box(planet.sample_brick(*face, *origin, N, true));
                    }
                });
            }
        });
        let many = started.elapsed().as_secs_f64() * 1e3;
        // The same on a world a thread (nothing shared but the machine):
        // what the threads cost each other apart from the world's caches.
        let planets: Vec<Planet> = (0..THREADS).map(|_| earth()).collect();
        let started = Instant::now();
        std::thread::scope(|scope| {
            for (t, planet) in planets.iter().enumerate() {
                let bricks = &bricks;
                scope.spawn(move || {
                    for (face, origin) in bricks.iter().skip(t).step_by(THREADS) {
                        std::hint::black_box(planet.sample_brick(*face, *origin, N, true));
                    }
                });
            }
        });
        let apart = started.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "QUERY_COST round {round}: 64 bricks 16^3 one thread {one:.1} ms, {THREADS} threads {many:.1} ms ({:.2}x), a world each {apart:.1} ms ({:.2}x)",
            one / many,
            one / apart
        );

        // A reader beside the samplers: cells 40 m off, each a new column.
        let planet = earth();
        let (cell, bricks) = origins(&planet);
        let done = std::sync::atomic::AtomicBool::new(false);
        let mut waits: Vec<f64> = Vec::new();
        std::thread::scope(|scope| {
            for t in 0..4 {
                let (planet, bricks, done) = (&planet, &bricks, &done);
                scope.spawn(move || {
                    for (face, origin) in bricks.iter().skip(t).step_by(4) {
                        std::hint::black_box(planet.sample_brick(*face, *origin, N, true));
                    }
                    done.store(true, std::sync::atomic::Ordering::Release);
                });
            }
            let mut n = 0;
            while !done.load(std::sync::atomic::Ordering::Acquire) {
                let started = Instant::now();
                std::hint::black_box(planet.solid_of(
                    Cell::new(
                        cell.face,
                        cell.i + 400 + n % 600,
                        cell.j + n / 600,
                        cell.k - 3,
                    ),
                    Matter::All,
                ));
                waits.push(started.elapsed().as_secs_f64() * 1e3);
                n += 1;
            }
        });
        waits.sort_by(f64::total_cmp);
        let at = |share: f64| waits[((waits.len() - 1) as f64 * share) as usize];
        eprintln!(
            "QUERY_COST round {round}: one cell beside 4 samplers: {} queries, median {:.4} ms, p99 {:.3} ms, max {:.3} ms",
            waits.len(),
            at(0.5),
            at(0.99),
            at(1.0)
        );
    }
}

/// What making a focus resident costs through the query workers (16 m
/// around an eye in a forest, as a sculpting camera or a player keeps),
/// and how long the thread that asked waits on a call meanwhile. A
/// measurement, run with `--ignored --nocapture`.
#[test]
#[ignore]
fn a_focus_loads() {
    use helio_voxel_world::queries::WorldQueries;
    for (round, threads) in [(0, 2), (1, 2), (2, 8)] {
        let planet = std::sync::Arc::new(earth());
        let eye = forest(&planet);
        let queries = WorldQueries::new(planet.clone(), threads);
        let started = Instant::now();
        queries.focus(1, eye, 16.0);
        let first = started.elapsed().as_secs_f64() * 1e3;
        let wanted = queries.loading();
        let mut waits: Vec<f64> = Vec::new();
        loop {
            let asked = Instant::now();
            let left = queries.loading();
            waits.push(asked.elapsed().as_secs_f64() * 1e3);
            if left == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        let ms = started.elapsed().as_secs_f64() * 1e3;
        // The focus follows its eye: each move past half a brick lists it
        // again, on the thread that calls (a game's, every few frames).
        let side = planet.grid().up(eye).any_orthonormal_vector();
        let mut moves: Vec<f64> = (1..=20)
            .map(|n| {
                let asked = Instant::now();
                queries.focus(1, eye + side * f64::from(n), 16.0);
                asked.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        moves.sort_by(f64::total_cmp);
        eprintln!(
            "QUERY_COST round {round}: focus() first {first:.2} ms; moved 1 m: median {:.2} ms, max {:.2} ms",
            moves[moves.len() / 2],
            moves[moves.len() - 1]
        );
        waits.sort_by(f64::total_cmp);
        eprintln!(
            "QUERY_COST round {round}: a 16 m focus, {threads} workers: {wanted} bricks in {ms:.0} ms ({:.3} ms a brick); the asking thread's calls: median {:.4} ms, p99 {:.3} ms, max {:.3} ms",
            ms / wanted as f64,
            waits[waits.len() / 2],
            waits[(waits.len() - 1) * 99 / 100],
            waits[waits.len() - 1],
        );
    }
}
