//! Each voxel terrain's world, owned by its runtime
//! (`voxel_component::runtime`): built off the thread that asked, kept when
//! a change does not build, configured by its layers, answering the ground
//! methods and stamping sculpt samples. The terrains attach by class name
//! through the world registry, so the plugin is linked through its root.

// The plugin as a game links it: its root, with every registration.
use voxel_plugin as _;

use std::sync::Arc;
use std::time::Duration;

use glam::DVec3;
use helio_pass_voxel_planet::Planet;
use helio_pass_voxel_planet::engine::PickResult;
use helio_pass_voxel_planet::engine::{PickHit, VolumeAnswer};
use helio_pass_voxel_planet::terrain::material;
use helio_pass_voxel_planet::{PlanetRecipe, TerrainSource};
use helio_voxel_data::{VoxelBrushEdit, VoxelBrushOp, VoxelBrushShape};
use pulsar_scenedb::{Entity, World};
use voxel_component::runtime::{VoxelTerrainRuntime, VoxelWorlds, wait_for_voxel_worlds};
use voxel_component::{VoxelTerrainComponent, VoxelTerrainLayersComponent, VoxelTerrainMaterial};

/// The world `PlanetRecipe::default()` describes, as a terrain.
fn planet_terrain() -> VoxelTerrainComponent {
    let mut terrain = VoxelTerrainComponent::planet(6_371_000.0);
    terrain.generator_parameters = String::new();
    terrain.voxel_size = 0.1;
    terrain.seed = TerrainSource::default().seed;
    terrain
}

fn scene_with(terrain: VoxelTerrainComponent) -> (World, Entity) {
    let mut world = World::new();
    let owner = world.spawn();
    let entity = pulsar_world_registry::attach_value(&mut world, owner, terrain).unwrap();
    (world, entity)
}

fn settle(world: &mut World) {
    assert!(wait_for_voxel_worlds(world, Duration::from_secs(300)));
}

fn shown(world: &World, entity: Entity) -> Arc<Planet> {
    VoxelTerrainRuntime::of(world, entity)
        .and_then(|r| r.world())
        .expect("a built world")
}

/// A ground method by name, as cameras and scripts call it.
fn ground(world: &mut World, entity: Entity, method: &str, p: DVec3) -> Option<f64> {
    let mut args: Vec<Box<dyn std::any::Any>> = vec![Box::new(p.x), Box::new(p.y), Box::new(p.z)];
    let answer = world
        .call_component_method(
            entity,
            pulsar_scenedb::component_id::<VoxelTerrainComponent>(),
            method,
            &mut args,
        )
        .ok()??;
    answer.downcast::<f64>().ok().map(|v| *v)
}

fn dig(center: DVec3, radius: f64) -> VoxelBrushEdit {
    VoxelBrushEdit {
        center: center.to_array(),
        radius,
        shape: VoxelBrushShape::Sphere,
        op: VoxelBrushOp::Remove,
        material: 0,
        height: Default::default(),
    }
}

/// One world per terrain, built off the thread that asked: an edit is in
/// the journal at once and in the world once its build lands (the old
/// world shown meanwhile), with the blocks it broke as events.
#[test]
fn edits_extend_the_terrains_one_world_off_thread() {
    let (mut world, entity) = scene_with(planet_terrain());
    assert!(voxel_component::world::terrain_world(&world, entity).is_err());
    settle(&mut world);
    let first = shown(&world, entity);
    let eye = first.surface_point(DVec3::new(0.2, 1.0, 0.3), 3.0);
    let target = first.raycast(eye, -eye.normalize(), 100.0).unwrap().cell;
    let at = first.grid().cell_center(target);
    voxel_component::world::append_edits(&mut world, entity, vec![dig(at, 0.5)]).unwrap();
    assert!(Arc::ptr_eq(&first, &shown(&world, entity)), "not built yet");
    settle(&mut world);
    let dug = shown(&world, entity);
    assert_eq!(dug.material(target), 0);
    assert_ne!(first.material(target), 0, "old worlds stay as they were");
    let terrain = world.get::<VoxelTerrainComponent>(entity).unwrap();
    assert!(!terrain.pending_block_broken().is_empty());
    assert_eq!(terrain.edits.len(), 1);
}

/// A change that does not build keeps the last good world shown, says
/// why, and is not built again until it changes.
#[test]
fn a_world_that_does_not_build_keeps_the_last_good_one() {
    use helio_pass_voxel_planet::feature_recipe::FeatureRecipe;
    let mut terrain = planet_terrain();
    terrain.features = voxel_component::feature_spawns(&FeatureRecipe::earth());
    let (mut world, entity) = scene_with(terrain);
    settle(&mut world);
    let good = shown(&world, entity);
    assert_eq!(
        good.recipe().features,
        FeatureRecipe::earth().grow(0.1, TerrainSource::default().seed)
    );
    world
        .get_mut::<VoxelTerrainComponent>(entity)
        .unwrap()
        .features
        .reverse();
    settle(&mut world);
    let runtime = VoxelTerrainRuntime::of(&world, entity).unwrap();
    let error = runtime.error().expect("rejected");
    assert!(error.contains("widest spacing"), "{error}");
    assert!(Arc::ptr_eq(&good, &shown(&world, entity)));
    VoxelWorlds::default().step(&mut world);
    assert!(!runtime.building(), "not built again");
}

#[test]
fn a_layers_component_configures_the_generator() {
    let mut terrain = planet_terrain();
    terrain.seed = 99;
    let (mut world, entity) = scene_with(terrain);
    let owner = pulsar_scene_model::attachments::owner_of(&world, entity).unwrap();
    let mut layers = VoxelTerrainLayersComponent::default();
    layers.stack.climate.snow_c = -7.5;
    layers.stack.layers[2].scale_km = 55.0;
    pulsar_world_registry::attach_value(&mut world, owner, layers).unwrap();
    settle(&mut world);
    let recipe = shown(&world, entity).recipe().clone();
    assert_eq!(
        recipe.terrain.generator,
        helio_pass_voxel_planet::layers::ID
    );
    assert_eq!(recipe.terrain.seed, 99);
    let settings: helio_pass_voxel_planet::layers::TerrainLayers =
        serde_json::from_str(&recipe.terrain.settings).unwrap();
    assert_eq!(settings.climate.snow_c, -7.5);
    assert_eq!(settings.layers[2].scale_km, 55.0);
}

#[test]
fn plane_worlds_follow_the_component_and_its_settings() {
    let (mut world, entity) = scene_with(VoxelTerrainComponent::plane(1_024.0));
    let owner = pulsar_scene_model::attachments::owner_of(&world, entity).unwrap();
    let mut flat = VoxelTerrainLayersComponent::flat(12.0);
    flat.stack.surface = VoxelTerrainMaterial::Sand;
    pulsar_world_registry::attach_value(&mut world, owner, flat).unwrap();
    settle(&mut world);
    let planet = shown(&world, entity);
    assert!(planet.grid().is_plane());
    let hit = planet
        .raycast(DVec3::new(3.0, 40.0, -5.0), -DVec3::Y, 100.0)
        .unwrap();
    assert!((hit.distance - 28.0).abs() < 0.11, "{}", hit.distance);
    assert_eq!(planet.material(hit.cell), material::SAND);
    let eye = DVec3::new(10.0, 400.0, -20.0);
    let up = ["ground_up_x", "ground_up_y", "ground_up_z"]
        .map(|m| ground(&mut world, entity, m, eye).unwrap());
    assert_eq!(up, [0.0, 1.0, 0.0]);
}

/// The ground methods cameras call: altitude over the ground below,
/// clip planes that follow the clear air, and a lift out of solid
/// ground that leaves dug caves alone.
#[test]
fn ground_methods_answer_from_the_world_shown() {
    let (mut world, entity) = scene_with(planet_terrain());
    settle(&mut world);
    let planet = Planet::new(PlanetRecipe::default()).unwrap();
    let near = planet.surface_point(DVec3::Y, 2.0);
    let altitude = ground(&mut world, entity, "ground_altitude", near).unwrap();
    let orbit = ground(&mut world, entity, "ground_altitude", near * 1.05).unwrap();
    assert!(
        (altitude - 2.0).abs() < 0.2 && orbit > 250_000.0,
        "{altitude} {orbit}"
    );
    let high = planet.surface_point(DVec3::Y, 3000.0);
    let altitude = ground(&mut world, entity, "ground_altitude", high).unwrap();
    assert!((altitude - 3000.0).abs() < 1.0, "{altitude}");
    let far = near.normalize() * (planet.grid().radius() + 300_000.0);
    assert!(ground(&mut world, entity, "clip_near", far).unwrap() > 1_000.0);
    assert!(ground(&mut world, entity, "clip_far", far).unwrap() > far.length());

    let surface = planet.surface_point(DVec3::Y, 0.0);
    let buried = surface - DVec3::Y * 3.0;
    let lift = ground(&mut world, entity, "ground_lift", buried).unwrap();
    assert!((lift - 3.5).abs() < 0.2, "{lift}");
    assert_eq!(
        ground(&mut world, entity, "ground_lift", surface + DVec3::Y * 2.0),
        Some(0.0)
    );
    voxel_component::world::append_edits(&mut world, entity, vec![dig(buried, 1.5)]).unwrap();
    settle(&mut world);
    assert_eq!(ground(&mut world, entity, "ground_lift", buried), Some(0.0));
}

/// Sculpt samples stamp where the renderer drew the surface under them
/// (a pick answered into the runtime's inbox), or with no answer in time
/// on the exact terrain; the stamps go through the journal.
#[test]
fn sculpt_samples_stamp_where_the_ray_hits() {
    use voxel_component::sculpt::{VoxelBrushRequest, VoxelBrushTool, sculpt};
    let (mut world, entity) = scene_with(planet_terrain());
    settle(&mut world);
    let runtime = VoxelTerrainRuntime::of(&world, entity).unwrap();
    runtime.set_wait_for_bricks(true);
    let planet = shown(&world, entity);
    let eye = planet.surface_point(DVec3::new(0.2, 1.0, 0.3), 3.0);
    let down = -eye.normalize();
    let exact = planet.raycast(eye, down, 100.0).unwrap();
    let brush = VoxelBrushRequest {
        op: VoxelBrushOp::Remove,
        shape: VoxelBrushShape::Sphere,
        radius: 0.05,
        material: 0,
        single_block: false,
        tool: VoxelBrushTool::Stamp,
    };
    // Drawn far from the eye: the renderer's hit is the exact cell.
    let remote = eye - down * 300_000.0;
    sculpt(&mut world, entity, remote, down, brush, true).unwrap();
    // The renderer's side: the pick the runtime asks, answered into its inbox.
    let requests = runtime.volume_requests();
    let pick = requests.picks[0];
    let inbox = requests.answers.expect("the runtime's inbox");
    inbox.push(VolumeAnswer::Pick(PickResult {
        id: pick.id,
        hit: Some(PickHit {
            volume: entity.bits(),
            distance: 300_000.0 + exact.distance,
            cell: exact.cell,
            level: 0,
            entered: Some(4),
            material: 0,
        }),
    }));
    VoxelWorlds::default().step(&mut world);
    let edits = world
        .get::<VoxelTerrainComponent>(entity)
        .unwrap()
        .edits
        .clone();
    assert_eq!(edits.len(), 1);
    settle(&mut world);
    assert_eq!(shown(&world, entity).material(exact.cell), 0);
    // No answer in time: the exact terrain within reach; the next sample
    // near it stamps at once, into the hole the first made.
    sculpt(&mut world, entity, eye, down, brush, true).unwrap();
    std::thread::sleep(Duration::from_millis(200));
    VoxelWorlds::default().step(&mut world);
    let first = world
        .get::<VoxelTerrainComponent>(entity)
        .unwrap()
        .edits
        .len();
    assert_eq!(first, 2);
    sculpt(&mut world, entity, eye, down, brush, false).unwrap();
    let next = world
        .get::<VoxelTerrainComponent>(entity)
        .unwrap()
        .edits
        .clone();
    assert!(next.len() > first, "stamped at once");
    let [a, b] = [first - 1, next.len() - 1].map(|n| next.listed().nth(n).unwrap().center);
    assert!(
        DVec3::from_array(a).distance(eye) + 0.05 < DVec3::from_array(b).distance(eye),
        "deeper"
    );
}
