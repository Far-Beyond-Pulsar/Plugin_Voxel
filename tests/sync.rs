//! A voxel terrain's dependents follow it (`voxel_component::sync`): edits
//! belong to the ground they were made on, and a world's sky and haze are
//! its air. The terrain and the sky and haze it drives are reached by class
//! name through the world registry, so the plugin and those kinds are
//! linked through their roots.

// The plugin as a game links it: its root, with every registration.
use voxel_plugin as _;
// The kinds a voxel world composes register through their roots.
use fog_plugin as _;
use sky_plugin as _;

use pulsar_scenedb::World;
use voxel_component::sync::{VoxelWorldSync, sync_air_to_terrains, sync_edit_journals};
use voxel_component::world::entity_source;
use voxel_component::{VoxelTerrainComponent, VoxelWorldShape};

/// Edits belong to the ground they were made on: changing the seed or
/// the layer stack drops them; the form's unused size does not.
#[test]
fn edits_are_dropped_when_the_ground_changes() {
    use voxel_component::{VoxelLayerKind, VoxelTerrainLayer, VoxelTerrainLayersComponent};
    let mut world = World::new();
    let object = world.spawn();
    let entity = pulsar_world_registry::attach_value(
        &mut world,
        object,
        VoxelTerrainComponent::planet(1_000.0),
    )
    .unwrap();
    let layers = pulsar_world_registry::attach_value(
        &mut world,
        object,
        VoxelTerrainLayersComponent::default(),
    )
    .unwrap();
    let edit = helio_voxel_data::VoxelBrushEdit {
        center: [0.0, 1_000.0, 0.0],
        radius: 1.0,
        shape: helio_voxel_data::VoxelBrushShape::Sphere,
        op: helio_voxel_data::VoxelBrushOp::Remove,
        material: 0,
        height: Default::default(),
    };
    let edits = |world: &World| {
        world
            .get::<VoxelTerrainComponent>(entity)
            .unwrap()
            .edits
            .len()
    };
    let projected = |world: &World| {
        let c = world.get::<VoxelTerrainComponent>(entity).unwrap();
        voxel_component::runtime::entity_target(world, entity, c)
            .unwrap()
            .edits
            .len()
    };
    world
        .get_mut::<VoxelTerrainComponent>(entity)
        .unwrap()
        .edits
        .push(edit);
    // A new journal adopts the terrain it is on.
    assert_eq!(sync_edit_journals(&mut world), 0);
    assert_eq!((edits(&world), projected(&world)), (1, 1));
    world
        .get_mut::<VoxelTerrainComponent>(entity)
        .unwrap()
        .plane_size = 77.0;
    assert_eq!(
        sync_edit_journals(&mut world),
        0,
        "a planet does not use the plane size"
    );
    // Another seed: other ground. The projection already shows none.
    world.get_mut::<VoxelTerrainComponent>(entity).unwrap().seed += 1;
    assert_eq!(projected(&world), 0);
    assert_eq!(sync_edit_journals(&mut world), 1);
    assert_eq!(edits(&world), 0);
    // So does another layer stack.
    world
        .get_mut::<VoxelTerrainComponent>(entity)
        .unwrap()
        .edits
        .push(edit);
    assert_eq!(sync_edit_journals(&mut world), 0);
    world
        .get_mut::<VoxelTerrainLayersComponent>(layers)
        .unwrap()
        .stack
        .layers
        .push(VoxelTerrainLayer::new(VoxelLayerKind::Craters));
    assert_eq!(sync_edit_journals(&mut world), 1);
    assert_eq!((edits(&world), projected(&world)), (0, 0));
}

#[test]
fn a_terrains_sky_follows_its_size_and_shape() {
    use sky_component::{AtmosphereComponent, AtmospherePlacement};
    let mut world = World::new();
    let object = world.spawn();
    let terrain = pulsar_world_registry::attach_value(
        &mut world,
        object,
        VoxelTerrainComponent::planet(250_000.0),
    )
    .unwrap();
    let sky =
        pulsar_world_registry::attach_value(&mut world, object, AtmosphereComponent::default())
            .unwrap();
    let mut sync = VoxelWorldSync::new(&world);
    let air = |world: &World| {
        let a = world.get::<AtmosphereComponent>(sky).unwrap();
        (a.placement, a.planet_radius_km)
    };
    sync.poll(&mut world);
    assert_eq!(air(&world), (AtmospherePlacement::PlanetAtOwner, 250.0));
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .planet_radius = 40_000.0;
    sync.poll(&mut world);
    assert_eq!(air(&world), (AtmospherePlacement::PlanetAtOwner, 40.0));
    // A plane's sky is flat ground: the Earth's curvature, not the old planet's.
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .shape = VoxelWorldShape::Plane;
    sync.poll(&mut world);
    let earth = AtmosphereComponent::default().planet_radius_km;
    assert_eq!(air(&world), (AtmospherePlacement::GroundAtOrigin, earth));
    // A sky edited by hand on a planet goes back onto it.
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .shape = VoxelWorldShape::Sphere;
    sync.poll(&mut world);
    world
        .get_mut::<AtmosphereComponent>(sky)
        .unwrap()
        .planet_radius_km = 9.0;
    sync.poll(&mut world);
    assert_eq!(air(&world), (AtmospherePlacement::PlanetAtOwner, 40.0));
    assert_eq!(sync_air_to_terrains(&mut world), 0);
}

#[test]
fn a_worlds_air_preset_makes_its_sky() {
    use sky_component::AtmosphereComponent;
    use voxel_component::VoxelAir;
    let mut world = World::new();
    let object = world.spawn();
    let terrain = pulsar_world_registry::attach_value(
        &mut world,
        object,
        VoxelTerrainComponent::planet(1_737_400.0),
    )
    .unwrap();
    let mut authored = AtmosphereComponent::default();
    authored.mie_anisotropy = 0.1;
    let sky = pulsar_world_registry::attach_value(&mut world, object, authored).unwrap();
    let mut sync = VoxelWorldSync::new(&world);
    let sky_of = |world: &World| world.get::<AtmosphereComponent>(sky).unwrap().clone();
    sync.poll(&mut world);
    assert_eq!(
        sky_of(&world).mie_anisotropy,
        0.1,
        "custom air keeps the authored sky"
    );
    world.get_mut::<VoxelTerrainComponent>(terrain).unwrap().air = VoxelAir::None;
    sync.poll(&mut world);
    assert!(!sky_of(&world).enabled, "a moon has no sky");
    assert_eq!(sky_of(&world).planet_radius_km, 1_737.4);
    world.get_mut::<VoxelTerrainComponent>(terrain).unwrap().air = VoxelAir::Thin;
    sync.poll(&mut world);
    let thin = sky_of(&world);
    assert!(
        thin.enabled
            && thin.rayleigh_scattering[2] < AtmosphereComponent::default().rayleigh_scattering[2]
    );
    // Hand edits to a derived sky go back to the air's.
    world
        .get_mut::<AtmosphereComponent>(sky)
        .unwrap()
        .mie_anisotropy = 0.2;
    sync.poll(&mut world);
    assert_eq!(sky_of(&world).mie_anisotropy, thin.mie_anisotropy);
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .air_density = 2.0;
    sync.poll(&mut world);
    assert!(
        (sky_of(&world).rayleigh_scattering[0] - thin.rayleigh_scattering[0] * 2.0).abs() < 1e-9
    );
    assert_eq!(sync_air_to_terrains(&mut world), 0);
}

#[test]
fn a_worlds_air_preset_makes_its_haze() {
    use fog_component::GlobalFogComponent;
    use voxel_component::VoxelAir;
    let mut world = World::new();
    let object = world.spawn();
    let mut planet = VoxelTerrainComponent::planet(6_371_000.0);
    planet.air = VoxelAir::Earthlike;
    let terrain = pulsar_world_registry::attach_value(&mut world, object, planet).unwrap();
    let fog =
        pulsar_world_registry::attach_value(&mut world, object, GlobalFogComponent::default())
            .unwrap();
    let mut sync = VoxelWorldSync::new(&world);
    let haze = |world: &World| world.get::<GlobalFogComponent>(fog).unwrap().clone();
    sync.poll(&mut world);
    let earth = helio_pass_volumetric_fog::GlobalFogRow {
        mode: 1,
        extinction: 0.0006,
        height_falloff: 0.004,
        anisotropy: 0.6,
        ..Default::default()
    }
    .around_planet([0.0; 3], 6_371_000.0);
    assert_eq!(
        haze(&world).medium.to_medium(),
        earth,
        "Earth's air, over the planet"
    );
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .air_haze = 2.0;
    sync.poll(&mut world);
    assert!((haze(&world).medium.extinction - 0.0012).abs() < 1e-9);
    world.get_mut::<VoxelTerrainComponent>(terrain).unwrap().air = VoxelAir::None;
    sync.poll(&mut world);
    assert!(!haze(&world).enabled, "a moon has no haze");
    // Custom air keeps an authored medium, placed on the ground.
    world.get_mut::<VoxelTerrainComponent>(terrain).unwrap().air = VoxelAir::Custom;
    world
        .get_mut::<GlobalFogComponent>(fog)
        .unwrap()
        .medium
        .albedo = [0.9, 0.8, 0.7];
    sync.poll(&mut world);
    assert_eq!(haze(&world).medium.albedo, [0.9, 0.8, 0.7]);
    assert_eq!(haze(&world).medium.planet_radius, 6_371_000.0);
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .shape = VoxelWorldShape::Plane;
    sync.poll(&mut world);
    assert_eq!(
        haze(&world).medium.planet_radius,
        0.0,
        "a plane's heights are along Y"
    );
    assert_eq!(sync_air_to_terrains(&mut world), 0);
}

/// Features are part of the ground: the entry carries their recipe,
/// and edits made before a change to them are dropped.
#[test]
fn features_are_part_of_the_ground() {
    use helio_pass_voxel_planet::feature_recipe::FeatureRecipe;
    let mut world = World::new();
    let object = world.spawn();
    let terrain = pulsar_world_registry::attach_value(
        &mut world,
        object,
        VoxelTerrainComponent::planet(1_000.0),
    )
    .unwrap();
    let source =
        |world: &World| entity_source(world, terrain, world.get(terrain).unwrap()).unwrap();
    let bare = source(&world).fingerprint();
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .features = voxel_component::feature_spawns(&FeatureRecipe::moon());
    let moon = source(&world);
    assert_eq!(moon.features, FeatureRecipe::moon());
    assert_ne!(moon.fingerprint(), bare);
    let mut sync = VoxelWorldSync::new(&world);
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .edits
        .push(helio_voxel_data::VoxelBrushEdit {
            center: [0.0, 1_000.0, 0.0],
            radius: 1.0,
            shape: helio_voxel_data::VoxelBrushShape::Sphere,
            op: helio_voxel_data::VoxelBrushOp::Remove,
            material: 0,
            height: Default::default(),
        });
    assert_eq!(sync.poll(&mut world), 0, "the journal adopts its ground");
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .features[0]
        .density = 0.9;
    assert_eq!(sync.poll(&mut world), 1, "other features are other ground");
}

#[test]
fn journals_are_checked_only_when_a_terrain_or_its_layers_change() {
    use voxel_component::{VoxelLayerKind, VoxelTerrainLayer, VoxelTerrainLayersComponent};
    let mut world = World::new();
    let object = world.spawn();
    let terrain = pulsar_world_registry::attach_value(
        &mut world,
        object,
        VoxelTerrainComponent::planet(1_000.0),
    )
    .unwrap();
    let layers = pulsar_world_registry::attach_value(
        &mut world,
        object,
        VoxelTerrainLayersComponent::default(),
    )
    .unwrap();
    let mut sync = VoxelWorldSync::new(&world);
    let edit = helio_voxel_data::VoxelBrushEdit {
        center: [0.0, 1_000.0, 0.0],
        radius: 1.0,
        shape: helio_voxel_data::VoxelBrushShape::Sphere,
        op: helio_voxel_data::VoxelBrushOp::Remove,
        material: 0,
        height: Default::default(),
    };
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .edits
        .push(edit);
    assert_eq!(sync.poll(&mut world), 0, "the journal adopts its ground");
    assert_eq!(sync.poll(&mut world), 0);
    // Another stack is other ground: its edits go.
    world
        .get_mut::<VoxelTerrainLayersComponent>(layers)
        .unwrap()
        .stack
        .layers
        .push(VoxelTerrainLayer::new(VoxelLayerKind::Craters));
    assert_eq!(sync.poll(&mut world), 1);
    assert_eq!(
        world
            .get::<VoxelTerrainComponent>(terrain)
            .unwrap()
            .edits
            .len(),
        0
    );
    // Dropping them wrote the terrain; the re-check finds nothing more.
    assert_eq!(sync.poll(&mut world), 0);
    world
        .get_mut::<VoxelTerrainComponent>(terrain)
        .unwrap()
        .edits
        .push(edit);
    assert_eq!(sync.poll(&mut world), 0, "edits on the current ground stay");
}
