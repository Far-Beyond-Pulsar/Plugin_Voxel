//! Ready-made worlds put into a level (`voxel_component::presets`): an
//! empty level gets the example level, a level's sky, sun, wind, haze, look
//! and terrain are taken over, and a reused terrain object keeps its place.
//! The world's classes resolve by name through the world registry, so the
//! plugin and the kinds the world composes are linked through their roots.

// The plugin as a game links it: its root, with every registration.
use voxel_plugin as _;
// The kinds a voxel world composes register through their roots.
use fog_plugin as _;
use foliage_plugin as _;
use lights_plugin as _;
use post_plugin as _;
use sky_plugin as _;

use engine_backend::scene::{Name, SceneWorldExt, SpawnObject, Transform};
use foliage_component::FoliageComponent;
use lights_component::LightComponent;
use pulsar_scene_model::attachments;
use pulsar_scenedb::World;
use sky_component::AtmosphereComponent;
use voxel_component::presets::{VoxelWorldPreset, add_voxel_world, object_ids, objects_changed_by};
use voxel_component::{VoxelTerrainComponent, VoxelTerrainLayersComponent};

fn level_of(world: &World) -> Vec<(String, Vec<String>)> {
    let mut objects: Vec<_> = world
        .children_of(None)
        .into_iter()
        .map(|object| {
            let mut classes: Vec<String> = attachments::instances(world, object)
                .into_iter()
                .filter_map(|instance| {
                    attachments::meta(world, instance).map(|m| m.class_name.clone())
                })
                .collect();
            classes.sort();
            (
                world
                    .get::<Name>(object)
                    .map(|n| n.0.clone())
                    .unwrap_or_default(),
                classes,
            )
        })
        .collect();
    objects.sort();
    objects
}

/// Every preset in an empty level is its example level, camera included.
#[test]
fn a_preset_in_an_empty_level_is_its_example() {
    for preset in VoxelWorldPreset::ALL {
        let mut world = World::new();
        let setup = add_voxel_world(&mut world, preset).unwrap();
        let template = preset.template().unwrap();
        let example = template.scene();
        assert_eq!(
            level_of(&world),
            level_of(&example.read().world),
            "{preset:?}"
        );
        assert_eq!(setup.camera, template.editor_camera(), "{preset:?}");
        assert!(setup.changed.is_empty());
        assert_eq!(setup.added.len(), world.children_of(None).len());
        let terrain = world.entity_for(&setup.terrain).unwrap();
        assert_eq!(
            attachments::enabled_components_of::<VoxelTerrainComponent>(&world, terrain).len(),
            1
        );
        assert!(sky_component::atmospheres(&world).len() <= 1);
        assert!(foliage_component::winds(&world).len() <= 1);
        assert_eq!(lights_component::directional_lights(&world).len(), 1);
    }
}

/// A level's sky, sun, wind, haze, look and terrain are taken over, not
/// doubled: the level rules hold and a second setup changes the first
/// world rather than adding another.
#[test]
fn a_second_world_takes_over_the_first() {
    let mut world = World::new();
    let first = add_voxel_world(&mut world, VoxelWorldPreset::Earth).unwrap();
    let objects = world.children_of(None).len();
    let moon = add_voxel_world(&mut world, VoxelWorldPreset::Moon).unwrap();
    assert_eq!(moon.terrain, first.terrain, "the terrain's object is kept");
    assert!(moon.added.is_empty(), "{moon:?}");
    // The meadow went with Earth.
    assert_eq!(world.children_of(None).len(), objects - 1);
    assert_eq!(world.query::<&FoliageComponent>().count(), 0);
    assert_eq!(sky_component::atmospheres(&world).len(), 1);
    assert_eq!(foliage_component::winds(&world).len(), 0);
    assert_eq!(lights_component::directional_lights(&world).len(), 1);
    assert_eq!(world.query::<&VoxelTerrainComponent>().count(), 1);
    assert_eq!(world.query::<&VoxelTerrainLayersComponent>().count(), 1);
    let terrain = world.query::<&VoxelTerrainComponent>().next().unwrap().1;
    assert_eq!(terrain.planet_radius, 1_737_400.0);
    let mut changed = moon.changed.clone();
    changed.sort();
    let mut before = first.added.clone();
    before.sort();
    assert_eq!(changed, before, "every Earth object is taken over");
    // An object the world does not know stays as it is.
    let mut sky_only = World::new();
    let lamp = sky_only.spawn_object(SpawnObject::new("Lamp")).unwrap();
    let mut point = LightComponent::default();
    point.general.light_type = lights_component::LightType::Point;
    pulsar_world_registry::attach_value(&mut sky_only, lamp, point).unwrap();
    let sky = sky_only.spawn_object(SpawnObject::new("Sky")).unwrap();
    pulsar_world_registry::attach_value(&mut sky_only, sky, AtmosphereComponent::default())
        .unwrap();
    assert_eq!(objects_changed_by(&sky_only), object_ids(&sky_only, &[sky]));
    add_voxel_world(&mut sky_only, VoxelWorldPreset::Earth).unwrap();
    assert!(!sky_only.is_alive(sky), "the bare sky object is removed");
    assert!(sky_only.is_alive(lamp));
    assert_eq!(sky_component::atmospheres(&sky_only).len(), 1);
}

/// A world replacing a terrain on a moved object sits, with its camera,
/// where that object is.
#[test]
fn a_reused_terrain_object_keeps_its_place() {
    let mut world = World::new();
    let owner = world
        .spawn_object(SpawnObject::new("Ground").with_transform(Transform {
            position: [100.0, 0.0, 0.0],
            ..Transform::default()
        }))
        .unwrap();
    pulsar_world_registry::attach_value(&mut world, owner, VoxelTerrainComponent::plane(512.0))
        .unwrap();
    let setup = add_voxel_world(&mut world, VoxelWorldPreset::Blocks).unwrap();
    assert_eq!(world.entity_for(&setup.terrain), Some(owner));
    assert_eq!(setup.changed, object_ids(&world, &[owner]));
    let example = VoxelWorldPreset::Blocks
        .template()
        .unwrap()
        .editor_camera()
        .unwrap();
    assert_eq!(
        setup.camera.unwrap().position[0],
        example.position[0] + 100.0
    );
}
