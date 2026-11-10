//! Ready-made voxel worlds, added to a level in one step: an Earth-like
//! planet, the Moon, a desert planet and a flat block world.
//!
//! A preset is an example project's level (`assets/examples/<name>`): Helio's
//! voxel_flight worlds as Pulsar levels, which the `voxel_example_levels`
//! test checks against voxel_flight, so a new world and its example are one
//! description. [`add_voxel_world`] puts a preset's world into a level: the
//! terrain (layers, features, look), sky and clouds, sun, haze, ground cover
//! and wind, the post-process look, and the camera over the ground.
//!
//! A level holds one sky, one wind and one directional light
//! ([`level_rules`](engine_backend::scene::level_rules)). The world's sky, haze, wind and
//! ground cover replace the level's; its sun and post-process look update the
//! level's directional light and unbound post-process volume in place; its
//! terrain replaces a voxel terrain of the level on that terrain's own
//! object. Objects left with nothing on them are removed.

use engine_backend::scene::{
    EditorCamera, Name, ObjectType, RuntimeLevel, SceneWorldExt, SpawnObject, Transform, Visibility,
};
use pulsar_scene_model::{NewInstance, attachments};
use pulsar_scenedb::{Entity, World};

use crate::others::{self, ATMOSPHERE, FOLIAGE, GLOBAL_FOG, POST_VOLUME, WIND};
use crate::{VoxelTerrainComponent, VoxelTerrainLayersComponent};

/// A ready-made voxel world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VoxelWorldPreset {
    /// Earth: woods, meadows, clouds and haze.
    Earth,
    /// The Moon: craters and boulders under a black sky.
    Moon,
    /// A desert planet: dunes, rocks and shrubs in dusty air.
    Desert,
    /// Flat grassland without edges under Earth's sky, to build on.
    Blocks,
}

impl VoxelWorldPreset {
    pub const ALL: [Self; 4] = [Self::Earth, Self::Moon, Self::Desert, Self::Blocks];

    /// The example project with this world (`assets/examples/<name>`).
    pub fn example(self) -> &'static str {
        match self {
            Self::Earth => "voxel_planet",
            Self::Moon => "voxel_moon",
            Self::Desert => "voxel_desert",
            Self::Blocks => "voxel_blocks",
        }
    }

    /// The example's level file.
    fn level(self) -> &'static str {
        match self {
            Self::Earth => {
                include_str!("../../../../../assets/examples/voxel_planet/scene/default.level")
            }
            Self::Moon => {
                include_str!("../../../../../assets/examples/voxel_moon/scene/default.level")
            }
            Self::Desert => {
                include_str!("../../../../../assets/examples/voxel_desert/scene/default.level")
            }
            Self::Blocks => {
                include_str!("../../../../../assets/examples/voxel_blocks/scene/default.level")
            }
        }
    }

    /// The world as a level of its own.
    pub fn template(self) -> Result<RuntimeLevel, String> {
        let file = serde_json::from_str(self.level())
            .map_err(|error| format!("{} level: {error}", self.example()))?;
        RuntimeLevel::from_scene_file_with_classes(file, &pulsar_class::ClassRegistry::default())
            .map_err(|error| format!("{} level: {error}", self.example()))
    }
}

/// What [`add_voxel_world`] did.
#[derive(Clone, Debug, PartialEq)]
pub struct VoxelWorldSetup {
    /// The terrain's object.
    pub terrain: String,
    /// Objects it added.
    pub added: Vec<String>,
    /// Existing objects it changed or removed.
    pub changed: Vec<String>,
    /// The world's camera over its ground.
    pub camera: Option<EditorCamera>,
}

/// What of a level a new world takes over.
struct Plan {
    /// A voxel terrain's object (the first by id): the new terrain goes on
    /// it.
    terrain_owner: Option<Entity>,
    /// The directional light, updated in place.
    sun: Option<Entity>,
    /// The first enabled unbound post-process volume, updated in place.
    look: Option<Entity>,
    /// Instances the world replaces: the sky, hazes, wind, ground cover and
    /// the terrain with its layers.
    replaced: Vec<Entity>,
}

impl Plan {
    fn of(world: &World) -> Self {
        let terrain_owner = world
            .query::<&VoxelTerrainComponent>()
            .filter_map(|(instance, _)| attachments::owner_of(world, instance))
            .min_by_key(|owner| world.stable_id_of(*owner).map(str::to_string));
        let mut replaced = others::instances_of(world, ATMOSPHERE);
        replaced.extend(others::instances_of(world, WIND));
        replaced.extend(others::instances_of(world, GLOBAL_FOG));
        // Ground cover belongs to the voxel world it grows on.
        replaced.extend(
            others::instances_of(world, FOLIAGE)
                .into_iter()
                .filter(|foliage| {
                    others::variant(world, *foliage, FOLIAGE, "kind") == Some("Block")
                }),
        );
        if let Some(owner) = terrain_owner {
            replaced.extend(
                attachments::instances(world, owner)
                    .into_iter()
                    .filter(|instance| {
                        world.get::<VoxelTerrainComponent>(*instance).is_some()
                            || world
                                .get::<VoxelTerrainLayersComponent>(*instance)
                                .is_some()
                    }),
            );
        }
        Self {
            terrain_owner,
            sun: others::suns(world).first().copied(),
            look: others::instances_of(world, POST_VOLUME)
                .into_iter()
                .find(|instance| {
                    attachments::is_enabled(world, *instance) && is_look(world, *instance)
                }),
            replaced,
        }
    }

    /// The existing objects the world changes.
    fn objects(&self, world: &World) -> Vec<Entity> {
        let mut objects: Vec<Entity> = self
            .replaced
            .iter()
            .chain(self.sun.iter())
            .chain(self.look.iter())
            .filter_map(|instance| attachments::owner_of(world, *instance))
            .chain(self.terrain_owner)
            .collect();
        objects.sort_by_key(|object| object.bits());
        objects.dedup();
        objects
    }
}

/// Whether a post-process volume is a level's look: enabled and unbound.
fn is_look(world: &World, volume: Entity) -> bool {
    others::property_is(
        world,
        volume,
        POST_VOLUME,
        "enabled",
        serde_json::json!(true),
    ) && others::property_is(
        world,
        volume,
        POST_VOLUME,
        "unbound",
        serde_json::json!(true),
    )
}

/// The ids of the existing objects [`add_voxel_world`] would change or
/// remove in `world` (an editor's undo scope).
pub fn objects_changed_by(world: &World) -> Vec<String> {
    let plan = Plan::of(world);
    object_ids(world, &plan.objects(world))
}

/// The stable ids of `objects` (those that have one).
pub fn object_ids(world: &World, objects: &[Entity]) -> Vec<String> {
    objects
        .iter()
        .filter_map(|object| world.stable_id_of(*object).map(str::to_string))
        .collect()
}

/// Add `preset`'s world to the level in `world` (see the module doc).
pub fn add_voxel_world(
    world: &mut World,
    preset: VoxelWorldPreset,
) -> Result<VoxelWorldSetup, String> {
    let template = preset.template()?;
    let scene = template.scene();
    let scene = scene.read();
    let source = &scene.world;

    let plan = Plan::of(world);
    let touched = plan.objects(world);
    let changed = object_ids(world, &touched);
    for instance in &plan.replaced {
        attachments::detach(world, *instance);
    }

    let mut added = Vec::new();
    let mut terrain = None;
    for object in source.children_of(None) {
        let instances = attachments::instances(source, object);
        let has_terrain = instances
            .iter()
            .any(|instance| source.get::<VoxelTerrainComponent>(*instance).is_some());
        let class =
            |instance: Entity| attachments::meta(source, instance).map(|m| m.class_name.as_str());
        let light = instances.iter().find(|instance| {
            class(**instance) == Some(others::LIGHT) && others::is_sun(source, **instance)
        });
        let look = instances.iter().find(|instance| {
            class(**instance) == Some(POST_VOLUME)
                && others::property_is(
                    source,
                    **instance,
                    POST_VOLUME,
                    "unbound",
                    serde_json::json!(true),
                )
        });
        match (light, plan.sun, look, plan.look) {
            (Some(light), Some(sun), _, _) => {
                set_value(source, *light, world, sun)?;
                // The sun shines along its object's rotation.
                let rotation = source.get::<Transform>(object).map(|t| t.rotation);
                if let (Some(owner), Some(rotation)) = (attachments::owner_of(world, sun), rotation)
                {
                    if let Some(mut transform) = world.get_mut::<Transform>(owner) {
                        transform.rotation = rotation;
                    }
                }
                continue;
            }
            (_, _, Some(look), Some(existing)) => {
                set_value(source, *look, world, existing)?;
                continue;
            }
            _ => {}
        }
        let owner = match plan.terrain_owner.filter(|_| has_terrain) {
            Some(owner) => owner,
            None => {
                let owner = copy_object(source, object, world)?;
                added.extend(world.stable_id_of(owner).map(str::to_string));
                owner
            }
        };
        if has_terrain {
            terrain = Some(owner);
        }
        for instance in instances {
            copy_instance(source, instance, world, owner)?;
        }
    }

    // Objects the replaced instances left bare.
    for object in touched {
        let bare = world.is_alive(object)
            && attachments::instances(world, object).is_empty()
            && world.children_of(Some(object)).is_empty()
            && world
                .get::<ObjectType>(object)
                .is_none_or(|kind| *kind == ObjectType::Empty);
        if bare {
            world.despawn_tree(object);
        }
    }

    let terrain =
        terrain.ok_or_else(|| format!("{} level has no voxel terrain", preset.example()))?;
    // The world sits on its terrain's object: so does its camera.
    let origin = world
        .get::<Transform>(terrain)
        .map_or([0.0; 3], |t| t.position);
    let camera = template.editor_camera().map(|camera| EditorCamera {
        position: [0, 1, 2].map(|axis| camera.position[axis] + origin[axis]),
        ..camera
    });
    Ok(VoxelWorldSetup {
        terrain: world
            .stable_id_of(terrain)
            .map(str::to_string)
            .unwrap_or_default(),
        added,
        changed,
        camera,
    })
}

/// A copy of `source`'s top-level `object` (name, transform, type,
/// visibility; no components) in `world`.
fn copy_object(source: &World, object: Entity, world: &mut World) -> Result<Entity, String> {
    let spec = SpawnObject {
        stable_id: None,
        name: source
            .get::<Name>(object)
            .map(|name| name.0.clone())
            .unwrap_or_default(),
        parent: None,
        transform: source.get::<Transform>(object).copied().unwrap_or_default(),
        visibility: source
            .get::<Visibility>(object)
            .cloned()
            .unwrap_or_default(),
        object_type: source
            .get::<ObjectType>(object)
            .cloned()
            .unwrap_or(ObjectType::Empty),
    };
    world.spawn_object(spec).map_err(|error| error.to_string())
}

/// Attach a copy of `source`'s `instance` to `owner` in `world`.
fn copy_instance(
    source: &World,
    instance: Entity,
    world: &mut World,
    owner: Entity,
) -> Result<(), String> {
    use pulsar_world_registry::{ComponentPayload, InstanceValue};
    let snapshot = pulsar_world_registry::snapshot_instance(source, instance)
        .ok_or("a preset component cannot be copied")?;
    let class = snapshot.meta.class_name;
    let InstanceValue::Value(value) = snapshot.value else {
        return Err(format!("{class} is not a class of this build"));
    };
    let spec = NewInstance {
        enabled: snapshot.enabled,
        ..NewInstance::new(class)
    };
    pulsar_world_registry::attach_component(world, owner, spec, ComponentPayload::Value(value))
        .map(|_| ())
        .map_err(|error| error.to_string())
}

/// Give `world`'s `instance` the value of `source`'s `from` (same class).
fn set_value(
    source: &World,
    from: Entity,
    world: &mut World,
    instance: Entity,
) -> Result<(), String> {
    let snapshot = pulsar_world_registry::snapshot_instance(source, from)
        .ok_or("a preset component cannot be copied")?;
    pulsar_world_registry::set_instance_value(world, instance, snapshot.value)
        .map_err(|error| error.to_string())
}
