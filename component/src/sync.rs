//! A voxel terrain's dependents follow it: edits belong to the ground they
//! were made on, and a world's sky and haze are its air. [`VoxelWorldSync`]
//! keeps both in step from the scene's change journals; the terrains'
//! scene subsystem ([`crate::runtime::VoxelWorlds`]) polls it.

use pulsar_scene_model::attachments;
use pulsar_scenedb::{Entity, World};

use crate::world::{WorldSource, entity_source};
use crate::{VoxelTerrainComponent, VoxelTerrainLayersComponent, VoxelWorldShape};
use helio_pass_voxel_planet::grid::Shape;

/// Edits belong to the terrain they were made on: a terrain whose form,
/// generator, seed, settings or features changed drops the edits made on
/// the old one (and saves without them). Returns how many edits were
/// dropped.
pub fn sync_edit_journals(world: &mut World) -> usize {
    let stale: Vec<(Entity, u64)> = attachments::enabled_components::<VoxelTerrainComponent>(world)
        .filter(|(_, _, component)| !component.generator.id.is_empty())
        .filter_map(|(instance, _, component)| {
            let fingerprint = entity_source(world, instance, component)
                .ok()?
                .fingerprint();
            (component.edits.terrain() != fingerprint).then_some((instance, fingerprint))
        })
        .collect();
    let mut dropped = 0;
    for (entity, fingerprint) in stale {
        if let Some(mut terrain) = world.get_mut::<VoxelTerrainComponent>(entity) {
            let count = terrain.edits.made_on(fingerprint);
            if count > 0 {
                tracing::info!(
                    "Voxel terrain {}: its ground changed; {count} edits made on the old ground were dropped",
                    entity.bits()
                );
                terrain.source_revision = terrain.source_revision.wrapping_add(1);
            }
            dropped += count;
        }
    }
    dropped
}

/// The form of a terrain's world: its shape, radius and plane size in
/// metres, with its owner's scale.
fn form(source: &WorldSource) -> (VoxelWorldShape, f64, f64) {
    let shape = match source.recipe.shape {
        Shape::Sphere => VoxelWorldShape::Sphere,
        Shape::Plane => VoxelWorldShape::Plane,
        Shape::InfinitePlane => VoxelWorldShape::InfinitePlane,
    };
    (shape, source.recipe.radius_m, source.recipe.plane_size_m)
}

/// The method a ground's dependents answer: what sits on a voxel terrain's
/// object and follows its ground (a sky, a haze, whatever kind brings one)
/// has `follow_ground(sphere, planet_radius_m, air, density, tint_r,
/// tint_g, tint_b, haze) -> bool`, called by name: voxel names no kind.
pub const FOLLOW_GROUND: &str = "follow_ground";

/// A voxel world's sky and haze are its air: what sits on a voxel
/// terrain's object and answers [`FOLLOW_GROUND`] follows that terrain's
/// ground, so resizing the planet or turning it into a plane takes it
/// along, and a world whose `air` is a preset (not `Custom`) has it derive
/// the rest from that air (the sky's and the fog's own rules: an airless
/// moon has neither; the sky's clouds stay as authored). Returns how many
/// dependents changed.
pub fn sync_air_to_terrains(world: &mut World) -> usize {
    let kinds = pulsar_world_registry::components_with_method(FOLLOW_GROUND);
    if kinds.is_empty() {
        return 0;
    }
    let world_ref: &World = world;
    let calls: Vec<(Entity, pulsar_scenedb::ComponentId, Ground)> = terrain_forms(world_ref)
        .flat_map(|(owner, terrain, (shape, planet_radius, _))| {
            let ground = Ground {
                sphere: shape == VoxelWorldShape::Sphere,
                planet_radius_m: planet_radius,
                air: format!("{:?}", terrain.air),
                density: f64::from(terrain.air_density),
                tint: terrain.air_tint.map(f64::from),
                haze: f64::from(terrain.air_haze),
            };
            let kinds = &kinds;
            attachments::instances(world_ref, owner)
                .into_iter()
                .filter(|instance| attachments::is_enabled(world_ref, *instance))
                .flat_map(move |instance| {
                    let ground = ground.clone();
                    kinds
                        .iter()
                        .filter(move |cid| world_ref.has_component(instance, **cid))
                        .map(move |cid| (instance, *cid, ground.clone()))
                })
        })
        .collect();
    let mut changed = 0;
    for (instance, cid, ground) in calls {
        let mut args: Vec<Box<dyn std::any::Any>> = vec![
            Box::new(ground.sphere),
            Box::new(ground.planet_radius_m),
            Box::new(ground.air),
            Box::new(ground.density),
            Box::new(ground.tint[0]),
            Box::new(ground.tint[1]),
            Box::new(ground.tint[2]),
            Box::new(ground.haze),
        ];
        match world.call_component_method(instance, cid, FOLLOW_GROUND, &mut args) {
            Ok(Some(answer)) => {
                if answer.downcast::<bool>().is_ok_and(|moved| *moved) {
                    changed += 1;
                }
            }
            Ok(None) => {}
            Err(error) => tracing::warn!("{FOLLOW_GROUND} on {}: {error:?}", instance.bits()),
        }
    }
    changed
}

/// A terrain's ground and air, as [`FOLLOW_GROUND`] takes them.
#[derive(Clone)]
struct Ground {
    sphere: bool,
    planet_radius_m: f64,
    air: String,
    density: f64,
    tint: [f64; 3],
    haze: f64,
}

/// The enabled terrains with their owner and form.
fn terrain_forms(
    world: &World,
) -> impl Iterator<Item = (Entity, &VoxelTerrainComponent, (VoxelWorldShape, f64, f64))> {
    attachments::enabled_components::<VoxelTerrainComponent>(world).filter_map(
        move |(instance, owner, component)| {
            let source = entity_source(world, instance, component).ok()?;
            Some((owner, component, form(&source)))
        },
    )
}

/// Keeps voxel worlds' dependents in step with them when a terrain, its
/// layer settings, an atmosphere or a fog changed (change cursors, so
/// frames that change none cost nothing): edits made on ground a terrain no
/// longer has are dropped ([`sync_edit_journals`]) and skies and hazes
/// follow their terrain ([`sync_air_to_terrains`]).
pub struct VoxelWorldSync {
    cursors: Vec<pulsar_scenedb::ChangeCursor>,
    /// The world revision at the last poll; a smaller one means the world
    /// was replaced, and the cursors with it.
    revision: u64,
    synced: bool,
    scratch: Vec<pulsar_scenedb::ComponentChange>,
}

impl VoxelWorldSync {
    pub fn new(world: &World) -> Self {
        Self {
            cursors: [
                world.open_change_cursor::<VoxelTerrainComponent>(),
                world.open_change_cursor::<VoxelTerrainLayersComponent>(),
            ]
            .into_iter()
            .chain(
                pulsar_world_registry::components_with_method(FOLLOW_GROUND)
                    .into_iter()
                    .map(|cid| world.open_change_cursor_id(cid)),
            )
            .collect(),
            revision: world.revision(),
            synced: false,
            scratch: Vec::new(),
        }
    }

    /// Drops edits made on ground a terrain no longer has and moves skies
    /// and hazes onto their terrain, when anything they depend on changed
    /// since the last poll. Returns how many edits were dropped.
    pub fn poll(&mut self, world: &mut World) -> usize {
        if world.revision() < self.revision {
            *self = Self::new(world);
        }
        self.revision = world.revision();
        let mut changed = !self.synced;
        for cursor in &mut self.cursors {
            self.scratch.clear();
            changed |= world.read_changes(cursor, &mut self.scratch)
                == pulsar_scenedb::ChangeRead::Overflowed
                || !self.scratch.is_empty();
        }
        if !changed {
            return 0;
        }
        self.synced = true;
        let dropped = sync_edit_journals(world);
        sync_air_to_terrains(world);
        // Its own writes are not changes to answer next poll (an overflow
        // leaves the cursor at the newest entry all the same).
        for cursor in &mut self.cursors {
            self.scratch.clear();
            let _ = world.read_changes(cursor, &mut self.scratch);
        }
        dropped
    }
}
