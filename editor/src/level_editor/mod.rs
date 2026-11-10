//! Voxel worlds in the level editor, which the plugin's root crate registers
//! through the level editor's extension points (`ui_level_editor::extend`):
//! - the **Voxel Sculpt** tool mode ([`sculpt`]): strokes through the
//!   terrain component's `sculpt` method on the render thread, each stroke
//!   one undo step ([`journal`]) once its last samples landed (an
//!   interaction the editor polls and finishes before undo);
//! - the **Terrain** tool mode ([`terrain`]): a ready-made world in one
//!   click ([`add_world`], one undoable command) and the level's terrains;
//! - what a level save does first ([`save`]): fold old edits into the
//!   journals' bases, keep the renderer's chunks of the edits;
//! - choosing a terrain's generator attaches its settings
//!   (`voxel_component::world::ensure_generator_settings`, the component's
//!   method);
//! - framing a voxel world over its ground.

pub mod add_world;
pub mod journal;
pub mod save;
pub mod sculpt;
pub mod state;
pub mod terrain;

use glam::DVec3;
use pulsar_scenedb::{Entity, World};
use ui_level_editor::{LevelEditorState, ToolMode, ToolModeId};

/// The Voxel Sculpt mode's id.
pub const SCULPT_MODE: ToolModeId = ToolModeId("voxel_sculpt");
/// The Terrain mode's id.
pub const TERRAIN_MODE: ToolModeId = ToolModeId("terrain");

/// A new Voxel Sculpt mode (its tool mode registration's factory).
pub fn sculpt_mode() -> Box<dyn ToolMode> {
    Box::new(sculpt::VoxelSculptMode)
}

/// A new Terrain mode (its tool mode registration's factory).
pub fn terrain_mode() -> Box<dyn ToolMode> {
    Box::new(terrain::TerrainMode)
}

/// Whether a sculpt stroke is still waiting for its last samples (the
/// editor finishes it before undo).
pub fn stroke_in_progress(state: &LevelEditorState) -> bool {
    state::in_progress(state).is_some()
}

/// Finishes the stroke in progress as one undo step (`force`: without
/// waiting for its last samples).
pub fn finish_stroke(state: &mut LevelEditorState, force: bool) {
    state::VoxelStroke::finish(state, force);
}

/// What a level save does first: fold old edits into the journals' bases
/// and keep the renderer's chunks of the edits ([`save`]).
pub fn before_save(scene: &engine_backend::scene::SharedScene) {
    save::compact_voxel_journals(scene);
    save::keep_voxel_chunks(scene);
}

/// The terrain's class, whose `generator` property [`generator_changed`]
/// follows.
pub const TERRAIN_CLASS: &str = "VoxelTerrainComponent";

/// Choosing a terrain's generator attaches its settings
/// (`voxel_component::world::ensure_generator_settings`).
pub fn generator_changed(world: &mut World, owner: Entity, instance: Entity) {
    if let Err(error) = voxel_component::world::ensure_generator_settings(world, owner, instance) {
        tracing::warn!("{error}");
    }
}

/// Frames an object's first enabled voxel terrain over its ground (`None`
/// when the object holds none).
pub fn frame_terrain(
    world: &World,
    object: Entity,
    eye: DVec3,
    forward: DVec3,
) -> Option<Result<(DVec3, DVec3), String>> {
    let terrain = engine_backend::scene::attachments::enabled_components_of::<
        voxel_component::VoxelTerrainComponent,
    >(world, object)
    .first()
    .map(|(instance, _)| *instance)?;
    Some(
        voxel_component::world::terrain_world(world, terrain)
            .map(|planet| voxel_component::world::frame_view(&planet, eye, forward, 30.0))
            .map_err(|error| error.to_string()),
    )
}
