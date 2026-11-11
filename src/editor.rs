//! What the voxel plugin registers for the editor (the `editor` feature):
//! its tool modes, the sculpt stroke, the save hook, the generator
//! follow-up, the terrain's camera framing and the inspector's editors of
//! its feature lists.

use ui_level_editor::extend::{
    CameraFraming, Interaction, PropertyFollowUp, SaveHook, ToolModeRegistration,
};
use voxel_editor::level_editor;

// The terrain's feature lists edit as reflected lists (one card per entry).
// A type attachment (no origin): it stays with the types it hints.
ui_common::register_reflected_list_editor!(voxel_component::VoxelFeatureSpawn);
ui_common::register_reflected_list_editor!(voxel_component::VoxelFeatureGrowth);

/// Everything the voxel plugin registers for the editor.
pub(crate) fn build(p: &mut pulsar_plugins::PluginBuilder) {
    // ── Tool modes ──────────────────────────────────────────────────────────

    // Voxel Sculpt: strokes through the terrain component's `sculpt` method.
    p.add(ToolModeRegistration {
        origin: p.origin(),
        id: "voxel.sculpt_mode",
        order: 20,
        make: level_editor::sculpt_mode,
    });

    // Terrain: a ready-made world in one click, and the level's terrains.
    p.add(ToolModeRegistration {
        origin: p.origin(),
        id: "voxel.terrain_mode",
        order: 25,
        make: level_editor::terrain_mode,
    });

    // ── Editing ─────────────────────────────────────────────────────────────

    // A sculpt stroke is one undo step once its last samples landed: the
    // editor polls it and finishes it before undo.
    p.add(Interaction {
        origin: p.origin(),
        id: "voxel.sculpt_stroke",
        in_progress: level_editor::stroke_in_progress,
        finish: level_editor::finish_stroke,
    });

    // A level save first folds old edits into the journals' bases and keeps
    // the renderer's chunks of the edits.
    p.add(SaveHook {
        origin: p.origin(),
        id: "voxel.journals",
        before_save: level_editor::before_save,
    });

    // Choosing a terrain's generator attaches its settings.
    p.add(PropertyFollowUp {
        origin: p.origin(),
        class_name: level_editor::TERRAIN_CLASS,
        property: "generator",
        follow: level_editor::generator_changed,
    });

    // ── Viewport ────────────────────────────────────────────────────────────

    // Framing a voxel world over its ground.
    p.add(CameraFraming {
        origin: p.origin(),
        id: "voxel.terrain",
        frame: level_editor::frame_terrain,
    });
}
