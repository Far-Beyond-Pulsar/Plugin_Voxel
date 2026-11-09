//! What the voxel plugin registers for the editor (the `editor` feature):
//! its tool modes, the sculpt stroke, the save hook, the generator
//! follow-up, the terrain's camera framing and the inspector's editors of
//! its feature lists.

use ui_level_editor::extend::{
    CameraFraming, Interaction, PropertyFollowUp, SaveHook, ToolModeRegistration,
};
use voxel_editor::level_editor;

// ── Tool modes ──────────────────────────────────────────────────────────────

// Voxel Sculpt: strokes through the terrain component's `sculpt` method.
inventory::submit! {
    ToolModeRegistration {
        origin: pulsar_plugins::origin!(),
        id: "voxel.sculpt_mode",
        order: 20,
        make: level_editor::sculpt_mode,
    }
}

// Terrain: a ready-made world in one click, and the level's terrains.
inventory::submit! {
    ToolModeRegistration {
        origin: pulsar_plugins::origin!(),
        id: "voxel.terrain_mode",
        order: 25,
        make: level_editor::terrain_mode,
    }
}

// ── Editing ─────────────────────────────────────────────────────────────────

// A sculpt stroke is one undo step once its last samples landed: the editor
// polls it and finishes it before undo.
inventory::submit! {
    Interaction {
        origin: pulsar_plugins::origin!(),
        id: "voxel.sculpt_stroke",
        in_progress: level_editor::stroke_in_progress,
        finish: level_editor::finish_stroke,
    }
}

// A level save first folds old edits into the journals' bases and keeps the
// renderer's chunks of the edits.
inventory::submit! {
    SaveHook {
        origin: pulsar_plugins::origin!(),
        id: "voxel.journals",
        before_save: level_editor::before_save,
    }
}

// Choosing a terrain's generator attaches its settings.
inventory::submit! {
    PropertyFollowUp {
        origin: pulsar_plugins::origin!(),
        class_name: level_editor::TERRAIN_CLASS,
        property: "generator",
        follow: level_editor::generator_changed,
    }
}

// ── Viewport ────────────────────────────────────────────────────────────────

// Framing a voxel world over its ground.
inventory::submit! {
    CameraFraming {
        origin: pulsar_plugins::origin!(),
        id: "voxel.terrain",
        frame: level_editor::frame_terrain,
    }
}

// ── Inspector editors ───────────────────────────────────────────────────────

// The terrain's feature lists edit as reflected lists (one card per entry).
ui_common::register_reflected_list_editor!(voxel_component::VoxelFeatureSpawn);
ui_common::register_reflected_list_editor!(voxel_component::VoxelFeatureGrowth);
