//! Voxel worlds (`pulsar.voxel`): the editor crate, over the plugin's
//! component crate.
//!
//! - The voxel terrain's inspector editors: the generator picker
//!   ([`generator_editor`]) and the layer stack ([`stack_editor`]), each a
//!   property editor hint of its type (registered with the type).
//! - The level editor UI ([`level_editor`]): the sculpt and terrain tool
//!   modes, their panels, strokes and their undo, the save hooks, the
//!   generator follow-up and the terrain's camera framing, acting through
//!   the component's methods.
//!
//! The level editor entries register nothing themselves: the plugin's root
//! crate (`voxel_plugin`, with its `editor` feature) submits them through
//! the level editor's extension points.

// The level editor UI's text (`locales/`).
rust_i18n::i18n!("locales", fallback = "en");

pub mod level_editor;

// The unit tests attach terrains to worlds, which takes a registered class.
// A test build of this crate does not link the root crate (the root links
// this crate under `editor`), so it registers the class itself; every other
// build gets it from the root only.
#[cfg(test)]
pulsar_world_registry::submit_world_component!(voxel_component::VoxelTerrainComponent);

pub mod generator_editor;
pub mod stack_editor;
