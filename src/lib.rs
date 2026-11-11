//! Voxel Worlds (`pulsar.voxel`): the plugin's root crate, the one place the
//! plugin registers everything it contributes.
//!
//! The plugin's other crates hold the code: `helio-voxel-data` (the data
//! formats other kinds and tools read), `helio-pass-voxel-planet` (the Helio
//! feature: the chunk terrain, its sun, GI tracer, foliage ground and water
//! terrain), `voxel_component` (the terrain components, their methods and
//! runtime, the file formats and ready-made worlds) and `voxel_editor` (the
//! terrain's inspector editors and the level editor's voxel tools). None of
//! them registers a contribution; this crate imports them and registers their
//! registrations, so this file and the two modules below are the whole list
//! of what the plugin brings:
//!
//! - [`runtime`]: what a game links. The component classes, the Helio
//!   feature and its settings, the terrain's material names for
//!   ground-material lists, the scene subsystem that steps the terrains'
//!   worlds and the frame payload that hands them to the voxel pass.
//! - `editor` (with the `editor` feature): what the editor adds. The Voxel
//!   Sculpt and Terrain tool modes, the sculpt stroke as an interaction, the
//!   save hook, the generator follow-up and the terrain's camera framing.
//!
//! Type attachments stay with their types: the terrain's component runtime
//! (its block events), the features list's reflected type and the inspector
//! editors' property hints are registered where the types are defined.
//!
//! Every registration names this crate as its origin
//! (`pulsar_plugins::origin!()`), so the plugin gate skips all of them while
//! a project disables the plugin, and the plugin catalog lists them under
//! `pulsar.voxel`. Other plugins never depend on this crate (only on
//! `helio-voxel-data` or by class name); hosts reach it through the
//! installed set or a generated game, a game without the `editor` feature.

// The plugin's descriptor (`pulsar-plugin.toml`): linking this crate installs
// the plugin; a project enables it or not (`pulsar_plugins`).
pulsar_plugins::installed!("pulsar-plugin.toml", build);

pub use voxel_component;
#[cfg(feature = "editor")]
pub use voxel_editor;

pub mod runtime;

#[cfg(feature = "editor")]
pub mod editor;

/// Everything the plugin registers (runtime always, editor with the `editor`
/// feature), through the builder the kernel hands the plugin's root.
pub fn build(p: &mut pulsar_plugins::PluginBuilder) {
    runtime::build(p);
    #[cfg(feature = "editor")]
    editor::build(p);
}
