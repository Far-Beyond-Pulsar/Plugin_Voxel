//! Voxel worlds (`pulsar.voxel`): the component crate of the plugin that
//! owns the voxel terrain.
//!
//! - Components ([`component`]): `VoxelTerrainComponent` and
//!   `VoxelTerrainLayersComponent`, their scripting methods (blocks,
//!   brushes, ground queries, sculpting) and block events.
//! - Their runtime ([`runtime`]): one world per terrain, built off the game
//!   thread; the scene subsystem that keeps journals on their ground, skies
//!   on their air ([`sync`]) and publishes the worlds to the renderer
//!   (`helio_pass_voxel_planet::engine::VOXEL_FRAME`).
//! - File formats: the edit journal and kept chunks (in the component),
//!   ready-made worlds ([`presets`]).
//! - The plugin's Helio feature ([`render_features`]:
//!   `helio_pass_voxel_planet::VoxelFeature`) and its settings
//!   ([`configure_renderer`], `VOXEL_RESIDENCY`), and the terrain's
//!   material names for ground-material lists ([`material_name`], foliage's
//!   ground).
//!
//! This crate registers nothing itself: the plugin's root crate
//! (`voxel_plugin`) submits the component classes, the feature, its
//! settings, the material names, the scene subsystem and the frame payload,
//! in one place.

pub mod component;
mod component_runtime;
pub mod features;
pub mod others;
pub mod presets;
pub mod runtime;
pub mod sculpt;
pub mod sync;
pub mod world;

pub use component::*;
pub use features::*;
/// The terrain's data formats: the edit journal, brushes, kept chunks.
pub use helio_voxel_data as data;

use helio_pass_voxel_planet::terrain::material;

/// The voxel feature: the chunk terrain at the G-buffer, its sun, GI tracer,
/// foliage ground and water terrain.
pub fn render_features() -> helio_feature_api::FeatureSet {
    helio_feature_api::FeatureSet::new().with(helio_pass_voxel_planet::VoxelFeature)
}

/// The feature's settings from the project's voxel settings.
pub fn configure_renderer(config: helio::RendererConfig) -> helio::RendererConfig {
    use engine_backend::subsystems::render::helio_renderer::project_settings;
    config.with_feature(helio_pass_voxel_planet::VoxelSettings {
        residency: helio_pass_voxel_planet::chunks::ResidencyConfig::from_settings(
            &project_settings::resolved(&helio_pass_voxel_planet::settings::VOXEL_RESIDENCY),
        ),
    })
}

/// A terrain material's name (ids start at 1).
pub fn material_name(id: u32) -> Option<&'static str> {
    let index = (id as usize).checked_sub(1)?;
    material::NAMES.get(index).copied()
}
