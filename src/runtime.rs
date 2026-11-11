//! What the voxel plugin registers for a game (and the editor): every
//! runtime contribution, grouped by the engine seam it plugs into.

use engine_backend::scene::{ClassSpec, ComponentsBuilder};
use engine_backend::subsystems::render::RenderFeatureRegistration;
use helio_component::GroundMaterialNames;
use helio_pass_voxel_planet::engine::VOXEL_FRAME;
use helio_pass_voxel_planet::terrain::material;
use pulsar_settings::project::rendering::SettingsGroupRegistration;
use pulsar_world_registry::{ScenePayloadRegistration, SceneSubsystemRegistration};
use voxel_component::runtime::VoxelWorlds;
use voxel_component::{VoxelTerrainComponent, VoxelTerrainLayersComponent};

/// Everything the voxel plugin registers for a game.
pub(crate) fn build(p: &mut pulsar_plugins::PluginBuilder) {
    // ── Component classes ───────────────────────────────────────────────────

    p.components()
        .class::<VoxelTerrainComponent>(ClassSpec::default())
        .class::<VoxelTerrainLayersComponent>(ClassSpec::default());

    // ── Rendering ───────────────────────────────────────────────────────────

    // The voxel feature (the chunk terrain at the G-buffer, its sun, GI
    // tracer, foliage ground and water terrain), configured from the
    // project's voxel settings.
    p.add(RenderFeatureRegistration {
        origin: p.origin(),
        features: voxel_component::render_features,
        configure: voxel_component::configure_renderer,
    });

    // ── Settings ────────────────────────────────────────────────────────────

    // The project's voxel residency settings.
    p.add(SettingsGroupRegistration {
        origin: p.origin(),
        group: &helio_pass_voxel_planet::settings::VOXEL_RESIDENCY,
    });

    // ── Materials ───────────────────────────────────────────────────────────

    // The terrain's material names in every ground-material list (a feature
    // spawn's surfaces, foliage's ground).
    p.add(GroundMaterialNames {
        origin: p.origin(),
        name: voxel_component::material_name,
        id: material::from_name,
    });

    // ── Scene ───────────────────────────────────────────────────────────────

    // The subsystem that keeps journals on their ground and skies on their
    // air, and steps every terrain's world.
    p.add(SceneSubsystemRegistration {
        origin: p.origin(),
        name: VoxelWorlds::NAME,
        install: |scene| scene.register_subsystem(VoxelWorlds::default()),
    });

    // The scene's voxel worlds, handed to the voxel pass every frame.
    p.add(ScenePayloadRegistration {
        origin: p.origin(),
        key: VOXEL_FRAME.key(),
        publish: voxel_component::runtime::publish_payload,
        settling: voxel_component::runtime::settling,
        // Its sun samples rotate over blocks of pixels for TSR to integrate.
        temporal: true,
    });
}

#[cfg(test)]
mod tests {
    use helio_component::GroundMaterialList;
    use helio_pass_voxel_planet::terrain::material;

    #[test]
    fn registers_its_feature_and_settings() {
        let features =
            engine_backend::subsystems::render::helio_renderer::render_features::plugin_features();
        assert!(
            features.ids().contains(&"helio.voxel"),
            "features: {:?}",
            features.ids()
        );
        assert!(pulsar_settings::project::rendering::groups()
            .iter()
            .any(|group| std::ptr::eq(
                *group,
                &helio_pass_voxel_planet::settings::VOXEL_RESIDENCY
            )));
    }

    /// The plugin names the terrain's materials for every ground-material
    /// list (a spawn's surfaces, foliage's ground).
    #[test]
    fn material_lists_read_names_and_ids() {
        let list = GroundMaterialList::parse("Grass, dark_stone 22").unwrap();
        assert_eq!(list.0, [material::GRASS, material::DARK_STONE, 22]);
        assert_eq!(list.text(), "Grass, DarkStone, 22");
        assert_eq!(GroundMaterialList::parse(&list.text()).unwrap(), list);
        assert!(GroundMaterialList::parse("Grass, Lava").is_err());
        assert_eq!(GroundMaterialList::parse(" ").unwrap().0, Vec::<u32>::new());
    }

    /// Every registration of the plugin names this root, so it resolves to
    /// the plugin's id.
    #[test]
    fn every_registration_resolves_to_the_plugin() {
        let ours: Vec<_> = pulsar_plugins::contributions()
            .into_iter()
            .filter(|summary| summary.origin == pulsar_plugins::origin!())
            .collect();
        assert!(ours
            .iter()
            .any(|summary| summary.category == "component.class"));
        for summary in &ours {
            assert_eq!(
                pulsar_plugins::plugin_of(summary.origin).map(|plugin| plugin.id.as_str()),
                Some("pulsar.voxel"),
                "{summary:?}"
            );
        }
    }
}
