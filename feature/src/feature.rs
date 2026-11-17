//! Voxel terrain as a feature: the node that adds the chunk pass to preset
//! graphs.
//!
//! The terrain is a drawer of the G-buffer phase, after the other drawers:
//! it writes the G-buffer over theirs and traces GI's probe rays (made before
//! the phase). It draws the volumes the scene publishes
//! ([`crate::engine::VOXEL_FRAME`]), offers its ground to foliage
//! placement ([`helio_foliage_api::OfferedGrounds`]) and answers every ray
//! stream of the frame ([`tracer`]).

use std::sync::Arc;

use helio_feature_api::{BuildEnv, Feature, FeatureRegistrar, NodeDescriptor, ParamDesc};
use helio_frame_api::rays::stream::{RegisterRayTracer, Tracer};
use helio_frame_api::{FrameNode, Phase};

use crate::chunks::rays::{SharedRayLink, VoxelRayPass};
use crate::chunks::{pass::ChunkPass, ResidencyConfig};

/// The voxel feature's settings (a renderer config's `features`): with
/// them, graphs draw the voxel volumes the scene publishes; without them
/// the voxel node builds nothing.
#[derive(Clone, Debug, Default)]
pub struct VoxelSettings {
    /// How chunks stream (budgets, cache; the pixel angle follows the
    /// camera).
    pub residency: ResidencyConfig,
}

/// Voxel worlds ([`crate`]).
pub struct VoxelFeature;

/// Whether a graph built in `env` draws terrain, so traces it: with
/// [`VoxelSettings`], in the primary view (see the chunk node).
fn draws_terrain(env: &BuildEnv<'_>) -> bool {
    env.setting::<VoxelSettings>().is_some() && helio_frame_api::view::view_of(env).is_primary()
}

/// The voxel terrain's ray stream tracer (`chunks::rays`), as
/// [`VoxelFeature`] registers it; a host without features offers it to the
/// graph being built (`helio_frame_api::rays::stream::offer(env,
/// tracer())`). It needs no hardware ray tracing: wherever the terrain is
/// drawn, it is traced. Its passes reach the world through the graph's
/// [`SharedRayLink`], which the chunk pass publishes its cut to.
pub fn tracer() -> Tracer {
    let add = |surface: bool| -> helio_frame_api::rays::stream::TracerBuild {
        Arc::new(move |build, stream, id| {
            let link = build.env.shared(SharedRayLink::default).0.clone();
            let pass = VoxelRayPass::new(build.env.device, link, *stream, id, surface);
            build.graph.add_pass(Box::new(pass));
            Ok(())
        })
    };
    Tracer {
        name: "voxel",
        supported: Arc::new(draws_terrain),
        trace: add(false),
        surface: add(true),
    }
}

impl Feature for VoxelFeature {
    fn id(&self) -> &'static str {
        "helio.voxel"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.consumes(helio_light_api::SCENE_LIGHTS);
        registrar.consumes(helio_mats::MATERIALS);
        registrar.ray_tracer(tracer());
        // With terrain to draw, it traces GI's probes every frame and its sun
        // samples rotate over blocks of pixels for TSR to integrate.
        registrar.prepare(std::sync::Arc::new(|env| {
            if env.setting::<VoxelSettings>().is_some() {
                env.insert_setting(helio_gi_api::TerrainTracer);
                env.insert_setting(helio_pass_tsr::StochasticSamples);
            }
        }));
        registrar.node(
            NodeDescriptor::new(
                "helio.pass.voxel_chunks",
                "Voxel: Chunk Terrain",
                "Passes: Geometry",
                "The voxel volumes the scene publishes: residency, generation, the G-buffer, \
                 the GI probes' rays and the cut ray streams walk. Builds nothing without \
                 VoxelSettings.",
                |build| {
                    let Some(settings) = build.env.setting::<VoxelSettings>().cloned() else {
                        return Ok(());
                    };
                    // A secondary view (a reflection, a portal's, `helio_core::view`)
                    // draws no terrain yet: the world is this pass's, and its
                    // draw state (targets, denoiser history, per-eye motion,
                    // parameters) is one camera's. Drawing the frame's world
                    // for another view needs that state per view, the world
                    // shared; until then a second pass here would build a
                    // second world.
                    if !draws_terrain(build.env) {
                        return Ok(());
                    }
                    let mut pass = ChunkPass::new(settings.residency);
                    pass.set_ray_link(build.env.shared(SharedRayLink::default).0.clone());
                    build
                        .env
                        .shared(helio_foliage_api::OfferedGrounds::default)
                        .0
                        .push(pass.foliage_ground());
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .with_param(ParamDesc::boolean("enabled", "Enabled", true))
            .with_generated_pins()
            .drawer(Phase::GBuffer, 40),
        );
    }
}
