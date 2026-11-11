//! The voxel terrain's pass is a node of render graphs: registered by the
//! plugin's root (through its render feature), pins found by the probe from what
//! the pass declares, built from a graph asset, and absent, with the reason,
//! while the plugin is disabled.

// The plugin as a game links it: its root, with every registration.
use voxel_plugin as _;

use std::sync::{Arc, Mutex};

use engine_backend::subsystems::render::helio_renderer::{pipelines, render_features};
use helio::{DebugDrawState, PassBuildContext, RendererConfig};
use helio_graph_asset::{validate, NodeInstance, PinKind};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

const VOXEL_NODE: &str = "helio.pass.voxel_chunks";

/// The plugin gate is process-wide state and a wgpu device is one at a time
/// (Pulsar-Native#1647): every test takes this first.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: helio::required_wgpu_features(adapter.features()),
                required_limits: helio::required_wgpu_limits(adapter.limits()),
                experimental_features: helio::required_experimental_features(adapter.features()),
                ..Default::default()
            })
            .await
            .ok()?;
        Some((Arc::new(device), Arc::new(queue)))
    })
}

fn buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn configured() -> RendererConfig {
    // The renderer config with the plugins' settings, as a host makes it.
    render_features::configure(RendererConfig::new(
        320,
        180,
        wgpu::TextureFormat::Rgba8Unorm,
    ))
}

/// The deferred preset's graph over the registry.
fn asset_of(registry: &helio_graph_asset::NodeRegistry) -> helio_graph_asset::GraphAsset {
    helio_presets::graphs::deferred_graph_asset_for(registry, &configured())
}

/// Compiles `asset` and counts the chunk passes in it (`None`: no GPU).
fn voxel_pass_count(asset: &helio_graph_asset::GraphAsset) -> Option<usize> {
    let (device, queue) = device()?;
    let gpu = EngineGpuContext::new(Arc::clone(&device), Arc::clone(&queue));
    let store = SceneGpuStore::new(
        &gpu,
        SceneGpuConfig {
            classes: Vec::new(),
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    );
    let camera = buffer(&device, 512);
    let debug_camera = buffer(&device, 64);
    let cull_stats = buffer(&device, 64);
    let mut env = helio_graph_asset::build_env(&PassBuildContext {
        device: &device,
        queue: &queue,
        config: configured(),
        debug_state: Arc::new(Mutex::new(DebugDrawState::default())),
        camera_buffer: &camera,
        debug_camera_buffer: &debug_camera,
        cull_stats_buffer: &cull_stats,
        owns_device: false,
        scene_db: GpuMirrorHandle::new(Arc::new(store), Arc::clone(&queue)),
    });
    let registry = helio_presets::graphs::registry_with(&render_features::plugin_features());
    let graph = helio_graph_asset::compile(asset, &registry, &mut env)
        .unwrap_or_else(|errors| panic!("the graph builds: {errors:?}"));
    Some(
        (0..graph.pass_count())
            .filter_map(|index| graph.pass_declaration(index))
            .filter(|pass| pass.type_name.contains("ChunkPass"))
            .count(),
    )
}

#[test]
fn the_plugin_registers_the_node_and_the_graph_builds_it() {
    let _serial = serial();
    pulsar_plugins::deactivate();
    let registry = helio_presets::graphs::registry_with(&render_features::plugin_features());
    let node = registry.get(VOXEL_NODE).expect("the voxel node is registered");
    assert_eq!(registry.feature_of(VOXEL_NODE), Some("helio.voxel"));
    assert!(node.generated, "pins come from what the pass declares");
    assert!(node.params.iter().any(|p| p.key == "enabled"));
    assert!(node.placement.is_some(), "the preset places it at the G-buffer");

    let mut asset = asset_of(&registry);
    assert!(
        asset.nodes.iter().any(|n| n.kind == VOXEL_NODE),
        "the default graph holds the voxel node"
    );
    let Some(enabled) = voxel_pass_count(&asset) else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: voxel graph node");
        return;
    };
    assert_eq!(enabled, 1, "the node builds the chunk pass");

    // The Enabled parameter bypasses the node: no pass.
    for node in asset.nodes.iter_mut().filter(|n| n.kind == VOXEL_NODE) {
        node.params.insert("enabled".into(), false.into());
    }
    assert_eq!(voxel_pass_count(&asset), Some(0));
}

#[test]
fn the_probe_gives_the_node_the_pins_its_pass_declares() {
    let _serial = serial();
    pulsar_plugins::deactivate();
    let registry = pipelines::graph_node_registry();
    let node = registry.get(VOXEL_NODE).expect("the voxel node is registered");
    let pins = node.pins(&Default::default());
    if pipelines::graph_pin_probe().is_none() {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: voxel node pins");
        return;
    }
    assert!(pins.inputs.iter().any(|p| p.ty.kind == PinKind::Flow));
    assert!(pins.outputs.iter().any(|p| p.ty.kind == PinKind::Flow));
    let written: Vec<_> = pins
        .outputs
        .iter()
        .filter(|p| p.ty.kind != PinKind::Flow)
        .map(|p| p.resource.as_str())
        .collect();
    assert!(!written.is_empty(), "the pass writes resources: {pins:?}");
    eprintln!("VOXEL_NODE_PINS {pins:?}");
}

#[test]
fn a_disabled_plugin_has_no_node_and_a_graph_naming_it_says_so() {
    let _serial = serial();
    pulsar_plugins::deactivate();
    let with_plugin = helio_presets::graphs::registry_with(&render_features::plugin_features());
    let mut asset = asset_of(&with_plugin);
    asset.nodes.push(NodeInstance {
        id: "terrain".into(),
        kind: VOXEL_NODE.into(),
        params: Default::default(),
        position: [0.0; 2],
    });

    pulsar_plugins::activate(&pulsar_plugins::resolve(
        pulsar_plugins::installed(),
        &pulsar_plugins::ProjectPlugins::default().with(&["pulsar.voxel"], false),
    ));
    let registry = pipelines::graph_node_registry();
    pulsar_plugins::deactivate();

    assert!(registry.get(VOXEL_NODE).is_none(), "its node is unregistered");
    assert_eq!(
        registry.absence(VOXEL_NODE),
        Some("plugin pulsar.voxel disabled")
    );
    let validation = validate(&asset, &registry);
    let messages: Vec<_> = validation.errors().map(|d| d.message.clone()).collect();
    assert!(
        messages.iter().any(|m| {
            m == "missing node `helio.pass.voxel_chunks` (plugin pulsar.voxel disabled)"
        }),
        "{messages:?}"
    );
    // Enabled again, the registry follows.
    assert!(pipelines::graph_node_registry().get(VOXEL_NODE).is_some());
}
