# Plugin_Voxel

Destructible voxel terrain: its components, runtime, file formats, ready-made worlds and the renderer feature that draws it.

`pulsar.voxel` ("Voxel Worlds") is a plugin of
[Pulsar-Native](https://github.com/Far-Beyond-Pulsar/Pulsar-Native), the engine and editor that renders with
[Helio](https://github.com/Far-Beyond-Pulsar/Helio). Pulsar-Native vendors this repository as a git submodule
at `plugins/vendor/voxel`.

## What it owns

Every object kind is a plugin. A plugin owns its SceneDB components (their
data and scriptable methods; the editor calls the same methods scripts do),
the contracts other kinds read (its `api` crate: graph keys, row ABIs, WGSL
snippets), the Helio feature that renders it (passes, scene derivations,
settings), its file formats and its editor UI. Helio knows no object kind,
and other kinds meet this one only through frame contracts, extension points,
draw phases and api crates.

| Crate | Path | What |
|---|---|---|
| `voxel_plugin` | `./` | Voxel Worlds (`pulsar.voxel`): the plugin's root crate, the one place it registers everything it contributes (runtime registrations always, the editor's with `editor`) |
| `helio-voxel-data` | `api/` | Renderer-independent voxel data and SceneDB publication contract |
| `voxel_component` | `component/` | Voxel worlds (pulsar.voxel): the voxel terrain components, their methods and runtime, the voxel file formats, ready-made worlds, and the plugin's Helio feature set; the plugin's root crate registers them |
| `voxel_editor` | `editor/` | Voxel worlds (pulsar.voxel): the editor crate -- the voxel terrain's inspector editors and the level editor's voxel tools, over the plugin's component crate |
| `helio-voxel-examples` | `examples/` | The voxel plugin's standalone hosts on Helio: Voxel Craft, Voxel Flight and Water World |
| `helio-pass-voxel-planet` | `feature/` | Destructible voxel worlds, drawn: chunk-tree residency, generation, drawing and lighting as Helio passes over helio-voxel-world |
| `helio-voxel-world` | `world/` | Destructible voxel worlds, the data: grid, terrain generators, edits, queries, planet maps and features, mirrored bit for bit by WGSL. No GPU. |

- **Root crate**: `voxel_plugin`. It is the one place the plugin submits
  its contributions (component classes, features, scene seams, settings,
  editor tools), each naming it as origin, so the plugin gate and the
  plugin catalog see all of them. The other crates submit none (only type
  attachments, such as a component's GPU mirror, stay with their types). A
  game links the root without its `editor` feature; the editor links it with.
- **Requires**: nothing. A project enables or disables the plugin in its
  `Pulsar.toml` (`[plugins]`); disabled, the kind is absent and the frame
  draws without it.
- **Reads other kinds through**: `helio-foliage-api` (foliage); `helio-gi-api` (gi); `helio-light-api` (lights); `helio-mats` (materials); `helio-water-api` (water).
- **Its standalone hosts also compose**: `helio-pass-decal` (decals); `helio-pass-volumetric-fog` (fog); `helio-pass-foliage-gbuffer, helio-pass-foliage-place` (foliage); `helio-pass-gi` (gi); `helio-pass-flare, helio-pass-lens` (lens); `helio-light-api, helio-light-host` (lights); `helio-mats` (materials); `helio-mesh-api, helio-mesh-host, helio-pass-forward-lit` (meshes); `helio-pass-corona` (particles); `helio-pass-portal-cull, helio-pass-portal-instances` (portals); `helio-pass-post-volumes` (post); `helio-pass-sky` (sky); `helio-pass-sprite-batch` (sprites); `helio-pass-water-sim` (water) (demos and GPU tests
  draw a whole scene; the plugin itself does not depend on these).

## Building and testing

Like [Plugin_Physics](https://github.com/Far-Beyond-Pulsar/Plugin_Physics), this plugin's crates are members
of Pulsar-Native's workspace: their dependencies are `workspace = true`, and
Helio comes from Pulsar-Native's `crates/renderer/helio` submodule. It builds
only inside a Pulsar-Native checkout:

```bash
git clone --recursive https://github.com/Far-Beyond-Pulsar/Pulsar-Native
cd Pulsar-Native
cargo check -p helio-voxel-data -p voxel_component -p voxel_plugin -p voxel_editor -p helio-voxel-examples -p helio-pass-voxel-planet -p helio-voxel-world --tests
cargo test -p helio-voxel-data -p voxel_plugin -p voxel_component -p helio-pass-voxel-planet
```

Suites that render real frames need a GPU; CI leaves them out by name
(Pulsar-Native's `.config/nextest.toml`, `ci` profile). This repository's CI
checks the plugin inside a Pulsar-Native checkout: rustfmt, clippy and the
`ci` profile's tests.

A change that spans this plugin and Pulsar-Native (or Helio) merges
dependency-first: merge here, re-pin `plugins/vendor/voxel` in Pulsar-Native to
this repository's `main`, then merge the Pulsar-Native PR (Pulsar-Native's
`.agents/CRATES.md`, "Submodule pins").

## Architecture

- [Every object kind is a plugin](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/docs/architecture/plugin-migration.md): the
  rule, what Helio is, the plugin table and the "As built" notes of each
  phase.
- [The layered rewrite](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/docs/architecture/layered-rewrite.md): layers, buffer
  contracts, extension points and why the renderer has this shape.
- [The `helio-architecture` skill](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/.agents/skills/helio-architecture/SKILL.md): required reading for any change
  here (see `AGENTS.md`).
- Pulsar-Native's `.agents/PLUGIN_SYSTEM.md` and `.agents/CRATES.md`: plugin
  descriptors, the installed set and the submodule pins.

History before 2026-11-02 (when the plugin moved into Pulsar-Native) is in
Helio's repository; this repository's history starts with that move
(`git subtree split` of `plugins/vendor/voxel`).

## License

MIT, as the rest of Pulsar (`LICENSE.md`).
