//! A bundle's pipelines compiled side by side.
//!
//! The helper is Helio's (`helio_core::compile::SideBySide`: a bundle's
//! pipelines named as they are made, compiled together on the host's compute
//! lanes inside the bundle's own compile job); the voxel feature's chunk
//! bundles keep their names for it.

pub(super) use helio_core::compile::{SideBySide as Kernels, Slot as Kernel};
