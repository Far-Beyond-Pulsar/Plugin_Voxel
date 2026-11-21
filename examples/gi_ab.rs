//! GI's quality for an A/B run of a voxel example: `HELIO_GI_QUALITY`
//! (`probes`, `pixels_16`, `pixels_4`, `pixels_1`: the `gi` settings group's
//! `quality`) and `HELIO_GI_PIXEL_REUSE=0` (no sample reuse) over the
//! example's own GI settings.

/// `settings` with the run's choice of GI over them.
pub fn gi_settings(mut settings: helio_pass_gi::GiSettings) -> helio_pass_gi::GiSettings {
    if let Ok(quality) = std::env::var("HELIO_GI_QUALITY") {
        settings.quality = helio_pass_gi::GiQuality::from_key(&quality)
            .unwrap_or_else(|| panic!("HELIO_GI_QUALITY: no quality `{quality}`"));
    }
    if let Ok(reuse) = std::env::var("HELIO_GI_PIXEL_REUSE") {
        settings.pixel_reuse = reuse != "0";
    }
    settings
}
