from highfre_shared import (
    EccentricResonantTidalGA as SharedEccentricResonantTidalGA,
    resolve_default_highfreq_cloud_mass_fraction,
    run_default_highfreq_entry,
)


class EccentricResonantTidalGA(SharedEccentricResonantTidalGA):
    """High-frequency preset for the downward Bohr transition |644> -> |544|."""

    def __init__(self, *args, **kwargs):
        defaults = {
            "resonance_harmonic": 1,
            "max_harmonic": 8,
            "multi_harmonic_drive": True,
            "harmonics_to_keep": 8,
            "binary_harmonics": 12,
            "transition_family": "bohr",
            "initial_state": (6, 4, 4),
            "final_state": (5, 4, 4),
            "tidal_m": 0,
            "cloud_mass_fraction": resolve_default_highfreq_cloud_mass_fraction(),
            "geom_factor": None,
            "module_stem": "highfre644v",
            "direction_tag": "downward",
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(*args, **kwargs)


__all__ = ["EccentricResonantTidalGA"]


if __name__ == "__main__":
    run_default_highfreq_entry(
        EccentricResonantTidalGA,
        "Building highfre644v: waveform-only high-frequency |644> -> |544| model...",
    )
