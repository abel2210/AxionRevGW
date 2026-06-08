from highfre_shared import (
    EccentricResonantTidalGA as SharedEccentricResonantTidalGA,
    resolve_default_highfreq_cloud_mass_fraction,
    run_default_highfreq_entry,
)


class EccentricResonantTidalGA(SharedEccentricResonantTidalGA):
    """High-frequency preset for the downward fine transition |322> -> |300|."""

    def __init__(self, *args, **kwargs):
        defaults = {
            "resonance_harmonic": 4,
            "max_harmonic": 8,
            "multi_harmonic_drive": True,
            "harmonics_to_keep": 8,
            "binary_harmonics": 12,
            "transition_family": "fine",
            "initial_state": (3, 2, 2),
            "final_state": (3, 0, 0),
            "tidal_m": -2,
            "cloud_mass_fraction": resolve_default_highfreq_cloud_mass_fraction(),
            "geom_factor": None,
            "module_stem": "highfre322v",
            "direction_tag": "downward",
        }
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        super().__init__(*args, **kwargs)


__all__ = ["EccentricResonantTidalGA"]


if __name__ == "__main__":
    run_default_highfreq_entry(
        EccentricResonantTidalGA,
        "Building highfre322v: waveform-only high-frequency |322> -> |300| model...",
    )
