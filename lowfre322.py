from pathlib import Path

from lowfre_shared import (
    EccentricResonantTidalGA as SharedEccentricResonantTidalGA,
    apply_default_lowfreq_source_kwargs,
    run_default_lowfreq_entry,
)


class EccentricResonantTidalGA(SharedEccentricResonantTidalGA):
    def __init__(self, *args, **kwargs):
        apply_default_lowfreq_source_kwargs(kwargs)
        kwargs.setdefault("transition_family", "fine")
        kwargs.setdefault("initial_state", (3, 0, 0))
        kwargs.setdefault("final_state", (3, 2, 2))
        kwargs.setdefault("orbital_backreaction_mode", "selected_rwa")
        kwargs.setdefault("module_stem", Path(__file__).stem)
        kwargs.setdefault("direction_tag", "upward")
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    run_default_lowfreq_entry(
        EccentricResonantTidalGA,
        transition_description="the fine transition |300> -> |322| with a 0.5 Msun companion",
    )
