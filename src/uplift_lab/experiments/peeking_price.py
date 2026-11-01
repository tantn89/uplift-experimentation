"""The price of peeking: the sample size mSPRT needs to match the fixed-horizon test's power.

The fixed-horizon Welch test at ``config.horizon_per_arm`` has the target power (80%) for the
design effect. mSPRT with the same mixing ``tau`` stays valid when checked at every look, but at
that horizon it rejects less often. An mSPRT decision at a look never depends on later data, so
one simulation with a longer horizon gives mSPRT's power under every shorter cap: the share of
runs that have rejected by that look. The price is the first cap at which that share reaches the
target, relative to the fixed horizon.

Because mSPRT stops as soon as the evidence is strong, its *expected* sample size under that cap
can still be below the fixed horizon. Both numbers are reported, plus mSPRT's type-I error under
the same cap (a fresh A/A simulation) to show the larger cap costs no validity.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from uplift_lab.experiments.peeking_simulation import PeekingConfig, simulate_peeking

MAX_SIMS = 10_000  # memory bound: the simulation holds n_sims x looks arrays


def price_of_peeking(
    config: PeekingConfig, *, target_power: float | None = None, max_horizon_factor: int = 4
) -> dict[str, Any]:
    """mSPRT power by cap for the design effect, and the cap that reaches ``target_power``."""
    target = config.power if target_power is None else target_power
    if not 0.0 < target < 1.0 or max_horizon_factor < 1:
        raise ValueError("target_power must be in (0, 1) and max_horizon_factor >= 1")
    effect = config.design_effect()  # also the mSPRT tau, as in the main simulation
    extended = replace(
        config,
        n_sims=min(config.n_sims, MAX_SIMS),
        looks=config.looks * max_horizon_factor,
        seed=config.seed + 2,
    )
    alternative = simulate_peeking(extended, effect, tau=effect)
    power_by_cap = alternative.cumulative_rejection("msprt")
    reached = np.flatnonzero(power_by_cap >= target)
    payload: dict[str, Any] = {
        "target_power": target,
        "effect": effect,
        "tau": effect,
        "n_sims": extended.n_sims,
        "fixed_horizon_per_arm": config.horizon_per_arm,
        "max_cap_per_arm": extended.horizon_per_arm,
        "msprt_power_at_fixed_horizon": float(power_by_cap[config.looks - 1]),
        "power_by_cap": {
            "n_per_arm": alternative.looks_n.tolist(),
            "msprt_power": power_by_cap.tolist(),
        },
        "msprt_cap_per_arm": None,
        "cap_ratio_vs_fixed": None,
        "msprt_mean_n_per_arm_under_cap": None,
        "mean_n_ratio_vs_fixed": None,
        "msprt_type_one_error_under_cap": None,
    }
    if reached.size == 0:
        return payload  # not reached within max_horizon_factor x the fixed horizon
    last = int(reached[0])
    cap = int(alternative.looks_n[last])
    first = alternative.first_rejection["msprt"]
    stops = np.where((first >= 0) & (first <= last), alternative.looks_n[np.maximum(first, 0)], cap)
    null = simulate_peeking(
        replace(extended, looks=last + 1, seed=config.seed + 3), 0.0, tau=effect
    )
    payload.update(
        msprt_cap_per_arm=cap,
        cap_ratio_vs_fixed=cap / config.horizon_per_arm,
        msprt_mean_n_per_arm_under_cap=float(stops.mean()),
        mean_n_ratio_vs_fixed=float(stops.mean()) / config.horizon_per_arm,
        msprt_type_one_error_under_cap=null.rejection_rate("msprt"),
    )
    return payload
