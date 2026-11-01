"""A/B analysis: Welch effects, CUPED, SRM, power and always-valid sequential tests.

Public names are re-exported lazily (``uplift_lab.lazy_exports``), so code that only needs a
submodule such as ``estimators`` (its ``FloatArray`` alias is used across the package) does not
import statsmodels or scipy.stats.
"""

from typing import TYPE_CHECKING

from uplift_lab.lazy_exports import lazy_exports

if TYPE_CHECKING:
    from uplift_lab.experiments.cuped import (
        CupedResult,
        RegressionAdjustment,
        cuped_difference,
        cuped_theta,
        regression_adjusted_effect,
    )
    from uplift_lab.experiments.estimators import EffectEstimate, welch_difference
    from uplift_lab.experiments.msprt import (
        always_valid_ci_halfwidth,
        always_valid_p_values,
        msprt_log_lr,
        msprt_reject,
    )
    from uplift_lab.experiments.peeking_simulation import (
        PeekingConfig,
        PeekingOutcome,
        simulate_peeking,
        summarise,
    )
    from uplift_lab.experiments.power import minimum_detectable_effect, power_at, required_n
    from uplift_lab.experiments.srm import SrmResult, srm_test

__all__ = [
    "CupedResult",
    "EffectEstimate",
    "PeekingConfig",
    "PeekingOutcome",
    "RegressionAdjustment",
    "SrmResult",
    "always_valid_ci_halfwidth",
    "always_valid_p_values",
    "cuped_difference",
    "cuped_theta",
    "minimum_detectable_effect",
    "msprt_log_lr",
    "msprt_reject",
    "power_at",
    "regression_adjusted_effect",
    "required_n",
    "simulate_peeking",
    "srm_test",
    "summarise",
    "welch_difference",
]

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "cuped": (
            "CupedResult",
            "RegressionAdjustment",
            "cuped_difference",
            "cuped_theta",
            "regression_adjusted_effect",
        ),
        "estimators": ("EffectEstimate", "welch_difference"),
        "msprt": (
            "always_valid_ci_halfwidth",
            "always_valid_p_values",
            "msprt_log_lr",
            "msprt_reject",
        ),
        "peeking_simulation": ("PeekingConfig", "PeekingOutcome", "simulate_peeking", "summarise"),
        "power": ("minimum_detectable_effect", "power_at", "required_n"),
        "srm": ("SrmResult", "srm_test"),
    },
)
