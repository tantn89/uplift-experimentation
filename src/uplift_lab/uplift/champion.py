"""The pre-declared rule that picks the model to register and serve (the champion).

1. Rank the learners by the Qini coefficient on the test set; the top one is the "best".
2. Test the best against every other learner with the paired bootstrap difference of their Qini
   coefficients, Holm-corrected across those comparisons (``evaluation.py``).
3. If every difference is significant, the best is a clear winner and becomes the champion.
4. Otherwise the learners that are not significantly worse than the best are tied with it, and
   the champion is the first of them in the fixed order ``CHAMPION_ORDER``: S > T > X >
   transformed outcome.

Why that order: it runs from the simplest model to the most propensity-sensitive one. S and T
fit outcome models and never use the propensity. X uses it only as a blending weight between two
effect models, which does not bias the blend. The transformed outcome weights every training row
by ``1 / e`` or ``1 / (1 - e)`` with one constant ``e``, so it is biased wherever the local
treated share differs from ``e``; the randomisation check (held-out AUC about 0.51 on Criteo)
shows that treatment is slightly predictable from the features. When the data cannot separate
the learners, the rule takes the one whose assumptions the data contradicts least.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

CHAMPION_ORDER = ("s_learner", "t_learner", "x_learner", "transformed_outcome")
CHAMPION_RULE = "highest_qini_if_clear_winner_else_simplest"
CHAMPION_RULE_TEXT = (
    "Rank learners by test-set Qini coefficient. If the best one's paired bootstrap Qini "
    "differences against every other learner are significant after Holm correction, it is the "
    "champion. Otherwise choose, among the learners not significantly worse than the best, the "
    "first in the fixed order s_learner > t_learner > x_learner > transformed_outcome "
    "(simplest and least propensity-sensitive first)."
)


def _preference(name: str) -> int:
    return CHAMPION_ORDER.index(name) if name in CHAMPION_ORDER else len(CHAMPION_ORDER)


def select_champion(
    best: str, rivals: Sequence[str], p_holm: Mapping[str, float], alpha: float
) -> dict[str, Any]:
    """Apply the rule; ``p_holm`` maps every rival to its Holm-adjusted p-value against ``best``.

    A rival whose adjusted p-value is missing or NaN (no finite replicate) counts as tied.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    tied = [name for name in rivals if not p_holm.get(name, float("nan")) <= alpha]
    payload: dict[str, Any] = {
        "rule": CHAMPION_RULE,
        "rule_text": CHAMPION_RULE_TEXT,
        "order": list(CHAMPION_ORDER),
        "best_by_qini": best,
        "tied_with_best": tied,
    }
    if not tied:
        detail = "every Holm-adjusted paired Qini difference is significant" if rivals else ""
        payload["model"] = best
        payload["reason"] = f"{best} has the highest Qini coefficient and is a clear winner" + (
            f": {detail}" if detail else " (no other learner was trained)"
        )
        return payload
    pool = [best, *tied]
    champion = min(pool, key=lambda name: (_preference(name), pool.index(name)))
    payload["model"] = champion
    payload["reason"] = (
        f"no clear winner: {best} has the highest Qini coefficient, but its Holm-adjusted paired "
        f"differences against {', '.join(tied)} are not significant at alpha {alpha:g}; "
        f"{champion} is the first of the tied learners in the order "
        f"{' > '.join(CHAMPION_ORDER)}"
    )
    return payload
