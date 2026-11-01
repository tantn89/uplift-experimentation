"""Policy value under a contact budget, estimated on a randomised test set.

A targeting policy treats the top ``budget`` share of users by predicted uplift and leaves the
rest untreated. Its incremental value over treating nobody is estimated two ways:

- **IPW** (inverse propensity weighting, the primary estimate): with the test treatment share
  ``e``, ``z = y (t / e - (1 - t) / (1 - e))`` is unbiased for each user's effect, so the
  policy's incremental outcomes per population user are ``sum(z over targeted) / N``. With
  ``e`` estimated on the same sample, treat-all equals the difference in means exactly;
- **difference in means** within the targeted users (uplift@k) times the targeted share.

Values are reported per 1,000 contacts (incremental outcomes per 1,000 treated users) and per
1,000 users of the population. Random targeting at any budget has the treat-all effect per
contact in expectation, so its per-contact value equals treat-all's.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from uplift_lab.uplift.metrics import Gains, top_count, uplift_at_k


@dataclass(frozen=True)
class PolicyValue:
    """Incremental outcomes of model targeting at one budget, with the two baselines."""

    budget: float
    targeted_share: float
    model_per_1000_contacts: float
    model_difference_per_1000_contacts: float
    model_per_1000_users: float
    treat_all_per_1000_contacts: float
    random_per_1000_users: float
    gain_over_random_per_1000_contacts: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _ipw_increment(g: Gains, j: int) -> float:
    """``sum(z)`` over the first ``j`` ranked users divided by the population weight."""
    e = g.n_t[-1] / g.total
    return float((g.y_t[j] / e - g.y_c[j] / (1.0 - e)) / g.total)


def average_treatment_effect(g: Gains) -> float:
    """Difference in outcome means between arms (equal to the IPW treat-all value)."""
    return _ipw_increment(g, len(g.weight) - 1)


def policy_value(g: Gains, budget: float) -> PolicyValue:
    """Value of treating the top ``budget`` share of the ranking behind ``g``."""
    j = top_count(g, budget)
    share = float(g.weight[j] / g.total)
    increment = _ipw_increment(g, j)
    ate = average_treatment_effect(g)
    per_contact = 1000.0 * increment / share if share > 0 else float("nan")
    return PolicyValue(
        budget=budget,
        targeted_share=share,
        model_per_1000_contacts=per_contact,
        model_difference_per_1000_contacts=1000.0 * uplift_at_k(g, budget),
        model_per_1000_users=1000.0 * increment,
        treat_all_per_1000_contacts=1000.0 * ate,
        random_per_1000_users=1000.0 * share * ate,
        gain_over_random_per_1000_contacts=per_contact - 1000.0 * ate,
    )
