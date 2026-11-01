"""Policy value under a contact budget, estimated on a randomised test set.

A targeting policy treats the top ``budget`` share of users by predicted uplift and leaves the
rest untreated. Its incremental value over treating nobody is estimated two ways:

- **difference in means** among the targeted users (the primary estimate): the treated minus
  control outcome rate inside the targeted segment, i.e. inverse propensity weighting with
  weights normalised within each arm of the segment (Hajek). It stays unbiased when the
  segment's treated share drifts from the overall share;
- **Horvitz-Thompson IPW** with the overall test treatment share ``e``:
  ``z = y (t / e - (1 - t) / (1 - e))`` summed over targeted users. It is unbiased only if the
  treated share is the same in every segment. On Criteo the share in the top decile is 0.865
  against 0.85 overall, and visit rates there are high, so this estimate runs high.

Values are reported per 1,000 contacts (incremental outcomes per 1,000 treated users) and per
1,000 users of the population. Random targeting at any budget has the treat-all effect per
contact in expectation, so its per-contact value equals treat-all's. With ``e`` taken from the
same sample, both estimators equal the overall difference in means at a budget of 1.
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
    model_ipw_per_1000_contacts: float
    model_per_1000_users: float
    treat_all_per_1000_contacts: float
    random_per_1000_users: float
    gain_over_random_per_1000_contacts: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _horvitz_thompson(g: Gains, j: int) -> float:
    """``sum(z)`` over the first ``j`` ranked users divided by the population weight."""
    e = g.n_t[-1] / g.total
    return float((g.y_t[j] / e - g.y_c[j] / (1.0 - e)) / g.total)


def average_treatment_effect(g: Gains) -> float:
    """Difference in outcome means between arms over the whole sample."""
    return float(g.y_t[-1] / g.n_t[-1] - g.y_c[-1] / g.n_c[-1])


def policy_value(g: Gains, budget: float) -> PolicyValue:
    """Value of treating the top ``budget`` share of the ranking behind ``g``."""
    j = top_count(g, budget)
    share = float(g.weight[j] / g.total)
    per_contact = uplift_at_k(g, budget)  # NaN when an arm is empty in the segment
    ate = average_treatment_effect(g)
    ipw = _horvitz_thompson(g, j) / share if share > 0 else float("nan")
    return PolicyValue(
        budget=budget,
        targeted_share=share,
        model_per_1000_contacts=1000.0 * per_contact,
        model_ipw_per_1000_contacts=1000.0 * ipw,
        model_per_1000_users=1000.0 * per_contact * share,
        treat_all_per_1000_contacts=1000.0 * ate,
        random_per_1000_users=1000.0 * share * ate,
        gain_over_random_per_1000_contacts=1000.0 * (per_contact - ate),
    )
