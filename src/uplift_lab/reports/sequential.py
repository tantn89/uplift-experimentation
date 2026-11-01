"""Peeking simulation report: ``python -m uplift_lab.reports.sequential [--large]``.

Simulates A/A and A/B experiments inspected at every look and scores a naive t-test against
mSPRT. All numbers come from a seeded simulation; the default run takes well under a second and
``--large`` (20,000 runs with 100 looks each) about two seconds.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from uplift_lab.experiments.peeking_simulation import PeekingConfig, simulate_peeking, summarise
from uplift_lab.reports.output import MISSING_MATPLOTLIB, write_json

LARGE = PeekingConfig(n_sims=20_000, looks=100, batch_per_arm=100)
MIXING_NOTE = (
    "tau (mixing sd) = the absolute lift the fixed-horizon test detects with the target power at "
    "the horizon; fixed before the data, used for both scenarios"
)


def write_sequential_report(
    config: PeekingConfig, output_dir: Path, *, figures: bool = True
) -> dict[str, Any]:
    """Simulate both scenarios, write ``sequential.json`` and figures, return the payload."""
    effect = config.design_effect()
    null = simulate_peeking(config, 0.0)
    alternative = simulate_peeking(replace(config, seed=config.seed + 1), effect)
    payload: dict[str, Any] = {
        "config": config.to_dict(),
        "tau": null.tau,
        "mixing_variance_choice": MIXING_NOTE,
        "design_effect": effect,
        "aa": summarise(null),
        "ab": summarise(alternative),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "sequential.json", payload)
    if figures:
        try:
            from uplift_lab.reports.sequential_figures import peeking_figure, stopping_figure
        except ImportError as error:
            raise SystemExit(MISSING_MATPLOTLIB) from error
        peeking_figure(null, alternative, config.alpha, output_dir / "sequential-peeking.png")
        stopping_figure(alternative, output_dir / "sequential-stopping.png")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Naive peeking vs mSPRT simulation report.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/sequential"))
    parser.add_argument("--large", action="store_true", help="20,000 runs x 100 looks")
    parser.add_argument("--n-sims", type=int, default=None)
    parser.add_argument("--looks", type=int, default=None)
    parser.add_argument("--batch-per-arm", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    config = LARGE if args.large else PeekingConfig()
    overrides = {
        name: value
        for name, value in (
            ("n_sims", args.n_sims),
            ("looks", args.looks),
            ("batch_per_arm", args.batch_per_arm),
            ("seed", args.seed),
        )
        if value is not None
    }
    try:
        config = replace(config, **overrides)
    except ValueError as error:
        parser.error(str(error))
    payload = write_sequential_report(config, args.output_dir, figures=not args.no_figures)
    for scenario in ("aa", "ab"):
        rates = {
            rule: round(stats["rejection_rate"], 4) for rule, stats in payload[scenario].items()
        }
        print(f"{scenario}: rejection rate {rates}")
    median_stop = payload["ab"]["msprt"]["median_stop_n_if_rejected"]
    print(f"mSPRT median stop (A/B, rejecting runs): {median_stop:.0f} units per arm")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
