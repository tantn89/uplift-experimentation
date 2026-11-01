"""Markdown and self-contained HTML renderings of the drift payload (same tables in both)."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from uplift_lab.monitoring.drift import PREDICTION

Table = tuple[list[str], list[list[str]]]


def _number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _decision_rate(rate: dict[str, Any] | None) -> str:
    if rate is None:
        return "-"
    mark = {"ok": "", "warn": ", warn", "alert": ", ALERT"}[rate["status"]]
    return f"{rate['treat_rate']:.3f} ({rate['ratio']:.2f}x{mark})"


def summary_table(payload: dict[str, Any]) -> Table:
    header = [
        "Scenario",
        "Status",
        "PSI-flagged features",
        "KS-rejected",
        "Uplift-score PSI",
        f"Treat rate @ budget {payload['decision']['budget']} (x budget)",
        "Treated share",
    ]
    rows = []
    for name, s in payload["scenarios"].items():
        prediction = s["prediction"]["psi"] if s["prediction"] else None
        rows.append(
            [
                name,
                s["status"].upper(),
                ", ".join(s["flagged_features"]) or "none",
                str(len(s["ks_rejected_features"])),
                _number(prediction),
                _decision_rate(s.get("decision_rate")),
                _number(s.get("treated_share")),
            ]
        )
    return header, rows


def psi_table(payload: dict[str, Any]) -> Table:
    scenarios = payload["scenarios"]
    columns = [*next(iter(scenarios.values()))["features"], PREDICTION]
    rows = []
    for column in columns:
        row = [column]
        for summary in scenarios.values():
            result = summary["prediction"] if column == PREDICTION else summary["features"][column]
            if result is None:
                row.append("-")
                continue
            mark = {"ok": "", "warn": " (warn)", "alert": " (ALERT)"}[result["status"]]
            row.append(f"{result['psi']:.3f}{mark}")
        rows.append(row)
    return ["PSI", *scenarios], rows


def _intro(payload: dict[str, Any]) -> list[str]:
    t, model, ref = payload["thresholds"], payload["model"], payload["reference"]
    rows = payload["config"]["traffic_rows"]
    (low_warn, high_warn), (low_alert, high_alert) = t["rate_warn"], t["rate_alert"]
    return [
        f"Overall status: {payload['overall_status'].upper()}.",
        f"Reference: {ref['n']:,} training rows; scenarios resample {rows:,} "
        f"rows of {ref['n_held_out']:,} held-out rows (seed {payload['config']['seed']}).",
        f"Model: {model['kind']} from {model['source']}.",
        f"PSI on reference deciles: warn >= {t['psi_warn']:g}, alert >= {t['psi_alert']:g}. "
        f"KS (Bonferroni alpha {t['ks_alpha']:g}) is reported, not alerted on.",
        f"Decision rate: share of traffic at or above the budget-{payload['decision']['budget']} "
        f"threshold ({payload['decision']['threshold']:.5f}) divided by the budget; "
        f"warn outside [{low_warn:g}, {high_warn:g}], "
        f"alert outside [{low_alert:g}, {high_alert:g}].",
    ]


def _markdown_table(table: Table) -> list[str]:
    header, rows = table
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def write_markdown(payload: dict[str, Any], path: Path, figure: Path | None) -> Path:
    lines = ["# Drift report", "", *(f"- {line}" for line in _intro(payload)), ""]
    lines += ["## Scenarios", "", *_markdown_table(summary_table(payload)), ""]
    for name, s in payload["scenarios"].items():
        lines.append(f"- **{name}**: {s['description']}")
    lines += ["", "## PSI per column", "", *_markdown_table(psi_table(payload)), ""]
    if figure is not None:
        lines += [f"![PSI per column]({figure.name})", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _html_table(table: Table) -> str:
    header, rows = table
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html(payload: dict[str, Any], path: Path, figure: Path | None) -> Path:
    """One file with inline CSS and the figure embedded as base64; no external resources."""
    intro = "".join(f"<li>{html.escape(line)}</li>" for line in _intro(payload))
    notes = "".join(
        f"<li><b>{html.escape(name)}</b>: {html.escape(s['description'])}</li>"
        for name, s in payload["scenarios"].items()
    )
    image = ""
    if figure is not None:
        encoded = base64.b64encode(figure.read_bytes()).decode("ascii")
        image = f'<img alt="PSI per column" src="data:image/png;base64,{encoded}">'
    style = (
        "body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}"
        "table{border-collapse:collapse;margin:1em 0}td,th{border:1px solid #ccc;padding:4px 8px;"
        "text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%}"
    )
    document = (
        f"<!doctype html><html lang=en><head><meta charset=utf-8><title>Drift report</title>"
        f"<style>{style}</style></head><body><h1>Drift report</h1><ul>{intro}</ul>"
        f"<h2>Scenarios</h2>{_html_table(summary_table(payload))}<ul>{notes}</ul>"
        f"<h2>PSI per column</h2>{_html_table(psi_table(payload))}{image}</body></html>\n"
    )
    path.write_text(document, encoding="utf-8")
    return path
