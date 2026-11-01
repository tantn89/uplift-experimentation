"""Latency benchmark for a running scoring service: ``uplift-serve-benchmark --url ...``.

Sends sequential requests (one client, one keep-alive connection) to ``/score`` and ``/decide``
at each batch size and records client-side wall time per request, which includes JSON encoding,
HTTP and server time. Warm-up requests are discarded. Customer rows are seeded random feature
values; LightGBM scoring time does not depend on them. Writes a JSON summary with p50/p95/mean/
max in milliseconds. Standard library only, so it also runs against a container or Cloud Run.
"""

from __future__ import annotations

import argparse
import http.client
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from uplift_lab.data.criteo import FEATURES
from uplift_lab.reports.output import write_json


class Endpoint:
    """A persistent HTTP(S) connection to one service; raises on any non-200 response."""

    def __init__(self, url: str, token: str | None, timeout: float) -> None:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError(f"expected an http(s) URL, got {url!r}")
        kind = (
            http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        )
        self.connection = kind(parts.hostname, parts.port, timeout=timeout)
        self.headers = {"content-type": "application/json"}
        if token:
            self.headers["authorization"] = f"Bearer {token}"

    def request(self, method: str, path: str, body: bytes | None = None) -> Any:
        self.connection.request(method, path, body=body, headers=self.headers)
        response = self.connection.getresponse()
        data = response.read()
        if response.status != 200:
            raise RuntimeError(f"{method} {path} -> {response.status}: {data[:300]!r}")
        return json.loads(data)


def _payload(batch: int, rng: np.random.Generator, extra: dict[str, Any]) -> bytes:
    rows = rng.normal(size=(batch, len(FEATURES))).round(6)
    customers = [dict(zip(FEATURES, map(float, row), strict=True)) for row in rows]
    return json.dumps({"customers": customers, **extra}).encode()


def _summary(milliseconds: list[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    p50, p95 = np.percentile(values, [50, 95])
    return {
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "mean_ms": float(values.mean()),
        "max_ms": float(np.max(values)),
    }


def run_benchmark(
    endpoint: Endpoint, *, batch_sizes: Sequence[int], n_requests: int, warmup: int, seed: int
) -> dict[str, Any]:
    health = endpoint.request("GET", "/health")
    budgets = sorted(health["model"]["budgets"])
    decide_rule = {"budget": budgets[0]} if budgets else {"budget_fraction": 0.1}
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}
    for batch in batch_sizes:
        for path, extra in (("/score", {}), ("/decide", decide_rule)):
            bodies = [_payload(batch, rng, extra) for _ in range(warmup + n_requests)]
            timings = []
            for body in bodies:
                start = time.perf_counter()
                endpoint.request("POST", path, body)
                timings.append(1000.0 * (time.perf_counter() - start))
            results[f"{path.strip('/')}|batch_{batch}"] = _summary(timings[warmup:])
    return {"model": health["model"], "decide_rule": decide_rule, "results": results}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Latency benchmark for the scoring service.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", default="local", help="name of this run, e.g. local/docker")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 100])
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--token", default=None, help="bearer token (an authenticated Cloud Run)")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/serving-benchmark"))
    args = parser.parse_args(argv)
    if args.requests < 1 or args.warmup < 0 or min(args.batch_sizes) < 1:
        parser.error("--requests >= 1, --warmup >= 0 and batch sizes >= 1")
    try:
        endpoint = Endpoint(args.url, args.token, args.timeout)
        payload = run_benchmark(
            endpoint,
            batch_sizes=args.batch_sizes,
            n_requests=args.requests,
            warmup=args.warmup,
            seed=args.seed,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"benchmark failed against {args.url}: {error}") from error
    payload = {"label": args.label, "url": args.url, "n_requests": args.requests, **payload}
    path = write_json(args.output_dir / f"{args.label}.json", payload)
    for name, stats in payload["results"].items():
        print(
            f"{args.label} {name:18s} p50 {stats['p50_ms']:7.2f} ms  p95 {stats['p95_ms']:7.2f} ms"
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
