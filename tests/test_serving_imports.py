"""The serving path runs without the analysis stack, and lazy package re-exports stay complete."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from serving_fixtures import customers, export_model, payload
from uplift_lab.lazy_exports import lazy_exports

# Installed by the analysis/report extras but absent from the serving image. scipy itself stays
# (LightGBM requires it); scipy.stats is the costly part to import.
ANALYSIS_ONLY = ("sklearn", "statsmodels", "pyarrow", "matplotlib", "scipy.stats")

SERVE_WITHOUT_ANALYSIS = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    blocked = sys.argv[1].split(",")
    sys.modules.update({name: None for name in blocked})  # any import of these now fails
    from fastapi.testclient import TestClient
    from uplift_lab.serving.app import create_app
    from uplift_lab.serving.settings import ServingSettings

    app = create_app(ServingSettings(model_dir=Path(sys.argv[2])))
    with TestClient(app) as client:
        body = json.loads(sys.argv[3])
        codes = [client.get("/health").status_code, client.post("/score", json=body).status_code]
        codes.append(client.post("/decide", json={**body, "budget": "0.1"}).status_code)
    print(json.dumps(codes))
    """
)


def test_serving_works_without_the_analysis_stack(tmp_path: Path) -> None:
    model_dir = export_model(tmp_path / "model")
    body = json.dumps(payload(customers(5)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            SERVE_WITHOUT_ANALYSIS,
            ",".join(ANALYSIS_ONLY),
            str(model_dir),
            body,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == [200, 200, 200]


@pytest.mark.parametrize("package", ["uplift_lab.uplift", "uplift_lab.experiments"])
def test_lazy_exports_resolve_every_public_name(package: str) -> None:
    module = importlib.import_module(package)
    for name in module.__all__:
        assert getattr(module, name) is not None
    assert set(module.__all__) <= set(dir(module))
    with pytest.raises(AttributeError, match="no attribute 'missing'"):
        _ = module.missing


def test_lazy_exports_reject_a_name_exported_twice() -> None:
    with pytest.raises(ValueError, match="exported by two submodules"):
        lazy_exports("pkg", {"a": ("x",), "b": ("x",)})
