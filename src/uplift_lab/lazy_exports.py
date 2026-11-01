"""Lazy package re-exports (PEP 562), so importing one submodule does not import its siblings.

``uplift_lab.uplift`` and ``uplift_lab.experiments`` re-export their public names for
convenience. Eager re-exports would make ``import uplift_lab.uplift.artifacts`` (what serving
needs) also import the training pipeline and scikit-learn, and ``uplift_lab.experiments.
estimators`` pull in statsmodels. With these hooks a name is imported from its submodule on first
access and then cached on the package, and the serving image can leave the analysis stack out.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def lazy_exports(
    package: str, exports: Mapping[str, Sequence[str]]
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """``(__getattr__, __dir__)`` for ``package``; ``exports`` maps submodule -> public names."""
    owners: dict[str, str] = {}
    for module, names in exports.items():
        for name in names:
            if name in owners:
                raise ValueError(f"{package}.{name} is exported by two submodules")
            owners[name] = f"{package}.{module}"

    def __getattr__(name: str) -> Any:
        owner = owners.get(name)
        if owner is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(owner), name)
        setattr(sys.modules[package], name, value)  # later lookups skip this hook
        return value

    def __dir__() -> list[str]:
        return sorted(set(vars(sys.modules[package])) | set(owners))

    return __getattr__, __dir__
