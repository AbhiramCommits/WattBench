"""Workload registry."""

from __future__ import annotations

from typing import Any

from .ai_inference import AiInferenceWorkload
from .base import Workload
from .gaming_proxy import GamingProxyWorkload
from .proviz_render import ProvizRenderWorkload

_REGISTRY: dict[str, type[Workload]] = {
    "ai_inference": AiInferenceWorkload,
    "proviz_render": ProvizRenderWorkload,
    "gaming_proxy": GamingProxyWorkload,
}


def available_workloads() -> list[str]:
    """Return the names of registered workloads."""
    return sorted(_REGISTRY)


def get_workload(name: str, **kwargs: Any) -> Workload:
    """Instantiate a workload by name."""
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown workload {name!r}; available: {', '.join(available_workloads())}"
        ) from exc
    return cls(**kwargs)


__all__ = ["Workload", "available_workloads", "get_workload"]
