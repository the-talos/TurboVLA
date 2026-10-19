"""Device selection shared by the inference path (XPU > CUDA > CPU)."""

from __future__ import annotations

import warnings
from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")
_warned: set[str] = set()


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in _ACCELERATORS:
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve ``None`` by detection. An explicit device is honoured verbatim."""
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device(default_device_type())
    return torch.device(device)


def seed_accelerators(seed: int) -> None:
    """Seed every available accelerator RNG."""
    if is_available("cuda"):
        torch.cuda.manual_seed_all(seed)
    if is_available("xpu"):
        torch.xpu.manual_seed_all(seed)


def autocast_device_type(device: Union[torch.device, str]) -> Optional[str]:
    """Backend to pass to ``torch.autocast`` for ``device``, or ``None`` on CPU."""
    device_type = torch.device(device).type
    return device_type if device_type in _ACCELERATORS else None


def resolve_attention_implementation(implementation: Optional[str]) -> Optional[str]:
    """Replace ``flash_attention_2`` by ``sdpa`` when CUDA is not available."""
    if implementation == "flash_attention_2" and not is_available("cuda"):
        if "flash_attention_2" not in _warned:
            _warned.add("flash_attention_2")
            warnings.warn(
                "flash_attention_2 needs CUDA; falling back to sdpa attention.",
                stacklevel=2,
            )
        return "sdpa"
    return implementation
