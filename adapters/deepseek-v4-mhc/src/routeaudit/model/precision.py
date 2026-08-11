"""Architecture-neutral checkpoint precision helpers."""

from __future__ import annotations

from typing import Optional

_NATIVE_LOW_PRECISION = (
    "fp8", "float8", "e4m3", "fp4", "float4", "e2m1", "nvfp4", "mxfp4",
)


def native_precision(model_ns) -> Optional[str]:
    """Return a declared native low-precision dtype, if present."""

    for key in ("expert_dtype", "weights_dtype", "dtype"):
        value = getattr(model_ns, key, None)
        if isinstance(value, str) and any(token in value.lower() for token in _NATIVE_LOW_PRECISION):
            return value
    return None
