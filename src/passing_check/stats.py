"""The one quantile convention and the one seeded generator.

    median = statistics.median   (mean of the two middle values for even n)
    q(p)   = sorted(values)[min(floor(p * n), n - 1)]   -- lower nearest rank, no interpolation

Both are exact functions of the sorted input, so a recomputation matches to the last digit.
Provenance: UPSTREAM.md.
"""
import math
import statistics

import numpy as np


def quantile(values, p: float) -> float:
    if not values:
        raise ValueError("quantile of an empty sequence; refusing to invent a number")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"quantile probability must be in [0, 1], got {p}")
    s = sorted(values)
    return s[min(int(math.floor(p * len(s))), len(s) - 1)]


def summary(values) -> dict:
    """{n, median, p10, p90} of a non-empty sequence; refuses (never NaN) when empty."""
    if not values:
        raise ValueError("summary of an empty sequence; refusing to invent a number")
    return {"n": len(values), "median": statistics.median(values),
            "p10": quantile(values, 0.10), "p90": quantile(values, 0.90)}


def make_rng(seed: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int from config, got {type(seed).__name__}")
    return np.random.default_rng(seed)
