# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Pure accounting for stage resources allocated before memory profiling."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SawtoothMemoryBudget:
    legacy_non_kv: int
    total_consumed: int
    transient_peak_headroom: int
    non_kv: int

    @property
    def persistent_adjustment(self) -> int:
        return self.non_kv - self.legacy_non_kv


def sawtooth_memory_budget(profile: Any, profile_torch_peak: int) -> SawtoothMemoryBudget:
    """Include persistent allocations without adding weights/buffers twice.

    All fields are bytes from MemoryProfilingResult. The explicit peak is taken
    before graph capture; graphs belong to the subsequent warmup accounting.
    Retain the legacy estimate as a floor if device free-memory measurements
    under-report consumption. Older profiling results can reconstruct the total
    from the same snapshots instead of losing persistent allocations.
    """
    legacy = int(profile.non_torch_increase + profile.torch_peak_increase + profile.weights_memory)
    total = getattr(profile, "total_consumed", None)
    if total is None:
        total = profile.before_create.free_memory - profile.after_profile.free_memory
    total = int(total)
    headroom = max(0, int(profile_torch_peak) - int(profile.after_profile.torch_allocated))
    return SawtoothMemoryBudget(legacy, total, headroom, max(legacy, total + headroom))


def sawtooth_warmup_non_kv(profile_non_kv: int, graph_memory: int) -> int:
    """The saved pre-graph budget already includes all persistent resources."""
    return profile_non_kv + graph_memory
