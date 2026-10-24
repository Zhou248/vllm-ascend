# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

from vllm_ascend.worker.v2.sawtooth_memory import sawtooth_memory_budget, sawtooth_warmup_non_kv


def profile(*, total=58, allocated=55, peak_increase=4):
    # A weight-only estimate is 40 + 3 + 4 = 47. The current torch
    # allocation also includes 15 persistent units created after load metering.
    return SimpleNamespace(
        weights_memory=40,
        non_torch_increase=3,
        torch_peak_increase=peak_increase,
        total_consumed=total,
        before_create=SimpleNamespace(free_memory=64),
        after_profile=SimpleNamespace(free_memory=64 - total, torch_allocated=allocated, torch_peak=999),
    )


def test_persistent_stage_buffers_are_charged_before_kv_allocation():
    result = sawtooth_memory_budget(profile(), profile_torch_peak=59)
    assert result.legacy_non_kv == 47
    assert result.transient_peak_headroom == 4
    assert result.non_kv == 62
    assert result.persistent_adjustment == 15


def test_legacy_estimate_is_floor_when_free_memory_measurement_is_lower():
    result = sawtooth_memory_budget(profile(total=35), profile_torch_peak=59)
    assert result.non_kv == 47
    assert result.persistent_adjustment == 0


def test_negative_transient_headroom_never_releases_persistent_budget():
    result = sawtooth_memory_budget(profile(allocated=61), profile_torch_peak=59)
    assert result.transient_peak_headroom == 0
    assert result.non_kv == 58


def test_older_profile_reconstructs_consumption_from_free_memory_snapshots():
    data = profile()
    del data.total_consumed
    result = sawtooth_memory_budget(data, profile_torch_peak=59)
    assert result.total_consumed == 58
    assert result.non_kv == 62


def test_profile_peak_and_weights_are_not_added_to_total_twice():
    result = sawtooth_memory_budget(profile(allocated=59), profile_torch_peak=59)
    assert result.non_kv == 58
    # after_profile.torch_peak=999 must not replace the pre-graph peak.
    assert result.transient_peak_headroom == 0


def test_warmup_adds_graph_once_to_saved_profile_base():
    result = sawtooth_memory_budget(profile(), profile_torch_peak=59)
    graph_memory = 3
    non_kv = sawtooth_warmup_non_kv(result.non_kv, graph_memory)
    assert non_kv == 65
    requested, margin = 80, 1
    assert requested - non_kv - margin == 14
    # Eager uses the same persistent budget, with no graph addition.
    assert sawtooth_warmup_non_kv(result.non_kv, 0) == result.non_kv
