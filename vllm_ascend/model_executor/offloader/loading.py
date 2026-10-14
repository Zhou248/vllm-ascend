"""Format-aware post-load CPU restoration for Ascend prefetch offloading.

The raw NZ transport follows upstream PR #13950. This adapter covers the
newer vLLM loader's empty_like(...).copy_ restoration as well.
"""

from contextlib import contextmanager
from functools import wraps
from typing import Any

import torch
from vllm.model_executor.offloader.base import get_offloader

from vllm_ascend.model_executor.offloader.prefetch import (
    AscendPrefetchOffloader,
    _capture_nz_host_storage,
    _get_nz_format,
    _raw_memcpy_available,
)


def _find_param_offloader(offloader: AscendPrefetchOffloader, parameter: Any) -> Any:
    # Resolve dotted names again: quantization can replace Parameter objects.
    for layer in offloader.module_offloaders:
        for candidate in layer._param_offloaders.values():
            try:
                if candidate._param is parameter:
                    return candidate
            except AttributeError:
                continue
    return None


def _raw_host_view(host: torch.Tensor, shape: tuple[int, ...], stride: tuple[int, ...]) -> torch.Tensor:
    if any(step < 0 for step in stride):
        raise ValueError("Negative strides are unsupported for raw NZ offload")
    required = (
        0 if any(size == 0 for size in shape) else 1 + sum((size - 1) * step for size, step in zip(shape, stride))
    )
    if required > host.numel():
        raise ValueError("Logical parameter view exceeds captured NZ storage")
    return host.as_strided(shape, stride)


def _release_offloaded_mxfp8_buffers(module: Any, device_pointers: set[int]) -> None:
    buffers = getattr(module, "_mxfp8_moe_buffers", None)
    if not isinstance(buffers, dict):
        return
    for name, tensors in list(buffers.items()):
        if any(tensor.data_ptr() in device_pointers for tensor in tensors):
            # Forward reads the parameters, not this RL-reload cache. Retaining
            # the old device tensors defeats offloading. Unselected entries stay.
            del buffers[name]


@contextmanager
def _prefetch_loading_context(module: Any, target_device: torch.device, offloader: AscendPrefetchOffloader):
    cpu_names = set()
    for name, parameter in module.named_parameters():
        if getattr(parameter, "_vllm_is_uva_offloaded", False):
            raise RuntimeError("Ascend raw NZ restoration cannot be combined with UVA parameters")
        if parameter.device.type == "cpu":
            cpu_names.add(name)
            parameter.data = parameter.data.to(target_device)
    try:
        yield module
    finally:
        device_pointers = set()
        for name, parameter in module.named_parameters():
            if name not in cpu_names:
                continue
            owner = _find_param_offloader(offloader, parameter)
            data = parameter.data
            nz_format = _get_nz_format(data)
            if nz_format is not None:
                if owner is None:
                    raise RuntimeError(f"NZ CPU parameter {name} is not owned by the prefetch offloader")
                if not _raw_memcpy_available():
                    raise RuntimeError("NZ weight offload requires the Ascend swap_blocks_batch custom op")
                host = _capture_nz_host_storage(data)
                view = _raw_host_view(host, tuple(data.shape), tuple(data.stride()))
                parameter._ascend_prefetch_nz_host = host
                parameter._ascend_prefetch_nz_format = nz_format
                parameter.data = view
            else:
                parameter.data = torch.empty_like(data, device="cpu", pin_memory=True).copy_(data)
            if owner is not None:
                device_pointers.add(data.data_ptr())
                # Release stale pre-quantization pinned storage layer by layer,
                # rather than keeping two complete host copies until post_init.
                owner._cpu_storage = parameter.data
        _release_offloaded_mxfp8_buffers(module, device_pointers)


def install_prefetch_loading_context() -> None:
    from vllm.model_executor.model_loader import utils as loader_utils

    original = loader_utils.device_loading_context
    if getattr(original, "_ascend_prefetch_loading_context", False):
        return

    @contextmanager
    @wraps(original)
    def context(module: Any, target_device: torch.device):
        offloader = get_offloader()
        if target_device.type == "cpu" or not isinstance(offloader, AscendPrefetchOffloader):
            with original(module, target_device) as value:
                yield value
            return
        with _prefetch_loading_context(module, target_device, offloader) as value:
            yield value

    context._ascend_prefetch_loading_context = True
    loader_utils.device_loading_context = context
