"""Ascend prefetch-based CPU offloading with NZ-format static buffers."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch_npu
from vllm.logger import logger
from vllm.model_executor.offloader.base import should_pin_memory
from vllm.model_executor.offloader.prefetch import (
    ParamInfo as VllmParamInfo,
)
from vllm.model_executor.offloader.prefetch import (
    PrefetchOffloader,
    StaticBufferPool,
)
from vllm.model_executor.offloader.prefetch import (
    _ModuleOffloader as VllmModuleOffloader,
)

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, bootstrap_custom_op_env

# FRACTAL_NZ variants with an explicit C0 size, used by newer SoCs.
ACL_FORMAT_FRACTAL_NZ_C0_16 = 50
ACL_FORMAT_FRACTAL_NZ_C0_32 = 51
ACL_FORMAT_FRACTAL_NZ_C0_2 = 52
ACL_FORMAT_FRACTAL_NZ_C0_4 = 53
ACL_FORMAT_FRACTAL_NZ_C0_8 = 54

# Storage formats whose physical layout is padded and reordered with respect to
# the logical shape. torch_npu cannot transfer them between host and device
# directly, so every copy_() against such a tensor also runs a TransData.
_NZ_FORMATS = frozenset(
    {
        ACL_FORMAT_FRACTAL_NZ,  # 29
        ACL_FORMAT_FRACTAL_NZ_C0_16,
        ACL_FORMAT_FRACTAL_NZ_C0_32,
        ACL_FORMAT_FRACTAL_NZ_C0_2,
        ACL_FORMAT_FRACTAL_NZ_C0_4,
        ACL_FORMAT_FRACTAL_NZ_C0_8,
    }
)

# `direction` argument of torch.ops._C_ascend.swap_blocks_batch.
_MEMCPY_HOST_TO_DEVICE = 0
_MEMCPY_DEVICE_TO_HOST = 1


def _get_nz_format(tensor: torch.Tensor) -> int | None:
    """Return the NZ storage format of an NPU tensor, or None if it is not NZ."""
    if tensor.device.type != "npu":
        return None

    try:
        acl_format = int(torch_npu.get_npu_format(tensor))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None

    return acl_format if acl_format in _NZ_FORMATS else None


def _is_using_nz_weight(param: nn.Parameter) -> bool:
    """Compatibility helper used by the existing offloader regressions."""
    return _get_nz_format(param.data) is not None


def _raw_memcpy_available() -> bool:
    """Check this transport op independently of generic compute-op gating.

    A5 disables some custom compute kernels, but its tested pointer-level
    transport remains available. Load lazily, after device initialization.
    """
    if not hasattr(torch.ops._C_ascend, "swap_blocks_batch"):
        try:
            bootstrap_custom_op_env()
            import vllm_ascend.vllm_ascend_C  # noqa: F401
        except (ImportError, RuntimeError):
            return False
    return hasattr(torch.ops._C_ascend, "swap_blocks_batch")


def _as_pointer_tensor(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device="cpu")


def _raw_memcpy_async(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    sizes: torch.Tensor,
    direction: int,
) -> None:
    """Enqueue pointer-level memcpys on the current NPU stream.

    The copy is issued from raw addresses and byte counts, so it moves the
    padded NZ storage verbatim instead of going through the format-aware
    torch_npu copy path.
    """
    torch.ops._C_ascend.swap_blocks_batch(src_ptrs, dst_ptrs, sizes, direction)


def _capture_nz_host_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Copy the padded NZ storage of a device tensor into host memory."""
    storage_numel = int(torch_npu.get_storage_size(tensor))
    host_storage = torch.empty(
        storage_numel,
        dtype=tensor.dtype,
        device="cpu",
        pin_memory=should_pin_memory(),
    )
    _raw_memcpy_async(
        _as_pointer_tensor([tensor.data_ptr()]),
        _as_pointer_tensor([host_storage.data_ptr()]),
        _as_pointer_tensor([storage_numel * tensor.element_size()]),
        _MEMCPY_DEVICE_TO_HOST,
    )
    torch.npu.current_stream().synchronize()
    return host_storage


@contextmanager
def _patched_vllm_prefetch(**attrs: Any) -> Generator[None, None, None]:
    """Temporarily swap symbols that the vLLM offloader resolves at call time."""
    import vllm.model_executor.offloader.prefetch as vllm_prefetch

    originals = {name: getattr(vllm_prefetch, name) for name in attrs}
    for name, value in attrs.items():
        setattr(vllm_prefetch, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(vllm_prefetch, name, value)


@dataclass
class ParamInfo(VllmParamInfo):
    """Ascend parameter metadata with static-buffer format requirements."""

    nz_format: int | None = None


def _collect_nz_formats(param_infos: list[ParamInfo]) -> dict[tuple, int]:
    """Map each buffer-pool key to the NZ format its parameters require."""
    key_to_format: dict[tuple, int | None] = {}

    for info in param_infos:
        if info.key in key_to_format and key_to_format[info.key] != info.nz_format:
            raise ValueError(
                "Conflicting NZ buffer requirements for prefetch static buffer "
                f"key {info.key}: {key_to_format[info.key]} and {info.nz_format} "
                "are both requested."
            )
        key_to_format[info.key] = info.nz_format

    return {key: acl_format for key, acl_format in key_to_format.items() if acl_format is not None}


class AscendStaticBufferPool(StaticBufferPool):
    """Static buffer pool whose slots use the NZ format of their parameters.

    The cast has to happen while the pool is being built: once
    ``assign_buffer_slot`` has pointed the parameters at their buffers,
    replacing the pooled tensors no longer has any effect.
    """

    def __init__(
        self,
        param_infos: list[ParamInfo],
        slot_capacity: int,
        device: torch.device,
    ):
        super().__init__(param_infos=param_infos, slot_capacity=slot_capacity, device=device)

        nz_formats = _collect_nz_formats(param_infos)
        for key, acl_format in nz_formats.items():
            nd_buffers = self._buffers[key]
            self._buffers[key] = [torch_npu.npu_format_cast(buffer, acl_format) for buffer in nd_buffers]
            for nd_buffer, nz_buffer in zip(nd_buffers, self._buffers[key]):
                padding_numel = int(torch_npu.get_storage_size(nz_buffer)) - nd_buffer.numel()
                self.total_bytes += padding_numel * nd_buffer.element_size()

        if nz_formats:
            logger.info(
                "[AscendPrefetchOffloader] %d of %d static buffer keys use NZ format",
                len(nz_formats),
                len(self._buffers),
            )


class AscendPrefetchOffloader(PrefetchOffloader):
    """Ascend prefetch offloader that reuses vLLM behavior with NZ buffers."""

    def __init__(
        self,
        group_size: int,
        num_in_group: int,
        prefetch_step: int,
        offload_params: set[str] | None = None,
        mode: str = "cpu",
    ):
        super().__init__(
            group_size=group_size,
            num_in_group=num_in_group,
            prefetch_step=prefetch_step,
            offload_params=offload_params,
            mode=mode,
        )
        self.module_offloaders: list[_ModuleOffloader] = []

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        with _patched_vllm_prefetch(_ModuleOffloader=_ModuleOffloader):
            return super().wrap_modules(modules_generator)

    def post_init(self):
        from vllm.config import get_current_vllm_config

        # Quantization can delete transient parameters. Prune them before the
        # format guard resolves names; the upstream post_init sync is idempotent.
        for module_offloader in self.module_offloaders:
            module_offloader.sync_cpu_storage()
        has_nz = any(
            getattr(param_offloader._param, "_ascend_prefetch_nz_host", None) is not None
            for module_offloader in self.module_offloaders
            for param_offloader in module_offloader._param_offloaders.values()
        )
        if has_nz and not get_current_vllm_config().model_config.enforce_eager:
            raise RuntimeError("Raw NZ weight offload is validated only in eager mode; use enforce_eager=True")
        with _patched_vllm_prefetch(StaticBufferPool=AscendStaticBufferPool):
            super().post_init()


class _ModuleOffloader(VllmModuleOffloader):
    """vLLM module offloader that prefetches NZ weights as raw bytes."""

    def __init__(
        self,
        mode: str,
        module: nn.Module,
        copy_stream: torch.cuda.Stream,
        whitelist_param_names: list[str],
        layer_idx: int,
    ):
        super().__init__(
            mode=mode,
            module=module,
            copy_stream=copy_stream,
            whitelist_param_names=whitelist_param_names,
            layer_idx=layer_idx,
        )
        # NZ storage format per parameter name, needed to allocate matching
        # static buffers.
        self._nz_formats: dict[str, int] = {}
        # Padded NZ storage per parameter <name, mirrored> in host memory.
        self._nz_host_storages: dict[str, torch.Tensor] = {}
        # Prefetch plan, built in post_init().
        self._nz_src_ptrs = _as_pointer_tensor([])
        self._nz_dst_ptrs = _as_pointer_tensor([])
        self._nz_sizes = _as_pointer_tensor([])
        self._nd_copies: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def get_param_infos(self) -> list[ParamInfo]:
        infos = []
        for name, offloader in self._param_offloaders.items():
            param = offloader._param
            host = getattr(param, "_ascend_prefetch_nz_host", None)
            if host is not None:
                self._nz_host_storages[name] = host
                self._nz_formats[name] = param._ascend_prefetch_nz_format
            cpu_storage = offloader._cpu_storage
            assert cpu_storage is not None, "CPU storage not initialized"
            infos.append(
                ParamInfo(
                    name=name,
                    shape=tuple(cpu_storage.shape),
                    stride=tuple(cpu_storage.stride()),
                    dtype=cpu_storage.dtype,
                    # NOTE(wangjin) different from base
                    nz_format=self._nz_formats.get(name),
                )
            )
        return infos

    def post_init(self):
        """Build the prefetch plan once the static buffers are assigned."""
        super().post_init()

        src_ptrs: list[int] = []
        dst_ptrs: list[int] = []
        sizes: list[int] = []

        for name, offloader in self._param_offloaders.items():
            gpu_buffer = offloader._gpu_buffer
            assert gpu_buffer is not None, f"GPU buffer for {name} not assigned"

            host_storage = self._nz_host_storages.get(name)
            if host_storage is None:
                cpu_storage = offloader._cpu_storage
                assert cpu_storage is not None, f"CPU storage for {name} not initialized"
                self._nd_copies.append((name, cpu_storage, gpu_buffer))
                continue

            buffer_numel = int(torch_npu.get_storage_size(gpu_buffer))
            if buffer_numel != host_storage.numel():
                raise RuntimeError(
                    f"NZ storage size mismatch for {name}: static buffer holds "
                    f"{buffer_numel} elements but the captured weight holds "
                    f"{host_storage.numel()}. The static buffer layout does not "
                    "match the weight produced by process_weights_after_loading."
                )

            src_ptrs.append(host_storage.data_ptr())
            dst_ptrs.append(gpu_buffer.data_ptr())
            sizes.append(host_storage.numel() * host_storage.element_size())
            # The ND host copy is superseded by the raw NZ bytes.
            offloader._cpu_storage = None

        self._nz_src_ptrs = _as_pointer_tensor(src_ptrs)
        self._nz_dst_ptrs = _as_pointer_tensor(dst_ptrs)
        self._nz_sizes = _as_pointer_tensor(sizes)

    def start_onload_to_static(self):
        """Start async copy from host storage to the static NPU buffers.

        Follows the same fork/record protocol as the base implementation; only
        the copy itself differs. NZ weights are transferred by a single
        pointer-level memcpy of their padded storage, which avoids the
        ND -> NZ TransData that torch_npu inserts for every H2D copy_ into a
        non-base-format tensor.
        """
        assert self._buffer_pool is not None, "Buffer pool not assigned"

        self._prefetch_in_capture = torch.cuda.is_current_stream_capturing()

        fork_event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(fork_event)
        self.copy_stream.wait_event(fork_event)

        with torch.cuda.stream(self.copy_stream):
            if self._nz_sizes.numel():
                _raw_memcpy_async(
                    self._nz_src_ptrs,
                    self._nz_dst_ptrs,
                    self._nz_sizes,
                    _MEMCPY_HOST_TO_DEVICE,
                )
            for name, cpu_storage, gpu_buffer in self._nd_copies:
                assert not should_pin_memory() or cpu_storage.is_pinned(), (
                    f"CPU storage for {name} is not pinned! "
                    "non_blocking=True H2D copy from non-pinned memory "
                    "causes stream synchronization that breaks "
                    "event-based fork synchronization."
                )
                gpu_buffer.copy_(cpu_storage, non_blocking=True)

        self._copy_done_event.record(self.copy_stream)
        self._event_valid_for_eager = not torch.cuda.is_current_stream_capturing()
