"""Device regression: loader restoration and repeated NZ prefetch preserve GEMM."""

from types import SimpleNamespace

import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.model_executor.offloader.loading import _prefetch_loading_context
from vllm_ascend.model_executor.offloader.prefetch import _as_pointer_tensor, _raw_memcpy_async


def test_loader_raw_nz_repeated_quant_matmul():
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(1024)
    cpu_weight = torch.randint(-4, 5, (64, 32), dtype=torch.int8, generator=generator)
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(cpu_weight, requires_grad=False)

    class Owner:
        _cpu_storage = cpu_weight

        @property
        def _param(self):
            return module.weight

    owner = Owner()
    offloader = SimpleNamespace(module_offloaders=[SimpleNamespace(_param_offloaders={"weight": owner})])
    inputs = torch.randint(-4, 5, (8, 64), dtype=torch.int8, generator=generator).npu()
    scale = torch.ones(32, dtype=torch.float32, device="npu")
    with _prefetch_loading_context(module, torch.device("npu:0"), offloader):
        module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
        reference = torch_npu.npu_quant_matmul(inputs, module.weight, scale, output_dtype=torch.bfloat16).cpu()
    host = module.weight._ascend_prefetch_nz_host
    buffer = torch_npu.npu_format_cast(torch.empty_like(cpu_weight, device="npu"), 29)
    for _ in range(3):
        _raw_memcpy_async(
            _as_pointer_tensor([host.data_ptr()]),
            _as_pointer_tensor([buffer.data_ptr()]),
            _as_pointer_tensor([host.numel() * host.element_size()]),
            0,
        )
        torch.npu.synchronize()
        actual = torch_npu.npu_quant_matmul(inputs, buffer, scale, output_dtype=torch.bfloat16).cpu()
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    assert owner._cpu_storage.is_pinned()
