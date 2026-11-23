"""Gather global/local block tables without specializing on batch shape."""

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.block_table import _load_ptr


@triton.jit(
    do_not_specialize=["num_blocks_stride", "global_reqs", "local_reqs", "global_pad"],
    do_not_specialize_on_alignment=["global_idx", "local_idx"],
)
def _fused_gather_kernel(
    global_idx,
    local_idx,
    src_ptrs,
    global_out_ptrs,
    local_out_ptrs,
    strides,
    num_blocks_ptr,
    num_blocks_stride,
    global_reqs,
    local_reqs,
    global_pad,
    BLOCK_SIZE: tl.constexpr,
):
    group = tl.program_id(0)
    combined_row = tl.program_id(1)
    if combined_row < global_pad:
        row = combined_row
        mapping = global_idx
        output_ptrs = global_out_ptrs
        num_reqs = global_reqs
    else:
        row = combined_row - global_pad
        mapping = local_idx
        output_ptrs = local_out_ptrs
        num_reqs = local_reqs

    stride = tl.load(strides + group)
    dst = _load_ptr(output_ptrs + group, tl.int32) + row * stride
    if row >= num_reqs:
        # Padded rows clear the full allocated width, just like upstream.
        for start in tl.range(0, stride, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            tl.store(dst + offsets, 0, mask=offsets < stride)
        return

    req_idx = tl.load(mapping + row)
    count = tl.load(num_blocks_ptr + group * num_blocks_stride + req_idx)
    src = _load_ptr(src_ptrs + group, tl.int32) + req_idx * stride
    for start in tl.range(0, count, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        values = tl.load(src + offsets, mask=offsets < count)
        # Preserve the original unwritten tail for real requests.
        tl.store(dst + offsets, values, mask=offsets < count)


def fused_gather(
    *, global_idx, local_idx, src_ptrs, global_out_ptrs, local_out_ptrs, strides, num_blocks, global_pad, local_pad
):
    groups = num_blocks.shape[0]
    if groups == 0 or global_pad + local_pad == 0:
        return
    _fused_gather_kernel[(groups, global_pad + local_pad)](
        global_idx,
        local_idx,
        src_ptrs,
        global_out_ptrs,
        local_out_ptrs,
        strides,
        num_blocks,
        num_blocks.stride(0),
        global_idx.numel(),
        local_idx.numel(),
        global_pad,
        BLOCK_SIZE=1024,
    )


def warmup_fused_gather(device: torch.device) -> None:
    """Load and launch the runtime-shape kernel before serving requests.

    Dummy graph capture uses the unfused path. Use separate tiny buffers here
    so warmup cannot modify real block tables or depend on request arrivals.
    Counts, padding and the count-table stride do not specialize the kernel;
    one launch therefore covers the normal int64-index/int32-table signature.
    """
    with torch.inference_mode():
        source = torch.zeros((1, 1), dtype=torch.int32, device=device)
        global_out = torch.empty((2, 1), dtype=torch.int32, device=device)
        local_out = torch.empty_like(global_out)
        indices = torch.zeros(1, dtype=torch.int64, device=device)
        strides = torch.ones(1, dtype=torch.int64, device=device)
        num_blocks = torch.ones((1, 1), dtype=torch.int32, device=device)
        src_ptrs = torch.tensor([source.data_ptr()], dtype=torch.uint64, device=device)
        global_ptrs = torch.tensor([global_out.data_ptr()], dtype=torch.uint64, device=device)
        local_ptrs = torch.tensor([local_out.data_ptr()], dtype=torch.uint64, device=device)
        fused_gather(
            global_idx=indices,
            local_idx=indices,
            src_ptrs=src_ptrs,
            global_out_ptrs=global_ptrs,
            local_out_ptrs=local_ptrs,
            strides=strides,
            num_blocks=num_blocks,
            global_pad=2,
            local_pad=2,
        )
        # The raw pointer tensors do not retain their pointees. Finish the
        # startup-only launch before any of the scratch buffers can be freed.
        torch.npu.current_stream(device).synchronize()
