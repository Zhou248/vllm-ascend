# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache metadata and execution backends for the GLM-Next pooled indexer."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from vllm_ascend.attention.context_parallel.common_cp import get_cp_local_query_key_lens
from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)
from vllm_ascend.device.hardware_profile import AttentionBackendFamily, get_current_hardware_profile
from vllm_ascend.utils import _round_up, enable_dsa_cp

GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE = 128

logger = init_logger(__name__)


@dataclass
class AscendIndexerKPoolQueryMetadata:
    """Token-local query addressing over the replicated pool cache."""

    positions: torch.Tensor
    cum_query_lens: torch.Tensor
    num_actual_tokens: int
    seq_lens: torch.Tensor | None = None
    pool_tail: torch.Tensor | None = None


@dataclass
class AscendIndexerKPoolMetadata:
    """Metadata for compressed indexer cache writes and top-k reads."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    block_size: int
    compress_ratio: int
    cache_role: str = "indexer"
    cum_query_lens: torch.Tensor | None = None
    raw_seq_lens: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    num_actual_tokens: int = 0
    query_metadata: AscendIndexerKPoolQueryMetadata | None = None
    query_start_loc: torch.Tensor | None = None
    start_pos: torch.Tensor | None = None
    pool_tail: torch.Tensor | None = None
    keypool_state_table: torch.Tensor | None = None
    keypool_start_offsets: torch.Tensor | None = None
    pooled_slot_mapping: torch.Tensor | None = None


@dataclass
class _KPoolMetadataBuffers:
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    cum_query_lens: torch.Tensor
    raw_seq_lens: torch.Tensor
    positions: torch.Tensor
    query_start_loc: torch.Tensor
    start_pos: torch.Tensor
    pool_tail: torch.Tensor
    start_offsets: torch.Tensor
    token_indices: torch.Tensor
    pooled_slot_mapping: torch.Tensor


class AscendIndexerKPoolMetadataBuilder(AttentionMetadataBuilder):
    """Build pool-level addressing for the compressed indexer cache."""

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        # This cache-only builder still participates in graph capability
        # reduction. Its decode metadata uses persistent buffers refreshed in
        # place, so it must not disable the main model's uniform decode graph.
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        if not isinstance(kv_cache_spec, MLAAttentionSpec):
            raise TypeError(
                f"Ascend Indexer KPool backend requires MLAAttentionSpec, got {type(kv_cache_spec).__name__}."
            )
        compress_ratio = get_kv_cache_compression_ratio(kv_cache_spec)
        if compress_ratio <= 1:
            raise ValueError(f"Ascend Indexer KPool cache requires compress_ratio > 1, got {compress_ratio}.")
        if not layer_names or any(not name.endswith(".indexer.k_cache") for name in layer_names):
            raise ValueError(f"Invalid Indexer KPool cache layer names: {layer_names}.")
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.logical_block_size = vllm_config.cache_config.block_size
        self.storage_block_size = get_storage_block_size(kv_cache_spec)
        if self.storage_block_size <= 0:
            raise ValueError(f"Indexer KPool storage block size must be positive, got {self.storage_block_size}.")
        self.compress_ratio = compress_ratio
        if self.logical_block_size % GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE:
            raise ValueError(
                "GLM-Next logical block size must be divisible by the SFA "
                f"kernel block size: logical={self.logical_block_size}, "
                f"kernel={GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE}."
            )
        self.kernel_row_block_size = GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE // self.compress_ratio
        # MTP tables count expanded kernel pages.
        self.kernel_blocks_per_logical_block = self.logical_block_size // GLM5_NEXT_SFA_KERNEL_BLOCK_SIZE
        scheduler_config = vllm_config.scheduler_config
        self.use_dsa_cp = enable_dsa_cp()
        self.dsa_cp_size = get_tp_group().world_size if self.use_dsa_cp else 1
        self._max_num_batched_tokens = _round_up(scheduler_config.max_num_batched_tokens, self.dsa_cp_size)
        # FULL draft graphs can pad requests beyond max_num_seqs.
        self._max_num_metadata_reqs = max(scheduler_config.max_num_seqs, self._max_num_batched_tokens)
        # Keep captured storage isolated between MTP draft steps.
        self._metadata_buffers: dict[int, _KPoolMetadataBuffers] = {}
        self._dsa_cp_query_buffers: dict[int, torch.Tensor] = {}
        self._captured_state_tables: dict[tuple[int, int, int], torch.Tensor] = {}
        self._state_table_scratch: dict[int, torch.Tensor] = {}
        self._state_table_shapes: dict[int, tuple[int, int]] = {}

    def _build_query_metadata(
        self, common: CommonAttentionMetadata, positions: torch.Tensor
    ) -> AscendIndexerKPoolQueryMetadata | None:
        if not self.use_dsa_cp:
            return None
        local_tokens = positions.shape[0] // self.dsa_cp_size
        local_start = get_tp_group().rank_in_group * local_tokens
        local_end = local_start + local_tokens
        key = common.slot_mapping.data_ptr()
        if key not in self._dsa_cp_query_buffers:
            self._dsa_cp_query_buffers[key] = torch.empty(
                (3, self._max_num_metadata_reqs), dtype=torch.int64, device=self.device
            )
        query_lens, pool_lens, pool_tail = self._dsa_cp_query_buffers[key][:, : common.num_reqs]
        local_query_lens, local_key_lens = get_cp_local_query_key_lens(
            common.query_start_loc,
            common.query_start_loc[1 : common.num_reqs + 1],
            common.seq_lens[: common.num_reqs],
            local_start,
            local_end,
        )
        query_lens.copy_(local_query_lens)
        torch.div(local_key_lens, self.compress_ratio, rounding_mode="floor", out=pool_lens)
        torch.remainder(local_key_lens, self.compress_ratio, out=pool_tail)
        return AscendIndexerKPoolQueryMetadata(
            positions=positions[local_start:local_end],
            cum_query_lens=query_lens,
            seq_lens=pool_lens,
            pool_tail=pool_tail,
            num_actual_tokens=max(min(local_end, common.num_actual_tokens) - local_start, 0),
        )

    def _get_metadata_buffers(self, common_attn_metadata: CommonAttentionMetadata) -> _KPoolMetadataBuffers:
        key = common_attn_metadata.slot_mapping.data_ptr()
        buffers = self._metadata_buffers.get(key)
        if buffers is None:
            reqs, tokens = self._max_num_metadata_reqs, self._max_num_batched_tokens
            buffers = _KPoolMetadataBuffers(
                slot_mapping=torch.empty(tokens, dtype=torch.int64, device=self.device),
                seq_lens=torch.empty(reqs, dtype=torch.int64, device=self.device),
                cum_query_lens=torch.empty(reqs, dtype=torch.int64, device=self.device),
                raw_seq_lens=torch.empty(reqs, dtype=torch.int32, device=self.device),
                positions=torch.empty(tokens, dtype=torch.int64, device=self.device),
                query_start_loc=torch.empty(reqs + 1, dtype=torch.int32, device=self.device),
                start_pos=torch.empty(reqs, dtype=torch.int32, device=self.device),
                pool_tail=torch.empty(reqs, dtype=torch.int64, device=self.device),
                start_offsets=torch.empty(reqs, dtype=torch.int32, device=self.device),
                token_indices=torch.arange(tokens, dtype=torch.int32, device=self.device),
                pooled_slot_mapping=torch.empty(tokens + 1, dtype=torch.int64, device=self.device),
            )
            self._metadata_buffers[key] = buffers
        return buffers

    def _get_state_table(self, common: CommonAttentionMetadata, graph_capture: bool) -> torch.Tensor:
        # Capture owns each table; eager shapes reuse one growing allocation.
        step = common.slot_mapping.data_ptr()
        width = getattr(common, "max_query_len", common.num_input_tokens)
        cols = 1 + (width + self.compress_ratio - 1) // self.compress_ratio
        shape = (common.num_reqs, cols)
        key = (step, *shape)
        table = self._captured_state_tables.get(key)
        initialize = table is None
        if table is None:
            if graph_capture:
                table = torch.empty(shape, dtype=torch.int32, device=self.device)
                self._captured_state_tables[key] = table
            else:
                size = common.num_reqs * cols
                scratch = self._state_table_scratch.get(step)
                if scratch is None or scratch.numel() < size:
                    scratch = torch.empty(size, dtype=torch.int32, device=self.device)
                    self._state_table_scratch[step] = scratch
                table = scratch[:size].view(shape)
                initialize = self._state_table_shapes.get(step) != shape
                self._state_table_shapes[step] = shape
        if initialize:
            # This table depends only on shape, so initialize it once.
            host = torch.empty(shape, dtype=torch.int32, pin_memory=self.device.type != "cpu")
            rows = torch.arange(1, common.num_reqs + 1, dtype=torch.int32)
            host.copy_((rows + common.num_reqs)[:, None].expand(shape))
            host[:, 0].copy_(rows)
            table.copy_(host, non_blocking=True)
        return table

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendIndexerKPoolMetadata:
        graph_capture = kwargs.pop("_graph_capture", False)
        del common_prefix_len, fast_build, kwargs
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        buffers = self._get_metadata_buffers(common_attn_metadata)
        num_cache_tokens = _round_up(num_input_tokens, self.dsa_cp_size)
        num_source_tokens = min(
            num_input_tokens, common_attn_metadata.positions.shape[0], common_attn_metadata.slot_mapping.shape[0]
        )
        if num_source_tokens < common_attn_metadata.num_actual_tokens:
            raise ValueError("KPool positions and slot mapping must cover every actual token.")
        from vllm_ascend.ops.cann_keypool import build_keypool_metadata

        state_table = None if self.use_dsa_cp else self._get_state_table(common_attn_metadata, graph_capture)
        (
            positions,
            slot_mapping,
            seq_lens,
            raw_seq_lens,
            query_start_loc,
            cum_query_lens,
            start_pos,
            pool_tail,
        ) = build_keypool_metadata(
            common_attn_metadata,
            buffers,
            num_cache_tokens,
            num_source_tokens,
            self.logical_block_size,
            self.compress_ratio,
        )
        seq_lens_cpu = None
        if self.use_dsa_cp:
            raw_seq_lens_cpu = getattr(common_attn_metadata, "_seq_lens_cpu", None)
            if raw_seq_lens_cpu is not None:
                seq_lens_cpu = raw_seq_lens_cpu[:num_reqs] // self.compress_ratio
        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        return AscendIndexerKPoolMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            positions=positions,
            block_size=self.kernel_row_block_size,
            compress_ratio=self.compress_ratio,
            cum_query_lens=cum_query_lens,
            raw_seq_lens=raw_seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_start_loc=query_start_loc,
            start_pos=start_pos,
            pool_tail=pool_tail,
            pooled_slot_mapping=buffers.pooled_slot_mapping[
                1 : 1 + min(num_cache_tokens, num_cache_tokens // self.compress_ratio + num_reqs)
            ],
            keypool_state_table=state_table,
            keypool_start_offsets=buffers.start_offsets[:num_reqs] if state_table is not None else None,
            query_metadata=self._build_query_metadata(common_attn_metadata, positions),
        )

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata) -> AscendIndexerKPoolMetadata:
        return self.build_for_graph_capture(common_attn_metadata)

    def build_for_graph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        attn_state: Any = None,
        **kwargs,
    ) -> AscendIndexerKPoolMetadata:
        del attn_state
        return self.build(0, common_attn_metadata, _graph_capture=True, **kwargs)

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
        **kwargs,
    ) -> AscendIndexerKPoolMetadata:
        del draft_index
        return self.build(0, common_attn_metadata, fast_build=True, **kwargs)


class AscendIndexerKPoolBackend(AttentionBackend):
    """Cache-only backend for the compressed indexer keys."""

    @staticmethod
    def get_impl_cls():
        return None

    @staticmethod
    def get_name() -> str:
        return "ASCEND_INDEXER_KPOOL"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type[AscendIndexerKPoolMetadataBuilder]:
        return AscendIndexerKPoolMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_type, cache_dtype_str
        if num_kv_heads != 1:
            raise ValueError(f"Indexer KPool cache requires one KV head, got {num_kv_heads}.")
        return (num_blocks, block_size, num_kv_heads, head_size)


@dataclass
class AscendIndexerKPoolTailMetadata:
    """Addressing required to update the compressor tail cache."""

    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int
    cann_inputs: Any = None


class AscendIndexerKPoolTailMetadataBuilder(AttentionMetadataBuilder):
    """Build independent metadata for the GLM-Next compressor tail."""

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        # Full-graph tail writes use the fixed-shape sentinel path. Do not let
        # the base class default NEVER downgrade FULL_DECODE_ONLY for the main
        # model merely because this cache-only builder is in the cache group.
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AscendIndexerKPoolTailSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        if not isinstance(kv_cache_spec, AscendIndexerKPoolTailSpec):
            raise TypeError(
                "Ascend Indexer KPool tail backend requires "
                f"AscendIndexerKPoolTailSpec, got {type(kv_cache_spec).__name__}."
            )
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self.use_dsa_cp = enable_dsa_cp()
        self.dsa_cp_size = get_tp_group().world_size if self.use_dsa_cp else 1
        self._max_num_batched_tokens = _round_up(vllm_config.scheduler_config.max_num_batched_tokens, self.dsa_cp_size)
        self._slot_buffers: dict[int, torch.Tensor] = {}
        # Persistent rings are separate from temporary CANN state pages.
        self._max_num_metadata_reqs = max(
            vllm_config.scheduler_config.max_num_seqs, vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._block_table_buffers: dict[int, torch.Tensor] = {}
        self._tail_inputs: dict[int, dict[str, torch.Tensor]] = {}
        self._pool_size = kv_cache_spec.compress_ratio
        self._page_rows = kv_cache_spec.page_size_bytes // (2 * kv_cache_spec.head_size * 4)
        self._lookahead = self.block_size > self._pool_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendIndexerKPoolTailMetadata:
        del common_prefix_len, fast_build, kwargs
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        key = common_attn_metadata.slot_mapping.data_ptr()
        if key not in self._block_table_buffers:
            self._block_table_buffers[key] = torch.empty(
                self._max_num_metadata_reqs, 1, dtype=torch.int32, device=self.device
            )
        block_table = self._block_table_buffers[key][:num_reqs]
        block_table.copy_(common_attn_metadata.block_table_tensor[:num_reqs, :1])
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        if self.use_dsa_cp:
            # Match the gathered K/gate rows, including TP alignment padding.
            key = common_attn_metadata.slot_mapping.data_ptr()
            if key not in self._slot_buffers:
                self._slot_buffers[key] = torch.empty(
                    self._max_num_batched_tokens, dtype=slot_mapping.dtype, device=self.device
                )
            padded_tokens = _round_up(num_input_tokens, self.dsa_cp_size)
            slots = self._slot_buffers[key][:padded_tokens]
            slots[:num_input_tokens].copy_(slot_mapping)
            slots[num_input_tokens:].fill_(-1)
            slot_mapping = slots
        cann_inputs = None
        if not self.use_dsa_cp:
            from vllm_ascend.ops.cann_keypool import build_tail_inputs

            if key not in self._tail_inputs:
                n, width = self._max_num_metadata_reqs, self.block_size
                self._tail_inputs[key] = {
                    "history": torch.empty(n, self._pool_size, dtype=torch.int64, device=self.device),
                    **{
                        name: torch.empty(n, width, dtype=torch.int64, device=self.device)
                        for name in ("writes", "state", "retained")
                    },
                    "columns": torch.arange(width, device=self.device),
                    "requests": torch.arange(n, device=self.device),
                }
            retained_width = (
                min(self.block_size, getattr(common_attn_metadata, "max_query_len", self.block_size))
                if self._lookahead
                else 0
            )
            cann_inputs = build_tail_inputs(
                common_attn_metadata,
                self._pool_size,
                self.block_size,
                self._page_rows,
                max(1, retained_width) if self._lookahead else 0,
                self._tail_inputs[key],
            )
        return AscendIndexerKPoolTailMetadata(
            block_table=block_table,
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            cann_inputs=cann_inputs,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        attn_state: Any = None,
        **kwargs,
    ) -> AscendIndexerKPoolTailMetadata:
        del attn_state
        return self.build(0, common_attn_metadata, **kwargs)

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
        **kwargs,
    ) -> AscendIndexerKPoolTailMetadata:
        del draft_index
        return self.build(0, common_attn_metadata, fast_build=True, **kwargs)


class AscendIndexerKPoolTailBackend(AttentionBackend):
    """Cache-only backend for the GLM-Next compressor tail."""

    @staticmethod
    def get_impl_cls():
        return None

    @staticmethod
    def get_name() -> str:
        return "ASCEND_INDEXER_KPOOL_TAIL"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Ring capacity is independent of the pool size and SFA C128.
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type[AscendIndexerKPoolTailMetadataBuilder]:
        return AscendIndexerKPoolTailMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_type, cache_dtype_str
        if num_kv_heads != 1:
            raise ValueError(f"Indexer KPool tail cache requires one KV head, got {num_kv_heads}.")
        return (num_blocks, block_size, 2 * head_size)


class Glm5NextKPoolIndexerBackend(nn.Module):
    """Model-side implementation of the unified seven-argument indexer API.

    The cache-only backends above own the engine-facing attention contracts.
    Keep execution independent of the standard SFA indexer and its operators.
    """

    def __init__(self, vllm_indexer: nn.Module, qk_rope_head_dim: int) -> None:
        super().__init__()
        if qk_rope_head_dim != 0:
            raise ValueError(
                f"GLM-Next KPool indexing supports NoPE queries only, got qk_rope_head_dim={qk_rope_head_dim}."
            )
        parallel_config = vllm_indexer.vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size > 1 or parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("GLM-Next KPool indexing does not support PCP or DCP.")

        if get_current_hardware_profile().attention_backend_family is AttentionBackendFamily.COMPATIBILITY:
            raise NotImplementedError("KPool sparse attention requires Ascend A2, A3 or A5.")

        self.n_head: int = vllm_indexer.n_head
        self.head_dim: int = vllm_indexer.head_dim
        self.topk_tokens: int = vllm_indexer.topk_tokens
        self.q_lora_rank: int = vllm_indexer.q_lora_rank
        self.index_kpool: int = vllm_indexer.index_kpool
        self.wq_b = vllm_indexer.wq_b
        self.wk_weights_proj = vllm_indexer.wk_weights_proj
        self.k_norm = vllm_indexer.k_norm
        self.softmax_scale = vllm_indexer.softmax_scale
        self.index_kpool_compress_ape = vllm_indexer.index_kpool_compress_ape
        self.index_kpool_compress_gate = vllm_indexer.index_kpool_compress_gate
        self.k_cache: Any = vllm_indexer.k_cache
        self.tail_cache: Any = vllm_indexer.tail_cache
        self.topk_indices_buffer: torch.Tensor | None = vllm_indexer.topk_indices_buffer
        # Load KPool operators only when constructing the model-side backend;
        # cache metadata is also imported during engine initialization.
        if enable_dsa_cp() or self.k_cache.dtype == torch.int8:
            from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool import SparseAttnIndexerKpool
        else:
            from vllm_ascend.models.glm5next.sparse_attn_indexer_kpool_cann import SparseAttnIndexerKpool

        if enable_dsa_cp() and self.k_cache.dtype == torch.float8_e4m3fn:
            self.indexer_op = SparseAttnIndexerKpool(self.topk_tokens, self.head_dim, use_cann_indexer=True)
        else:
            self.indexer_op = SparseAttnIndexerKpool(self.topk_tokens, self.head_dim)
        self.enable_sparse_li_c8 = self.k_cache.dtype in (torch.int8, torch.float8_e4m3fn)
        if self.enable_sparse_li_c8:
            logger.info_once(
                "KPool indexer LI C8 enabled: dtype=%s, backend=%s, dsa_cp=%s",
                self.k_cache.dtype,
                "CANN CP"
                if enable_dsa_cp() and self.k_cache.dtype == torch.float8_e4m3fn
                else ("Triton" if enable_dsa_cp() or self.k_cache.dtype == torch.int8 else "CANN"),
                enable_dsa_cp(),
            )
        for name in (
            "_key_weight",
            "_gate_weight",
            "_head_weight_f32",
            "_norm_weight_f32",
            "_norm_bias_f32",
            "_wk_weight_f32",
            "_gate_weight_f32",
        ):
            self.register_buffer(name, None, persistent=False)

    @property
    def topk_output_width(self) -> int:
        return self.topk_tokens + self.index_kpool - 1

    def get_topk_lengths(self, positions: torch.Tensor) -> torch.Tensor:
        visible = (positions + 1).clamp_min(0)
        history = (visible // self.index_kpool * self.index_kpool).clamp(max=self.topk_tokens)
        return history + visible % self.index_kpool

    @property
    def num_cache_tensors(self) -> int:
        return 2 if self.k_cache.dtype == torch.float8_e4m3fn else 1

    def process_weights_after_loading(self) -> None:
        if enable_dsa_cp() or self.k_cache.dtype == torch.int8:
            self._wk_weight_f32 = self.wk_weights_proj.weight.detach().float()
            self._gate_weight_f32 = self.index_kpool_compress_gate.detach().float()
        self._key_weight = self.wk_weights_proj.weight[: self.head_dim].detach().contiguous()
        self._gate_weight = self.index_kpool_compress_gate.detach().to(self._key_weight.dtype).contiguous()
        self._head_weight_f32 = self.wk_weights_proj.weight[self.head_dim :].detach().float().contiguous()
        self._norm_weight_f32 = self.k_norm.weight.detach().float() if self.k_norm.weight is not None else None
        self._norm_bias_f32 = self.k_norm.bias.detach().float() if self.k_norm.bias is not None else None

    @staticmethod
    def _bound_cache(layer: Any, slot: int = 0) -> torch.Tensor:
        context = get_forward_context()
        cache = layer.kv_cache
        if isinstance(cache, list):
            virtual_engine = getattr(context, "virtual_engine", 0) or 0
            if virtual_engine >= len(cache):
                raise IndexError(f"Cache virtual engine {virtual_engine} is out of range.")
            cache = cache[virtual_engine]
        if isinstance(cache, (list, tuple)):
            expected = 2 if cache and cache[0].dtype == torch.float8_e4m3fn else 1
            if len(cache) != expected:
                raise TypeError("GLM KPool cache requires a key tensor and, for FP8, one scale tensor.")
            cache = cache[slot]
        elif slot != 0:
            raise TypeError("GLM KPool FP8 cache requires a bound scale tensor.")
        if not isinstance(cache, torch.Tensor):
            raise TypeError(f"GLM KPool cache {layer.prefix!r} is not bound to a tensor.")
        return cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k_hidden_states: torch.Tensor,
        indexer_metadata: Any,
        compute_topk: bool = True,
        attn_q_gather_handle: torch.distributed.Work | None = None,
    ) -> torch.Tensor | None:
        if not isinstance(indexer_metadata, AscendIndexerKPoolMetadata):
            raise TypeError("GLM KPool backend requires AscendIndexerKPoolMetadata.")
        context = get_forward_context()
        if not isinstance(context.attn_metadata, dict):
            raise TypeError("GLM KPool backend requires per-layer metadata.")
        tail_metadata = context.attn_metadata[self.tail_cache.prefix]
        if not isinstance(tail_metadata, AscendIndexerKPoolTailMetadata):
            raise TypeError("GLM KPool backend requires tail-cache metadata.")

        num_tokens = hidden_states.shape[0]
        if indexer_metadata.query_metadata is None and context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            num_tokens = min(num_tokens, indexer_metadata.num_actual_tokens)
        hidden = hidden_states[:num_tokens]
        k_hidden = k_hidden_states[:num_tokens]
        if self._key_weight is None:
            self.process_weights_after_loading()
        assert self._key_weight is not None
        cp_hidden = cp_projected = None
        use_triton = indexer_metadata.query_metadata is not None or self.k_cache.dtype == torch.int8
        if use_triton:
            assert self._wk_weight_f32 is not None
            cp_hidden = k_hidden.float()
            cp_projected = F.linear(cp_hidden, self._wk_weight_f32)
        q_values = None
        weights = None
        if compute_topk:
            if isinstance(q_c, tuple):
                raise TypeError("GLM KPool backend requires an unquantized q_c tensor.")
            q_values = self.wq_b(q_c[:num_tokens])[0].view(num_tokens, self.n_head, self.head_dim)
            weights = (
                cp_projected[:, self.head_dim :]
                if cp_projected is not None and k_hidden_states is hidden_states
                else F.linear(hidden.float(), self._head_weight_f32)
            ).to(q_values.dtype)
            weights = weights * (self.softmax_scale * self.n_head**-0.5)

        indexer_cache = self._bound_cache(self.k_cache)
        indexer_scale_cache = (
            self._bound_cache(self.k_cache, slot=1) if indexer_cache.dtype == torch.float8_e4m3fn else None
        )
        tail_cache = self._bound_cache(self.tail_cache)
        if use_triton:
            assert cp_projected is not None and cp_hidden is not None and self._gate_weight_f32 is not None
            k = F.layer_norm(
                cp_projected[:, : self.head_dim],
                (self.head_dim,),
                self._norm_weight_f32,
                self._norm_bias_f32,
                getattr(self.k_norm, "eps", getattr(self.k_norm, "variance_epsilon", 1e-6)),
            )
            gate_score = F.linear(cp_hidden, self._gate_weight_f32)
            if indexer_metadata.query_metadata is not None:
                # Gather global K/gate rows before per-rank pooling and C8
                # quantization; top-k continues to use rank-local queries.
                gathered = get_tp_group().all_gather(torch.cat((k, gate_score), dim=-1), dim=0)
                k, gate_score = gathered.split(self.head_dim, dim=-1)
            tail_planes = tail_cache.view(tail_cache.shape[0], tail_cache.shape[1], 2, self.head_dim).permute(
                0, 2, 1, 3
            )
            return self.indexer_op(
                k,
                q_values,
                weights,
                indexer_metadata.positions[: k.shape[0]],
                indexer_cache,
                tail_planes,
                indexer_metadata,
                tail_metadata,
                gate_score=gate_score,
                compress_ape=self.index_kpool_compress_ape,
                index_kpool=self.index_kpool,
                max_pool_seq_len=(
                    indexer_metadata.block_table.shape[1] * indexer_cache.shape[1]
                    if context.cudagraph_runtime_mode == CUDAGraphMode.FULL or indexer_metadata.seq_lens_cpu is None
                    else int(indexer_metadata.seq_lens_cpu.max())
                    if indexer_metadata.seq_lens_cpu.numel()
                    else 0
                ),
                compute_topk=compute_topk,
                output_buffer=self.topk_indices_buffer,
                allow_cache_packing=context.cudagraph_runtime_mode != CUDAGraphMode.FULL,
                query_metadata=indexer_metadata.query_metadata,
                indexer_scale_cache=indexer_scale_cache,
            )
        positions = indexer_metadata.positions[:num_tokens]
        result = self.indexer_op(
            k_hidden,
            q_values,
            weights,
            positions,
            indexer_cache,
            tail_cache,
            indexer_metadata,
            tail_metadata,
            key_weight=self._key_weight,
            gate_weight=self._gate_weight,
            norm_weight=self._norm_weight_f32,
            norm_bias=self._norm_bias_f32,
            norm_eps=getattr(self.k_norm, "eps", getattr(self.k_norm, "variance_epsilon", 1e-6)),
            compress_ape=self.index_kpool_compress_ape,
            index_kpool=self.index_kpool,
            compute_topk=compute_topk,
            output_buffer=self.topk_indices_buffer,
            indexer_scale_cache=indexer_scale_cache,
        )
        return result
