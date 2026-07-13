#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from enum import Enum

import torch
from vllm.logger import logger
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.kvcomp_attn.attention_utils import (
    get_kvcomp_decode_params,
    is_enable_hamming_sparse,
    reshape_and_cache_kvcomp,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    notify_kv_cache_written,
    split_decodes_and_prefills,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.memcache_comm_fence import record_attention_compute_start
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.utils import weak_ref_tensors
from vllm_ascend.worker.kvcomp_utils import KVCompMetaData

from vllm_ascend.attention.hada_trans import block_diag_random_hadamard
# default max value of sliding window size
SWA_INT_MAX = 2147483647
_ATTN_KEYS_BUFFER = None

HIGH_PRECISION_WINDOW_SIZE = 128
ATTENTION_SINK_SIZE = 128


_MXFP4_EBITS = 2
_MXFP4_MBITS = 3
_MXFP4_EMAX = 2
_MXFP4_MAX_NORM = 6.0
_MXFP4_BLOCK_SIZE = 32
# K sliding-window high-precision tail size (tokens kept full-precision).
# Decoupled from _MXFP4_BLOCK_SIZE (the per-quantize batch / MXFP4 group size):
# the tail K window is 64 tokens, but each trigger still quantizes only the
# oldest _MXFP4_BLOCK_SIZE (32) not-yet-quantized tokens.
K_HIGH_PRECISION_WINDOW = 128
_MXFP4_MIN_EXP = 0.0
_MXFP4_SCALE_FACTOR = 2.0
_MXFP4_INV_SCALE_FACTOR = 0.5
_MXFP4_EPSILON = 1.17e-38
_E8M0_SCALE_EMAX = 127
# Precomputed block-diagonal random Hadamard rotation matrix (head_size x head_size)
# used by MXFP4 pseudo-quant. Loaded once at weight-load time, shared by Q and K.
_MXFP4_ROT_H_PATH = "/home/z00909726/block_rht_matrix.pt"

# Pseudo-quantization strategy switch (hardcoded).
#   False -> MXFP4: rot_h rotation + E2M1/E8M0 quant, windowed decode.
#   True  -> HiF4 : S1E2M1 three-level scale quant;
#                  prefill q/k/v quantized with sink protection
#                  (first ATTENTION_SINK_SIZE tokens stay full-precision);
#                  decode reuses the windowed late-quantization framework
#                  (K on head_size dim, V on seq_len dim per V_HIF4_WINDOW
#                  tokens), all via the HiF4 kernel.
_ENABLE_HIF4 = True
# K decode quantization strategy (HiF4 path only):
#   False -> windowed late-quant: new K is written high-precision, only the
#            tokens sliding out of the window are quantized in-place later.
#   True  -> full quantization: each token's K is HiF4-quantized before being
#            written to cache (no windowing). Prefill is unaffected (still
#            sink-protected).
_K_FULL_QUANT = False
# Hadamard rotation switch (HiF4 path only).
#   True  -> orthogonal rotation x'=x@rot_h on Q/K before quantization to spread
#            energy across channels and reduce per-block quant error; V is NOT
#            rotated. Requires rot_h, loaded in process_weights_after_loading.
#   False -> no rotation (plain HiF4).
# R is orthogonal (R@R^T=I); with the same fixed R on both Q and K, the dot
# product cancels the rotation (Q'@K'^T = Q@K^T), which is why V is excluded.
# The rotation MUST be identical at prefill and decode (cache holds rotated K).
_ENABLE_ROT_H = True


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    # Per-request prompt token count (fixed prefill length / decode starting
    # point). Used by windowed V quantization to anchor the high-precision
    # window on the decode region instead of the absolute sequence origin.
    num_prompt_tokens_list: list[int] = None  # type: ignore
    # CPU tensor form of num_prompt_tokens, for graph-mode update_graph_params
    # to refresh _prefill_len_buffer in-place.
    num_prompt_tokens_tensor: torch.Tensor = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # pcp
    prefill: AscendMetadataForPrefill | None = None
    # dcp
    decode_meta: AscendMetadataForDecode | None = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    kvcomp_metadata: KVCompMetaData | None = None


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )
        block_table = common_attn_metadata.block_table_tensor
        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # this slot_mapping override doesn't work since vllm will override it again. We should fix it vllm.
        # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
        elif self.speculative_config and self.speculative_config.parallel_drafting:
            seq_lens = common_attn_metadata.seq_lens

        attn_state = common_attn_metadata.attn_state

        # Per-request prompt token count (prefill length / decode anchor) for
        # windowed V quantization. Defensive: may be None under async_spec_decode.
        npt_cpu = common_attn_metadata.num_prompt_tokens_cpu
        num_prompt_tokens_list = npt_cpu[:num_reqs].tolist() if npt_cpu is not None else None
        num_prompt_tokens_tensor = npt_cpu[:num_reqs] if npt_cpu is not None else None

        # Get attn_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config)

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            num_prompt_tokens_list=num_prompt_tokens_list,
            num_prompt_tokens_tensor=num_prompt_tokens_tensor,
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
            kvcomp_metadata=common_attn_metadata.kvcomp_metadata,
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata




class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.enable_c8_quant = self.vllm_config.quant_config is not None and getattr(
            self.vllm_config.quant_config, "enable_c8_quant", False
        )
        self.sinks = sinks
        self.layerIndex = 0
        self.enable_hamming_sparse = is_enable_hamming_sparse()
        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        self._seq_lens_buffer = torch.zeros(
            max_batch_size, dtype=torch.int32, device="npu"
        )
        # Graph-mode windowed V quantization: per-request prefill length
        # (decode starting point), refreshed in-place by update_graph_params.
        self._prefill_len_buffer = torch.zeros(
            max_batch_size, dtype=torch.int32, device="npu"
        )
    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        if using_paged_attention(num_tokens, vllm_config):
            # Paged Attention update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
            else:
                graph_params = get_graph_params()
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    forward_context.attn_metadata,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        num_heads,
                        scale,
                        block_table,
                        seq_lens,
                        output,
                    ) = param
                    seq_lens = forward_context.attn_metadata[key].seq_lens

                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                        workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        elif _EXTRA_CTX.sinks:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        num_kv_heads,
                        num_heads,
                        scale,
                        sliding_window,
                        sinks,
                        attn_output,
                        softmax_lse,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        attn_count = attn_count + 1
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score_v2.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_qlen=actual_seq_lengths_q,
                        actual_seq_kvlen=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_query_heads=num_heads,
                        sparse_mode=4 if sliding_window is not None else 3,
                        pre_tokens=sliding_window if sliding_window is not None else SWA_INT_MAX,
                        next_tokens=0,
                        softmax_scale=scale,
                        learnable_sink=sinks,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        else:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
                # In some speculative methods (such as DFlash), the order of attn_keys in the Target model
                # will be disrupted instead of increasing by layer index, so need regular expressions to
                # reorder the attn_keys and stor the results in _ATTN_KEYS_BUFFER.
                attn_keys_length = len(graph_params.attn_params[num_tokens])
                global _ATTN_KEYS_BUFFER
                if _ATTN_KEYS_BUFFER is None:
                    import regex as re

                    def extract_layer_index(key: str) -> int:
                        match = re.search(r"(\d+)", key)
                        return int(match.group(1)) if match else 0

                    attn_keys_tmp = attn_keys[:attn_keys_length]
                    attn_keys_tmp.sort(key=extract_layer_index)
                    _ATTN_KEYS_BUFFER = attn_keys_tmp
                attn_keys[:attn_keys_length] = _ATTN_KEYS_BUFFER
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        query_start_loc,
                        num_kv_heads,
                        num_heads,
                        scale,
                        attn_output,
                        softmax_lse,
                        sparse_mode,
                        pre_tokens,
                        next_tokens,
                        c8_k_aq_scale,
                        c8_k_aq_offset,
                        c8_v_aq_scale,
                        c8_v_aq_offset,
                        evicted_kv_seq_lens_buffer,
                        prefill_len_buffer,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        latest_cpu_seq_lens = attn_metadata[draft_step][key].seq_lens
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        block_tables = attn_metadata[draft_step][key].block_tables
                        latest_npt = attn_metadata[draft_step][key].num_prompt_tokens_tensor
                        attn_count = attn_count + 1
                        if not attn_metadata[draft_step][key].causal:
                            sparse_mode = 0
                    else:
                        latest_cpu_seq_lens = attn_metadata[key].seq_lens
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
                        latest_npt = attn_metadata[key].num_prompt_tokens_tensor
                        # NOTE:
                        # For models with sliding-window attention on the FIA full-graph replay path,
                        # rebinding `block_tables` to the latest metadata tensor causes corrupted /
                        # repeated outputs in our repro on Ascend NPU.
                        #
                        # Keep the captured block_tables tensor on this affected path.
                        # Non-SWA models preserve the original behavior and continue to refresh
                        # block_tables from attn_metadata.
                        if not hasattr(vllm_config.model_config.hf_text_config, "sliding_window"):
                            block_tables = attn_metadata[key].block_tables

                    if evicted_kv_seq_lens_buffer is not None and latest_cpu_seq_lens is not None:
                        num_reqs = latest_cpu_seq_lens.numel()
                        evicted_kv_seq_lens_buffer[:num_reqs].copy_(latest_cpu_seq_lens, non_blocking=True)
                        # Refresh per-request prefill length for windowed V quant.
                        # None (async_spec_decode fallback) -> set to seq_lens so
                        # decode_len = 0 never triggers window quantization.
                        if prefill_len_buffer is not None:
                            if latest_npt is not None:
                                npt_reqs = latest_npt.numel()
                                prefill_len_buffer[:npt_reqs].copy_(
                                    latest_npt.to(torch.int32), non_blocking=True)
                                if npt_reqs < prefill_len_buffer.numel():
                                    prefill_len_buffer[npt_reqs:].zero_()
                            else:
                                prefill_len_buffer.copy_(evicted_kv_seq_lens_buffer, non_blocking=True)

                    torch.npu.graph_task_update_begin(update_stream, handle)

                    input_layout = "TND"
                    extra_args = {}
                    if c8_k_aq_scale is not None:
                        extra_args = {
                            "key_antiquant_scale": c8_k_aq_scale,
                            "value_antiquant_scale": c8_v_aq_scale,
                            "key_antiquant_mode": 0,
                            "value_antiquant_mode": 0,
                            "inner_precise": 1,
                        }
                        input_layout = "BNSD"
                        sparse_mode = 0
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout=input_layout,
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=sparse_mode,
                        pre_tokens=pre_tokens,
                        next_tokens=next_tokens,
                        **extra_args,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.post_process_after_loading()

        # Load the rotation matrix (RHT) once per layer and cache it. Used by
        # MXFP4 (always) and by HiF4 when _ENABLE_ROT_H is set. It MUST stay
        # fixed across all steps (prefill + decode) and be shared by Q and K,
        # otherwise Q K^T cannot cancel R R^T = I.
        # See hada_trans.block_diag_random_hadamard.
        need_rot_h = (not _ENABLE_HIF4) or (_ENABLE_HIF4 and _ENABLE_ROT_H)
        if need_rot_h and not hasattr(self, "rot_h"):
            self.rot_h = torch.load(_MXFP4_ROT_H_PATH).to(device="npu", dtype=act_dtype)

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer=None,
    ) -> torch.Tensor:
        passed_key = key
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        if self.enable_hamming_sparse and attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            reshape_and_cache_kvcomp(attn_metadata.kvcomp_metadata, self.layerIndex, passed_key)
        elif self.enable_hamming_sparse:
            block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(
                self.layerIndex, attn_metadata.kvcomp_metadata, query, passed_key, block_table, actual_seq_lengths_kv
            )

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        input_layout = "TND"
        attn_mask = attn_metadata.attn_mask
        sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0
        pre_tokens = self.sliding_window or SWA_INT_MAX
        next_tokens = 0 if self.sliding_window else SWA_INT_MAX

        extra_args = {}
        if self.enable_c8_quant and layer is not None:
            extra_args = {
                "key_antiquant_scale": layer._c8_k_aq_scale_nz_bnsd,
                "value_antiquant_scale": layer._c8_v_aq_scale_nz_bnsd,
                "key_antiquant_mode": 0,
                "value_antiquant_mode": 0,
                "inner_precise": 1,
            }

            # change key/value shape
            _, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self._nz_5d_view(self.key_cache, block_size)
            value = self._nz_5d_view(self.value_cache, block_size)

            # TODO: change layerout from BNSD to TND.
            input_layout = "BNSD"
            query = query.unsqueeze(2)
            output = output.unsqueeze(2)
            attn_mask = None
            sparse_mode = 0
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout=input_layout,
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                scale=self.scale,
                **extra_args,
            )
            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        attn_params = (
            weak_ref_tensors(query),
            weak_ref_tensors(key),
            weak_ref_tensors(value),
            weak_ref_tensors(block_table),
            weak_ref_tensors(attn_mask) if attn_mask is not None else None,
            block_size,
            actual_seq_lengths_kv,
            actual_seq_lengths_q,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(output),
            weak_ref_tensors(softmax_lse),
            sparse_mode,
            pre_tokens,
            next_tokens,
        )

        # weak_ref_tensors 确保引用正确
        if self.enable_c8_quant and layer is not None:
            attn_params = attn_params + (
                weak_ref_tensors(layer._c8_k_aq_scale_nz_bnsd),
                None,
                weak_ref_tensors(layer._c8_v_aq_scale_nz_bnsd),
                None,
                self._seq_lens_buffer,
                self._prefill_len_buffer,
            )  # type: ignore
        else:
            attn_params = attn_params + (None, None, None, None, self._seq_lens_buffer, self._prefill_len_buffer)
        graph_params.attn_params[num_tokens].append(attn_params)

        # if self.enable_c8_quant and layer is not None:
        #     attn_params = attn_params + (
        #         weak_ref_tensors(layer._c8_k_aq_scale_nz_bnsd),
        #         None,
        #         weak_ref_tensors(layer._c8_v_aq_scale_nz_bnsd),
        #         None,
        #     )  # type: ignore
        # else:
        #     attn_params = attn_params + (None, None, None, None)  # type: ignore
        # graph_params.attn_params[num_tokens].append(attn_params)

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout=input_layout,
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            workspace=workspace,
            out=[output, softmax_lse],
            **extra_args,
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_fia_v2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        actual_seq_lengths_kv = attn_metadata.seq_lens
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()

        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=actual_seq_lengths_q,
                actual_seq_kvlen=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=4 if self.sliding_window is not None else 3,
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                learnable_sink=self.sinks,
            )

            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                weak_ref_tensors(query),
                weak_ref_tensors(key),
                weak_ref_tensors(value),
                weak_ref_tensors(block_table),
                weak_ref_tensors(attn_metadata.attn_mask),
                block_size,
                actual_seq_lengths_kv,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                self.sliding_window,
                self.sinks,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
            )
        )
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_qlen=actual_seq_lengths_q,
            actual_seq_kvlen=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            sparse_mode=4 if self.sliding_window is not None else 3,
            pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
            next_tokens=0,
            softmax_scale=self.scale,
            learnable_sink=self.sinks,
            workspace=workspace,
            out=[output, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        if _EXTRA_CTX.capturing:
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append(
                (
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens,
                    weak_ref_tensors(output),
                )
            )

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata, kv_cache=None):
        # PrefillNoCache doesn't need key_cache, but other modes do
        # Only initialize/require cache for modes that actually use it
        if attn_metadata.attn_state != AscendAttentionState.PrefillNoCache:
            # Initialize cache from kv_cache if not already set (for DecodeOnly mode)
            if self.key_cache is None and kv_cache is not None:
                if (
                    isinstance(kv_cache, torch.Tensor)
                    and kv_cache.dim() > 0
                    and kv_cache.shape[0] == 2
                    or isinstance(kv_cache, (list, tuple))
                    and len(kv_cache) >= 2
                ):
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if self.key_cache is None:
                raise RuntimeError(
                    f"key_cache is None in _get_fia_params for mode {attn_metadata.attn_state}. kv_cache={kv_cache}"
                )


        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()

        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
            #
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
            #
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
            #
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if _EXTRA_CTX.capturing:
            if self.sinks is not None:
                attn_output, num_tokens = self.full_graph_fia_v2(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            else:
                attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
        passed_key = key

        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )

        if self.enable_hamming_sparse and attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            reshape_and_cache_kvcomp(attn_metadata.kvcomp_metadata, self.layerIndex, passed_key)
        elif self.enable_hamming_sparse:
            block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(
                self.layerIndex, attn_metadata.kvcomp_metadata, query, passed_key, block_table, actual_seq_lengths_kv
            )
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        # Get workspace from cache or calculate it if not present.
        if self.sinks is not None:
            actual_seq_qlen = attn_metadata.actual_seq_lengths_q
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                actual_seq_qlen = torch.tensor([1] * len(attn_metadata.seq_lens_list), dtype=torch.int32).cumsum(dim=0)
            if self.sliding_window is not None:
                sparse_mode = 4
            else:
                sparse_mode = 3
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key,
                value,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                atten_mask=attn_metadata.attn_mask,
                sparse_mode=sparse_mode,
                softmax_scale=self.scale,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                learnable_sink=self.sinks,
            )
        else:
            if not attn_metadata.causal:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=0,
                )
            elif self.sliding_window is not None:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    pre_tokens=self.sliding_window,
                    next_tokens=0,
                    sparse_mode=4,
                )
            else:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=3,
                )

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        # use default sparse_mode 0 in normal scenario, which means no mask works on it
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
            actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
        )[0]

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY):
            return

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slot_mapping,
        )

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                # quick fix to make sure slots is int32 for cross attention case.
                # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32),
            )
            notify_kv_cache_written()
        return query, key, value, output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        num_tokens = query.shape[0]
        record_attention_compute_start()
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(num_tokens, self.vllm_config)
            and self.sliding_window is None
        ):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if self.enable_hamming_sparse:
            self.layerIndex = int(layer.layer_name.split(".")[2])

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Initialize key_cache and value_cache from kv_cache if not already set.
        # This is needed for DecodeOnly mode where key/value are None but we still
        # need access to the cache for attention computation.
        if self.key_cache is None and kv_cache is not None:
            if (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, (list, tuple))
                and len(kv_cache) >= 2
            ):
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        output_padded = None
        is_decode = attn_metadata.attn_state == AscendAttentionState.DecodeOnly

        # logger.info_once(f"is_decode: {is_decode}")
        
        # self._prepare_c4_scales(layer, query.device)
        
        if key is not None and value is not None:
            output_padded = output
            # Prefill阶段: 全部伪量化再写入cache
            # Decode阶段: 延迟量化——新KV不量化直接写入, 稍后对滑出窗口的旧KV做原地伪量化
            
            # if not is_decode:
            # #     key, value = self._quantize_kv_to_fp4(key, value, layer, attn_metadata.num_actual_tokens)
            #     key, value = self._quantize_kv_to_fp4_sink(key, value, layer, attn_metadata.num_actual_tokens, attn_metadata.query_start_loc)
            # key, value = self._quantize_kv_to_fp4(key, value, layer, attn_metadata.num_actual_tokens)
            
            # key, value : [num_tokens, num_kv_heads, head_size]
            # logger.info_once(f"key.shape: {key.shape}, attn_metadata.num_actual_tokens:{attn_metadata.num_actual_tokens}")
            
            num_tokens = attn_metadata.num_actual_tokens
            num_tokens_q = attn_metadata.actual_seq_lengths_q[-1]
            # logger.info_once(f"key.shape : {key.shape}, query.shape: {query.shape}, value.shape: {value.shape}")
            num_block = int(query.shape[-1] / _MXFP4_BLOCK_SIZE)
            # logger.info_once(f"query.shape[-1]:{query.shape[-1]}, num_block : {num_block} ")
            rot_h = self.rot_h
            # logger.info_once(f"分块旋转矩阵: {rot_h}, shape: {rot_h.shape}")
            # if not is_decode:
            #     logger.info_once(f"------------------------------prefill:::::key.shape : {key.shape}, query.shape: {query.shape}, value.shape: {value.shape}")
            # else:
            #     logger.info_once(f"++++++++++++++++++++++++++++++decode:::::key.shape : {key.shape}, query.shape: {query.shape}, value.shape: {value.shape}")
            
            # K 量化策略:
            #   Prefill: 旋转 + MXFP4 整段量化后写 cache (保持现状)。
            #   Decode : 只旋转 (matmul key, rot_h), 不量化, 高精写入 cache;
            #            稍后由 _quantize_window_k_slots_* 做批量 FIFO 滑窗量化
            #            (尾部 32 token 始终高精, 触发时量化最老的 32 个未量化 token)。


            # V quantization along seq_len (token) dim, per-request granularity.
            # Prefill: rotate+quantize Q/K/V now (before writing to cache).
            # Decode: rotate+quantize the new Q/K now so the cached key is
            # consistently rotated; keep new V high-precision here and run
            # windowed in-place V quantization after reshape_and_cache.
            if not is_decode:
                query, key, value = self._quantize_qkv_to_mxfp4_sink(
                    attn_metadata, query, key, value, layer, num_tokens,
                    attn_metadata.query_start_loc)
                logger.info_once(
                    f"after sink quantize query.shape: {query.shape}, key.shape : {key.shape}, "
                    f"value.shape : {value.shape}, num_tokens: {num_tokens}, num_tokens_q: {num_tokens_q}"
                )
            else:
                # Rotate+quantize decode Q/K BEFORE writing K to cache. The
                # attention kernel reads cached K/V, so the cache must contain
                # the same rotated+quantized representation used at prefill
                # time; otherwise Q@K^T is computed between rotated and
                # unrotated keys and output becomes gibberish.
                key = self._mxfp4_quant_tf(torch.matmul(key, rot_h), -1).to(key.dtype)
                query = self._mxfp4_quant_tf(torch.matmul(query, rot_h), -1).to(query.dtype)

            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )
            # if is_decode:
            #     logger.info_once(f"after reshape: q_:{query.shape},k_:{key.shape}")

            # Decode: the new token was just written to cache high-precision.
            # Now in-place MXFP4-quantize the V of the just-filled 32-token
            # decode window, if any request crossed a 32-boundary this step.
            if is_decode:
                if _EXTRA_CTX.capturing:
                    self._quantize_window_v_slots_graph(layer, attn_metadata, attn_metadata.num_decodes)
                    self._quantize_window_k_slots_graph(layer, attn_metadata, attn_metadata.num_decodes)
                else:
                    query = self._quant_dequant_hif4(query)
                    # K decode quantization branch:
                    #   full-quant: each token's K is HiF4-quantized before writing
                    #               to cache (no windowing).
                    #   windowed  : K is written high-precision (rotated value) and
                    #               quantized in-place later by the K window below.
                    if _K_FULL_QUANT:
                        key = self._quant_dequant_hif4(key)

                query, key, value, output_padded = self.reshape_and_cache(
                    query, key, value, kv_cache, attn_metadata, output
                )

                # Decode: new KV has been written to cache.
                #   V is always quantized in-place by its window (seq_len dim,
                #   per V_HIF4_WINDOW tokens).
                #   K: already quantized before write when _K_FULL_QUANT, so the
                #      window is skipped; otherwise the K window runs here.
                if is_decode:
                    if _EXTRA_CTX.capturing:
                        self._quantize_window_v_slots_hif4_graph(layer, attn_metadata, attn_metadata.num_decodes)
                        if not _K_FULL_QUANT:
                            self._quantize_window_k_slots_hif4_graph(layer, attn_metadata, attn_metadata.num_decodes)
                    else:
                        self._quantize_window_v_slots_hif4_eager(layer, attn_metadata)
                        if not _K_FULL_QUANT:
                            self._quantize_window_k_slots_hif4_eager(layer, attn_metadata)
            else:
                # ============ MXFP4 path (unreachable while _ENABLE_HIF4=True) ===== #
                # Prefill: fully pseudo-quantize before writing to cache.
                # Decode : late quantization -- new KV is written unquantized and the
                #          tokens sliding out of the window are quantized in-place.
                num_tokens = attn_metadata.num_actual_tokens
                num_tokens_q = attn_metadata.actual_seq_lengths_q[-1]
                num_block = int(query.shape[-1] / _MXFP4_BLOCK_SIZE)
                rot_h = self.rot_h
                # K strategy:
                #   Prefill: rotate + full MXFP4 quant, then write to cache.
                #   Decode : rotate only (matmul key, rot_h), no quant, high-precision
                #            write; the FIFO window (_quantize_window_k_slots_*)
                #            quantizes the oldest _MXFP4_BLOCK_SIZE not-yet-quantized
                #            tokens when a batch is ready, keeping the K_HIGH_PRECISION_WINDOW
                #            tail high-precision.
                if not is_decode:
                    key = self._mxfp4_quant_tf(torch.matmul(key, rot_h), -1)
                else:
                    key = torch.matmul(key, rot_h)
                query = self._mxfp4_quant_tf(torch.matmul(query, rot_h), -1)

                # V quantization along seq_len (token) dim, per-request granularity.
                # Prefill: quantize each request's V now (before writing to cache).
                # Decode: keep new V high-precision here; windowed in-place quant of
                # the just-filled 32-token window runs after reshape_and_cache.
                if not is_decode:
                    value = self._quantize_v_prefill(value, attn_metadata)

                query, key, value, output_padded = self.reshape_and_cache(
                    query, key, value, kv_cache, attn_metadata, output
                )

                # Decode: the new token was just written to cache high-precision.
                # Now in-place MXFP4-quantize the V of the just-filled 32-token
                # decode window, if any request crossed a 32-boundary this step.
                if is_decode:
                    if _EXTRA_CTX.capturing:
                        self._quantize_window_v_slots_graph(layer, attn_metadata, attn_metadata.num_decodes)
                        self._quantize_window_k_slots_graph(layer, attn_metadata, attn_metadata.num_decodes)
                    else:
                        self._quantize_window_v_slots_eager(layer, attn_metadata)
                        self._quantize_window_k_slots_eager(layer, attn_metadata)

        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def _quantize_qkv_to_mxfp4_sink(
            self,
            attn_metadata,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            layer: AttentionLayer,
            num_actual_tokens: int,
            query_start_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sink_size = ATTENTION_SINK_SIZE

        logger.info_once(f"start sink, sink size: {sink_size}")

        actual_query = query[:num_actual_tokens]
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        global_idx = torch.arange(num_actual_tokens, device=key.device)
        seq_ids = torch.bucketize(global_idx, query_start_loc[1:], right=True)
        seq_start_locs = query_start_loc[seq_ids]
        relative_idx = global_idx - seq_start_locs
        quant_mask = (relative_idx >= sink_size).view(-1, 1, 1)

        # 全序列统一RoPE
        # Apply the same fixed Hadamard rotation to Q/K before MXFP4
        # quantization. The rotation must be identical at prefill and decode
        # time so that Q@K^T is preserved after de-quantization.
        query_rotated = torch.matmul(actual_query, self.rot_h)
        key_rotated = torch.matmul(actual_key, self.rot_h)

        q_quant = self._mxfp4_quant_tf(query_rotated, -1)
        k_quant = self._mxfp4_quant_tf(key_rotated, -1)
        v_quant = self._quantize_v_prefill(actual_value, attn_metadata)

        dtype = key.dtype
        # Sink tokens keep the rotated full-precision value; non-sink tokens
        # use the rotated+quantized value. Cast back to the original dtype so
        # the KV cache and attention inputs stay in the model's activation
        # dtype instead of float32.
        q_out = torch.where(quant_mask, q_quant, query_rotated).to(dtype)
        k_out = torch.where(quant_mask, k_quant, key_rotated).to(dtype)
        v_out = torch.where(quant_mask, v_quant, actual_value).to(dtype)

        return q_out, k_out, v_out

    def _mxfp4_quant_tf(self, tensor, qdim, blocksize=_MXFP4_BLOCK_SIZE, stochastic_rounding=False):
        orig_shape = tensor.shape
        tensor = tensor.unflatten(qdim, (-1, blocksize))

        max_val = torch.amax(tensor.abs(), qdim, keepdim=True)
        inv_constant = 1 / 7
        shared_exp = torch.ceil(torch.log2(max_val.clamp(min=_MXFP4_EPSILON) * inv_constant))
        shared_exp = shared_exp.clamp(-127, 127)

        tensor = tensor * torch.exp2(-shared_exp)

        private_exp = torch.floor(torch.log2(tensor.abs().clamp(min=_MXFP4_EPSILON))).clamp(min=_MXFP4_MIN_EXP)
        tensor = tensor * torch.exp2(-private_exp) * _MXFP4_SCALE_FACTOR

        tensor_sign = torch.sign(tensor)
        tensor = tensor_sign * torch.floor_(tensor.abs() + 0.5)

        tensor = (tensor * _MXFP4_INV_SCALE_FACTOR * torch.exp2(private_exp)).clamp(-_MXFP4_MAX_NORM, _MXFP4_MAX_NORM)

        recovered_tensor = tensor * torch.exp2(shared_exp)

        return recovered_tensor.reshape(orig_shape)

    def _mxfp4_quant_tf_grouped(self, tensor):
        """MXFP4 pseudo-quant on a pre-grouped tensor of shape
        ``[num_groups, _MXFP4_BLOCK_SIZE, num_kv_heads, head_size]``.

        Identical math to :meth:`_mxfp4_quant_tf` but the reduction is fixed on
        dim=1 (the 32-token block axis), so one shared exp is computed per
        (group, kv_head, head_channel) block of 32 tokens. Used by the decode
        windowed V quantization where 32 tokens are gathered as one group.
        """
        orig_shape = tensor.shape
        # reduction on the 32-block axis (dim=1)
        max_val = torch.amax(tensor.abs(), 1, keepdim=True)
        inv_constant = 1 / 7
        shared_exp = torch.ceil(torch.log2(max_val.clamp(min=_MXFP4_EPSILON) * inv_constant))
        shared_exp = shared_exp.clamp(-127, 127)

        tensor = tensor * torch.exp2(-shared_exp)

        private_exp = torch.floor(torch.log2(tensor.abs().clamp(min=_MXFP4_EPSILON))).clamp(min=_MXFP4_MIN_EXP)
        tensor = tensor * torch.exp2(-private_exp) * _MXFP4_SCALE_FACTOR

        tensor_sign = torch.sign(tensor)
        tensor = tensor_sign * torch.floor_(tensor.abs() + 0.5)

        tensor = (tensor * _MXFP4_INV_SCALE_FACTOR * torch.exp2(private_exp)).clamp(-_MXFP4_MAX_NORM, _MXFP4_MAX_NORM)

        recovered_tensor = tensor * torch.exp2(shared_exp)

        return recovered_tensor.reshape(orig_shape)

    # ===================== HiF4 pseudo-quantization ===================== #
    # Ported from attention_v1_hif4.py. Acts on the head_size dim (qdim=-1),
    # each token independent (no token-axis grouping). Used by the HiF4 branch
    # when _ENABLE_HIF4=True. Rotation (if any) is applied by the caller before
    # these kernels run (see _ENABLE_ROT_H in forward); the kernels themselves
    # only quantize.

    def _quant_dequant_hif4(self, x: torch.Tensor, quant_type: str = "hifx4", axe: int = -1):
        dtype_ori = x.dtype
        device = x.device
        C = x.shape[axe]
        blk_size_total = 64
        padC = (blk_size_total - C % blk_size_total) % blk_size_total

        pad_shape = list(x.shape)
        pad_shape[axe] = padC

        x_pad = torch.zeros(pad_shape, dtype=dtype_ori, device=device)
        x_padded = torch.cat([x, x_pad], dim=axe)

        total_C = C + padC
        arange_vec = torch.arange(total_C, device=device)
        mask_vector = (arange_vec < C).to(dtype_ori)

        unsqueezes = [None] * len(x_padded.shape)
        unsqueezes[axe] = slice(None)
        attention_mask = mask_vector[tuple(unsqueezes)]

        qdq_out = self._quantize_hif4_kernel(x, -1)
        qdq_out *= attention_mask
        return qdq_out.to(dtype_ori)

    def _quantize_hif4_kernel(self, x: torch.Tensor, qdim: int):
        # Split the qdim into a 3-level block: (outer, 8, 2, 4) where 8*2*4 = 64
        # scalars form one HiF4 group. (-1) infers the outer count from head_size.
        x = x.unflatten(qdim, (-1, 8, 2, 4))
        man_bits = 3
        x_unsigned = torch.abs(x)
        sign = torch.sign(x)

        # Three-level max: innermost 4 / middle 8 / outer 64 channels
        max_lv3 = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
        max_lv2 = torch.max(max_lv3, dim=qdim - 1, keepdim=True)[0]
        max_lv1 = torch.max(max_lv2, dim=qdim - 2, keepdim=True)[0]

        # div7 = 1/7: allows L2/L3 to each shrink by up to 2x so the largest
        # value lands in [0, 2) after all scaling, ready for mantissa quant.
        div7 = torch.ones_like(max_lv1) / 7.0
        div7 = div7.to(torch.bfloat16).to(x.dtype)
        scale_factor = max_lv1 * div7  # base scale

        # round scale_factor to bf16 mantissa (simulate HW behavior)
        e_sf = torch.floor(torch.log2(scale_factor))
        mant_sf = scale_factor / 2 ** e_sf * 2 ** 7
        scale_factor = torch.round(mant_sf) / 2 ** 7 * 2 ** e_sf

        # round scale_factor to e6m2 (8-bit: 1 sign, 6 exp, 2 mant)
        e_sf = torch.floor(torch.log2(scale_factor))
        scale_factor = torch.round(scale_factor * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)

        # per-sub-block dynamic shift
        rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)
        # L2 sub-block: scale_lv2 = 2 if max_lv2 >= 4*scale_factor else 1
        scale_lv2 = (max_lv2 * rec_sf)
        scale_lv2 = torch.exp2((scale_lv2.clip(0, 4) / 4).floor())
        # L3 sub-block: scale_lv3 = 2 if max_lv3 >= 2*scale_factor else 1
        scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())

        # mantissa quant (3-bit)
        mant = x_unsigned / scale_lv2 / scale_lv3 * rec_sf
        mant = torch.floor(mant * 2 ** (man_bits - 1) + 0.5) / 2 ** (man_bits - 1)
        upper_bound = 2 - 2 ** (-man_bits + 1)
        mant = torch.clamp_max(mant, max=upper_bound)

        # dequant: sign * mant * three-level scale (S1E2M1 carries sign)
        out = sign * mant * scale_lv2 * scale_lv3 * scale_factor

        out = out.flatten(qdim - 3, qdim)
        return out

    # ===================== end HiF4 pseudo-quantization ================= #

    def _quantize_hif4_with_sink(
        self,
        x: torch.Tensor,
        attn_metadata: AscendMetadata,
    ) -> torch.Tensor:
        """HiF4-quantize a prefill tensor with attention-sink protection.

        Mirrors the masking idea of the legacy ``_quantize_kv_to_fp4_sink``:
        for each request the first ``ATTENTION_SINK_SIZE`` tokens stay at their
        original (high-precision) values; only tokens at relative position
        ``>= sink_size`` are HiF4-quantized. Used for prefill q/k/v so the sink
        tokens survive the quantization round-trip. Token->sequence mapping is
        derived from ``query_start_loc`` via ``bucketize`` (handles multi-request
        batches). Only the prefill path calls this; decode uses windowed quant.
        """
        sink_size = ATTENTION_SINK_SIZE
        num_actual_tokens = attn_metadata.num_actual_tokens
        query_start_loc = attn_metadata.query_start_loc
        device = x.device

        # 取真正有效的 prefill token (剔除 padding).
        actual_x = x[:num_actual_tokens]

        # 每个 token 在其所属序列内的相对位置 (用 query_start_loc 做 bucketize).
        global_idx = torch.arange(num_actual_tokens, device=device)
        seq_ids = torch.bucketize(global_idx, query_start_loc[1:], right=True)
        seq_start_locs = query_start_loc[seq_ids]
        relative_idx = global_idx - seq_start_locs            # (num_actual_tokens,)

        # 前 sink_size 个 token 保持原值, 其余做 HiF4 量化.
        quant_mask = (relative_idx >= sink_size).view(-1, 1, 1)
        x_quant_all = self._quant_dequant_hif4(actual_x)
        actual_out = torch.where(quant_mask, x_quant_all, actual_x)

        # 回填到原张量 (保持 padding/无效尾部不变).
        out = x.clone()
        out[:num_actual_tokens] = actual_out
        return out

    def _quantize_v_prefill(self, value: torch.Tensor, attn_metadata: AscendMetadata) -> torch.Tensor:
        """Per-request MXFP4 quantization of V along the seq_len (token) dim.

        Splits ``value`` ``[num_tokens, num_kv_heads, head_size]`` by request
        using ``query_start_loc``; each request is zero-padded to a multiple of
        ``_MXFP4_BLOCK_SIZE`` (32), reshaped to ``[num_blocks, 32, H, D]`` and
        run through :meth:`_mxfp4_quant_tf_grouped`, then truncated back.
        Padding zeros do not raise ``amax(abs)`` so real-token scales are clean.
        """
        qsl = attn_metadata.query_start_loc
        num_reqs = qsl.shape[0] - 1
        parts = []
        for r in range(num_reqs):
            s, e = int(qsl[r]), int(qsl[r + 1])
            vr = value[s:e]                       # [L_r, H, D]
            Lr = vr.shape[0]
            if Lr == 0:
                continue
            pad = (-Lr) % _MXFP4_BLOCK_SIZE
            if pad:
                vr = torch.nn.functional.pad(vr, (0, 0, 0, 0, 0, pad))  # pad token dim
            num_blocks = vr.shape[0] // _MXFP4_BLOCK_SIZE
            vr = vr.view(num_blocks, _MXFP4_BLOCK_SIZE, self.num_kv_heads, self.head_size)
            vr = self._mxfp4_quant_tf_grouped(vr)
            vr = vr.reshape(num_blocks * _MXFP4_BLOCK_SIZE, self.num_kv_heads, self.head_size)
            parts.append(vr[:Lr])
        if not parts:
            return value
        out = torch.cat(parts, dim=0)
        # Diagnostic counters (retained): how much V changed after quantization.
        n_changed = (out.detach() != value.detach()).sum().item()
        max_abs_diff = (out.detach() - value.detach()).abs().max().item()
        return out

    def _get_prefill_len(self, attn_metadata: AscendMetadata, req_idx: int) -> int:
        """Per-request prefill length (decode anchor) for windowed V quant.

        Returns 0 when ``num_prompt_tokens_list`` is None (async_spec_decode
        fallback), treating the whole sequence as decode-generated.
        """
        npt = attn_metadata.num_prompt_tokens_list
        if npt is None:
            return 0
        return int(npt[req_idx])

    def _quantize_window_v_slots_eager(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
    ) -> None:
        """Eager windowed V quantization.

        For each decode request whose ``decode_len = seq_len - prefill_len``
        is a positive multiple of ``_MXFP4_BLOCK_SIZE`` (32), the most recent
        32 V tokens (``[seq_len-32, seq_len-1]``) are gathered from the cache,
        MXFP4-quantized along the token axis, and scattered back in-place.
        Only V is touched; K is left untouched. Python control flow + .item().
        """
        if self.value_cache is None:
            return
        seq_lens_list = attn_metadata.seq_lens_list
        if not seq_lens_list:
            return
        num_decodes = attn_metadata.num_decodes
        if num_decodes <= 0:
            return
        block_tables = attn_metadata.block_tables
        if block_tables is None:
            return

        block_size = self.value_cache.shape[1]
        device = self.value_cache.device
        W = _MXFP4_BLOCK_SIZE
        HPW = HIGH_PRECISION_WINDOW_SIZE

        evicted_slots = []
        for req_idx in range(num_decodes):
            seq_len = seq_lens_list[req_idx]
            prefill_len = self._get_prefill_len(attn_metadata, req_idx)
            decode_len = seq_len - prefill_len
            # Trigger only when the decode region filled a whole 32-window.
            if decode_len < HPW + W or ((decode_len - HPW) % W) != 0:
                continue
            start_pos = seq_len - HPW - W
            end_pos = seq_len - HPW  # exclusive
            for pos in range(start_pos, end_pos):
                block_idx = pos // block_size
                offset_in_block = pos % block_size
                phys_block = block_tables[req_idx, block_idx].item()
                if phys_block >= 0:
                    evicted_slots.append(phys_block * block_size + offset_in_block)

        if not evicted_slots:
            return

        evicted_indices = torch.tensor(evicted_slots, dtype=torch.long, device=device)
        flat_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)
        v_win = flat_value[evicted_indices]                     # [N, H, D], N = num_triggers*W
        # reshape so each 32 consecutive tokens form one MXFP4 group
        v_grouped = v_win.view(-1, _MXFP4_BLOCK_SIZE, self.num_kv_heads, self.head_size)
        v_quantized = self._mxfp4_quant_tf_grouped(v_grouped).reshape(v_win.shape)
        # Diagnostic counters (retained): how much the decode-window V changed.
        n_changed = (v_quantized.detach() != v_win.detach()).sum().item()
        max_abs_diff = (v_quantized.detach() - v_win.detach()).abs().max().item()
        flat_value[evicted_indices] = v_quantized

    def _quantize_window_k_slots_eager(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
    ) -> None:
        """Eager batched-FIFO windowed K quantization.

        Keeps the most recent ``HPW`` (=K_HIGH_PRECISION_WINDOW) decode K tokens
        high-precision in the cache. Each time the region ahead of that tail
        window has accumulated a whole ``W`` (=_MXFP4_BLOCK_SIZE) batch — i.e.
        decode_len >= HPW+W and (decode_len-HPW)%W==0 — the oldest
        not-yet-quantized ``W`` K tokens at absolute positions
        ``[seq_len-HPW-W, seq_len-HPW-1]`` are gathered and MXFP4-quantized
        **along the head_size axis** (each token independent, 32 channels per
        group), then scattered back in-place.
        Uses ``.item()`` and Python control flow -> eager only.
        """
        if self.key_cache is None:
            return
        seq_lens_list = attn_metadata.seq_lens_list
        if not seq_lens_list:
            return
        num_decodes = attn_metadata.num_decodes
        if num_decodes <= 0:
            return
        block_tables = attn_metadata.block_tables
        if block_tables is None:
            return

        block_size = self.key_cache.shape[1]
        device = self.key_cache.device
        W = _MXFP4_BLOCK_SIZE
        HPW = K_HIGH_PRECISION_WINDOW

        evicted_slots = []
        for req_idx in range(num_decodes):
            seq_len = seq_lens_list[req_idx]
            prefill_len = self._get_prefill_len(attn_metadata, req_idx)
            decode_len = seq_len - prefill_len
            # Trigger only when the region ahead of the high-precision tail
            # window (HPW) has accumulated a whole W-token batch to quantize:
            # decode_len in {96, 128, 160, ...}
            #  i.e. >= HPW+W and (decode_len-HPW)%W==0.
            if decode_len < HPW + W or ((decode_len - HPW) % W) != 0:
                continue
            # Oldest W not-yet-quantized tokens: [seq_len-HPW-W, seq_len-HPW-1].
            start_pos = seq_len - HPW - W
            end_pos = seq_len - HPW  # exclusive
            for pos in range(start_pos, end_pos):
                block_idx = pos // block_size
                offset_in_block = pos % block_size
                phys_block = block_tables[req_idx, block_idx].item()
                if phys_block >= 0:
                    evicted_slots.append(phys_block * block_size + offset_in_block)

        if not evicted_slots:
            return

        evicted_indices = torch.tensor(evicted_slots, dtype=torch.long, device=device)
        flat_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        k_win = flat_key[evicted_indices]                       # [N, H, D], N = num_triggers*W
        # Quantize along head_size (-1): each token independent, no token-axis
        # grouping. Math identical to the prefill K path (matmul(k,rot_h) was
        # already applied at write time, so cache holds rotated K).
        k_quantized = self._mxfp4_quant_tf(k_win, -1)
        # Diagnostic counters (retained): how much the decode-window K changed.
        n_changed = (k_quantized.detach() != k_win.detach()).sum().item()
        max_abs_diff = (k_quantized.detach() - k_win.detach()).abs().max().item()
        flat_key[evicted_indices] = k_quantized

    def _quantize_window_v_slots_graph(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
        num_decodes: int,
    ) -> None:
        """Graph-compatible batched windowed V quantization.

        Pure tensor ops, no ``.item()``, no data-dependent control flow, fixed
        shape ``(num_decodes, W)`` so it can be traced into ACLGraph. Processes
        only the first ``num_decodes`` rows (real decode requests). Invalid
        (non-triggered) rows write back their own window slots' original values
        (a no-op): per-request physical blocks are exclusive, window slots never
        overlap across requests -> no write race, no scratch buffer needed.

        seq_lens / prefill_len come from pre-allocated buffers refreshed
        in-place by :meth:`update_graph_params`. Trigger uses
        ``decode_len = seq_len - prefill_len`` so the window never crosses the
        prefill/decode boundary. Only V is quantized.
        """
        if self.value_cache is None:
            return
        if num_decodes <= 0:
            return

        seq_lens_t = self._seq_lens_buffer[:num_decodes]      # (num_decodes,)
        prefill_t = self._prefill_len_buffer[:num_decodes]    # (num_decodes,)    
        block_tables_t = attn_metadata.block_tables           # (max_num_seqs, max_blocks)

        W = _MXFP4_BLOCK_SIZE
        HPW = HIGH_PRECISION_WINDOW_SIZE
        block_size = self.value_cache.shape[1]
        device = self.value_cache.device

        decode_len = seq_lens_t - prefill_t                                    # (num_decodes,)
        # Trigger when the region ahead of the tail window (HPW) holds a full
        # W-token batch: decode_len in {96,128,160,...}
        #  i.e. >= HPW+W and (decode_len-HPW)%W==0.
        window_filled_mask = (decode_len >= HPW + W) & (((decode_len - HPW) % W) == 0)

        # Logical positions of the W-token window per request: [seq_len-W, seq_len-1].
        rel_positions = torch.arange(-HPW - W, -HPW, device=device)            # (W,)
        positions = seq_lens_t.unsqueeze(1) + rel_positions.unsqueeze(0)       # (num_decodes, W)
        safe_positions = torch.clamp(positions, min=0)
        block_idx = (safe_positions // block_size).long()                      # (num_decodes, W)
        offset_in_block = (safe_positions % block_size).long()                 # (num_decodes, W)

        row_indices = torch.arange(num_decodes, device=device).unsqueeze(1).expand(-1, W)
        phys_blocks = block_tables_t[row_indices, block_idx]                   # (num_decodes, W)

        all_slots_grid = (phys_blocks * block_size + offset_in_block).long()   # (num_decodes, W)
        slot_mask = window_filled_mask.unsqueeze(1) & (phys_blocks >= 0)       # (num_decodes, W)

        flat_slots = all_slots_grid.flatten()                                  # (num_decodes*W,)
        flat_write_mask = slot_mask.flatten()                                  # (num_decodes*W,)

        flat_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)
        v_win = flat_value[flat_slots]                                         # (num_decodes*W, H, D)
        # group as (num_decodes, W, H, D) -> one MXFP4 block per request window
        v_grouped = v_win.view(num_decodes, W, self.num_kv_heads, self.head_size)
        v_quantized = self._mxfp4_quant_tf_grouped(v_grouped).reshape(v_win.shape)

        write_mask = flat_write_mask.unsqueeze(-1).unsqueeze(-1)              # (num_decodes*W, 1, 1)
        updated_v = torch.where(write_mask, v_quantized, v_win)
        flat_value[flat_slots] = updated_v

    def _quantize_window_k_slots_graph(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
        num_decodes: int,
    ) -> None:
        """Graph-compatible batched-FIFO windowed K quantization.

        Pure tensor ops, no ``.item()``, no data-dependent control flow, fixed
        shape ``(num_decodes, W)`` -> ACLGraph-traceable. Keeps the most recent
        ``HPW`` (=K_HIGH_PRECISION_WINDOW) decode K tokens high-precision; when
        the region ahead of that tail window holds a full ``W`` (=_MXFP4_BLOCK_SIZE)
        batch — decode_len >= HPW+W and (decode_len-HPW)%W==0 — the oldest
        not-yet-quantized ``W`` K tokens (absolute positions
        ``[seq_len-HPW-W, seq_len-HPW-1]``) are MXFP4-quantized **along the
        head_size axis** (each token independent) and scattered back in-place.
        Non-triggered rows write back their own slots' original values (no-op):
        per-request physical blocks are exclusive -> no write race.
        """
        if self.key_cache is None:
            return
        if num_decodes <= 0:
            return

        seq_lens_t = self._seq_lens_buffer[:num_decodes]      # (num_decodes,)
        prefill_t = self._prefill_len_buffer[:num_decodes]    # (num_decodes,)
        block_tables_t = attn_metadata.block_tables           # (max_num_seqs, max_blocks)

        W = _MXFP4_BLOCK_SIZE
        HPW = K_HIGH_PRECISION_WINDOW
        block_size = self.key_cache.shape[1]
        device = self.key_cache.device

        decode_len = seq_lens_t - prefill_t                                    # (num_decodes,)
        # Trigger when the region ahead of the tail window (HPW) holds a full
        # W-token batch: decode_len in {96,128,160,...}
        #  i.e. >= HPW+W and (decode_len-HPW)%W==0.
        batch_ready_mask = (decode_len >= HPW + W) & (((decode_len - HPW) % W) == 0)

        # Logical positions of the oldest W not-yet-quantized tokens:
        # [seq_len-HPW-W, seq_len-HPW-1]  -> relative offsets arange(-HPW-W, -HPW).
        rel_positions = torch.arange(-HPW - W, -HPW, device=device)            # (W,)
        positions = seq_lens_t.unsqueeze(1) + rel_positions.unsqueeze(0)       # (num_decodes, W)
        safe_positions = torch.clamp(positions, min=0)
        block_idx = (safe_positions // block_size).long()                      # (num_decodes, W)
        offset_in_block = (safe_positions % block_size).long()                 # (num_decodes, W)

        row_indices = torch.arange(num_decodes, device=device).unsqueeze(1).expand(-1, W)
        phys_blocks = block_tables_t[row_indices, block_idx]                   # (num_decodes, W)

        all_slots_grid = (phys_blocks * block_size + offset_in_block).long()   # (num_decodes, W)
        slot_mask = batch_ready_mask.unsqueeze(1) & (phys_blocks >= 0)         # (num_decodes, W)

        flat_slots = all_slots_grid.flatten()                                  # (num_decodes*W,)
        flat_write_mask = slot_mask.flatten()                                  # (num_decodes*W,)

        flat_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        k_win = flat_key[flat_slots]                                           # (num_decodes*W, H, D)
        # Quantize along head_size (-1): each token independent (no token-axis
        # grouping). cache holds rotated K (matmul(k,rot_h) at write time), so
        # Q*Kt rotation cancellation still holds after this MXFP4 step.
        k_quantized = self._mxfp4_quant_tf(k_win, -1)

        write_mask = flat_write_mask.unsqueeze(-1).unsqueeze(-1)               # (num_decodes*W, 1, 1)
        updated_k = torch.where(write_mask, k_quantized, k_win)
        flat_key[flat_slots] = updated_k

    # ============ HiF4 windowed-decode quantization (mirror of MXFP4) ============ #
    # Same gather/trigger/write-mask skeleton as the MXFP4 versions above; only
    # the quant kernel differs. K is quantized on head_size dim (each token
    # independent). V is quantized on the seq_len dim in groups of V_HIF4_WINDOW
    # tokens (matches MXFP4 V window semantics); _hif4_quant_v_seq does the
    # reshape. The HiF4 kernel internally groups the last dim in 64-scalar
    # blocks, so one window of V_HIF4_WINDOW tokens yields V_HIF4_WINDOW/64 such
    # groups (2 groups when V_HIF4_WINDOW=128).

    def _hif4_quant_v_seq(self, v_win: torch.Tensor) -> torch.Tensor:
        """HiF4-quantize V along the seq_len (token) dim, in groups of W tokens.

        ``v_win`` is ``[num_trig*W, H, D]`` where ``W = V_HIF4_WINDOW``. The HiF4
        kernel groups the *last* dim into 64-scalar blocks, so we reshape +
        permute to put the W tokens of each group on the last dim
        (-> ``[num_trig, H, D, W]``), run the unchanged kernel (which splits W
        into W/64 sub-blocks), then permute back. This quantizes V along the
        token axis — matching the MXFP4 V window semantics. head_size (D) is NOT
        quantized here (it is not on the last dim).
        """
        W = V_HIF4_WINDOW
        num_trig = v_win.shape[0] // W
        H = self.num_kv_heads
        D = self.head_size
        vg = v_win.view(num_trig, W, H, D)                 # [num_trig, W, H, D]
        vt = vg.permute(0, 2, 3, 1).contiguous()           # [num_trig, H, D, W]
        qt = self._quantize_hif4_kernel(vt, -1)            # W tokens -> W/64 groups
        qg = qt.permute(0, 3, 1, 2).contiguous()           # [num_trig, W, H, D]
        return qg.reshape(v_win.shape)

    def _quantize_window_v_slots_hif4_eager(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
    ) -> None:
        """Eager windowed V quantization (HiF4 kernel, seq_len dim, W/group).

        For each decode request whose ``decode_len = seq_len - prefill_len`` is
        a positive multiple of ``V_HIF4_WINDOW`` (W), the most recent W V tokens
        (``[seq_len-W, seq_len-1]``) are gathered, HiF4-quantized along the
        **token axis** (tokens share a three-level scale, head_size kept), and
        scattered back in-place. Sink protection is implicit: ``decode_len`` is
        anchored at ``prefill_len`` so the window never crosses into the
        prefill (sink) region. Python control flow + .item() -> eager only.
        """
        if self.value_cache is None:
            return
        seq_lens_list = attn_metadata.seq_lens_list
        if not seq_lens_list:
            return
        num_decodes = attn_metadata.num_decodes
        if num_decodes <= 0:
            return
        block_tables = attn_metadata.block_tables
        if block_tables is None:
            return

        block_size = self.value_cache.shape[1]
        device = self.value_cache.device
        W = V_HIF4_WINDOW

        evicted_slots = []
        for req_idx in range(num_decodes):
            seq_len = seq_lens_list[req_idx]
            prefill_len = self._get_prefill_len(attn_metadata, req_idx)
            decode_len = seq_len - prefill_len
            # Trigger only when the decode region filled a whole 64-window.
            if decode_len < W or (decode_len % W) != 0:
                continue
            start_pos = seq_len - W
            for pos in range(start_pos, seq_len):
                block_idx = pos // block_size
                offset_in_block = pos % block_size
                phys_block = block_tables[req_idx, block_idx].item()
                if phys_block >= 0:
                    evicted_slots.append(phys_block * block_size + offset_in_block)

        if not evicted_slots:
            return

        evicted_indices = torch.tensor(evicted_slots, dtype=torch.long, device=device)
        flat_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)
        v_win = flat_value[evicted_indices]                     # [num_trig*W, H, D]
        # HiF4 along seq_len dim (W tokens/group), head_size kept intact.
        v_quantized = self._hif4_quant_v_seq(v_win)
        flat_value[evicted_indices] = v_quantized

    def _quantize_window_k_slots_hif4_eager(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
    ) -> None:
        """Eager batched-FIFO windowed K quantization (HiF4 kernel).

        Same trigger/gather as :meth:`_quantize_window_k_slots_eager`: keeps the
        most recent ``HPW`` (=K_HIGH_PRECISION_WINDOW) decode K tokens
        high-precision; when the region ahead of that tail holds a full ``W``
        (=_MXFP4_BLOCK_SIZE) batch (decode_len >= HPW+W and
        (decode_len-HPW)%W==0), the oldest ``W`` K tokens at
        ``[seq_len-HPW-W, seq_len-HPW-1]`` are HiF4-quantized along head_size.
        This method only quantizes; if ``_ENABLE_ROT_H`` is set the cached K was
        already rotated at write time. Sink protection is implicit:
        ``decode_len`` is anchored at ``prefill_len`` (see
        :meth:`_get_prefill_len`), so the quantized region never crosses into
        the prefill (sink) segment.
        """
        if self.key_cache is None:
            return
        seq_lens_list = attn_metadata.seq_lens_list
        if not seq_lens_list:
            return
        num_decodes = attn_metadata.num_decodes
        if num_decodes <= 0:
            return
        block_tables = attn_metadata.block_tables
        if block_tables is None:
            return

        block_size = self.key_cache.shape[1]
        device = self.key_cache.device
        W = _MXFP4_BLOCK_SIZE
        HPW = K_HIGH_PRECISION_WINDOW

        evicted_slots = []
        for req_idx in range(num_decodes):
            seq_len = seq_lens_list[req_idx]
            prefill_len = self._get_prefill_len(attn_metadata, req_idx)
            decode_len = seq_len - prefill_len
            # Trigger when region ahead of tail window (HPW) holds a full W batch.
            if decode_len < HPW + W or ((decode_len - HPW) % W) != 0:
                continue
            start_pos = seq_len - HPW - W
            end_pos = seq_len - HPW  # exclusive
            for pos in range(start_pos, end_pos):
                block_idx = pos // block_size
                offset_in_block = pos % block_size
                phys_block = block_tables[req_idx, block_idx].item()
                if phys_block >= 0:
                    evicted_slots.append(phys_block * block_size + offset_in_block)

        if not evicted_slots:
            return

        evicted_indices = torch.tensor(evicted_slots, dtype=torch.long, device=device)
        flat_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        k_win = flat_key[evicted_indices]                       # [N, H, D]
        k_quantized = self._quant_dequant_hif4(k_win)
        flat_key[evicted_indices] = k_quantized

    def _quantize_window_v_slots_hif4_graph(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
        num_decodes: int,
    ) -> None:
        """Graph-compatible windowed V quantization (HiF4 kernel, seq_len, W/group).

        Pure tensor ops, fixed shape ``(num_decodes, W)`` -> ACLGraph-traceable.
        Same skeleton as :meth:`_quantize_window_v_slots_graph`; V quantized on
        the **token axis** via :meth:`_hif4_quant_v_seq` (tokens share a HiF4
        scale, head_size kept). Sink protection implicit via
        ``decode_len = seq_len - prefill_len`` (window stays in decode region).
        """
        if self.value_cache is None:
            return
        if num_decodes <= 0:
            return

        seq_lens_t = self._seq_lens_buffer[:num_decodes]      # (num_decodes,)
        prefill_t = self._prefill_len_buffer[:num_decodes]    # (num_decodes,)
        block_tables_t = attn_metadata.block_tables           # (max_num_seqs, max_blocks)

        W = V_HIF4_WINDOW
        block_size = self.value_cache.shape[1]
        device = self.value_cache.device

        decode_len = seq_lens_t - prefill_t                                    # (num_decodes,)
        window_filled_mask = (decode_len >= W) & ((decode_len % W) == 0)       # (num_decodes,)

        rel_positions = torch.arange(-W, 0, device=device)                     # (W,)
        positions = seq_lens_t.unsqueeze(1) + rel_positions.unsqueeze(0)       # (num_decodes, W)
        safe_positions = torch.clamp(positions, min=0)
        block_idx = (safe_positions // block_size).long()                      # (num_decodes, W)
        offset_in_block = (safe_positions % block_size).long()                 # (num_decodes, W)

        row_indices = torch.arange(num_decodes, device=device).unsqueeze(1).expand(-1, W)
        phys_blocks = block_tables_t[row_indices, block_idx]                   # (num_decodes, W)

        all_slots_grid = (phys_blocks * block_size + offset_in_block).long()   # (num_decodes, W)
        slot_mask = window_filled_mask.unsqueeze(1) & (phys_blocks >= 0)       # (num_decodes, W)

        flat_slots = all_slots_grid.flatten()                                  # (num_decodes*W,)
        flat_write_mask = slot_mask.flatten()                                  # (num_decodes*W,)

        flat_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)
        v_win = flat_value[flat_slots]                                         # (num_decodes*W, H, D)
        # HiF4 along seq_len dim (W tokens/group); num_trig == num_decodes here.
        v_quantized = self._hif4_quant_v_seq(v_win)

        write_mask = flat_write_mask.unsqueeze(-1).unsqueeze(-1)              # (num_decodes*W, 1, 1)
        updated_v = torch.where(write_mask, v_quantized, v_win)
        flat_value[flat_slots] = updated_v

    def _quantize_window_k_slots_hif4_graph(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
        num_decodes: int,
    ) -> None:
        """Graph-compatible batched-FIFO windowed K quantization (HiF4 kernel).

        Same skeleton as :meth:`_quantize_window_k_slots_graph`. Keeps the most
        recent ``HPW`` (=K_HIGH_PRECISION_WINDOW) decode K tokens high-precision;
        when the region ahead of the tail holds a full ``W`` (=_MXFP4_BLOCK_SIZE)
        batch (decode_len >= HPW+W and (decode_len-HPW)%W==0), the oldest ``W`` K
        tokens at ``[seq_len-HPW-W, seq_len-HPW-1]`` are HiF4-quantized along
        head_size. This method only quantizes; if ``_ENABLE_ROT_H`` is set the
        cached K was already rotated at write time. Sink protection implicit via
        ``decode_len = seq_len - prefill_len``.
        """
        if self.key_cache is None:
            return
        if num_decodes <= 0:
            return

        seq_lens_t = self._seq_lens_buffer[:num_decodes]      # (num_decodes,)
        prefill_t = self._prefill_len_buffer[:num_decodes]    # (num_decodes,)
        block_tables_t = attn_metadata.block_tables           # (max_num_seqs, max_blocks)

        W = _MXFP4_BLOCK_SIZE
        HPW = K_HIGH_PRECISION_WINDOW
        block_size = self.key_cache.shape[1]
        device = self.key_cache.device

        decode_len = seq_lens_t - prefill_t                                    # (num_decodes,)
        batch_ready_mask = (decode_len >= HPW + W) & (((decode_len - HPW) % W) == 0)

        rel_positions = torch.arange(-HPW - W, -HPW, device=device)            # (W,)
        positions = seq_lens_t.unsqueeze(1) + rel_positions.unsqueeze(0)       # (num_decodes, W)
        safe_positions = torch.clamp(positions, min=0)
        block_idx = (safe_positions // block_size).long()                      # (num_decodes, W)
        offset_in_block = (safe_positions % block_size).long()                 # (num_decodes, W)

        row_indices = torch.arange(num_decodes, device=device).unsqueeze(1).expand(-1, W)
        phys_blocks = block_tables_t[row_indices, block_idx]                   # (num_decodes, W)

        all_slots_grid = (phys_blocks * block_size + offset_in_block).long()   # (num_decodes, W)
        slot_mask = batch_ready_mask.unsqueeze(1) & (phys_blocks >= 0)         # (num_decodes, W)

        flat_slots = all_slots_grid.flatten()                                  # (num_decodes*W,)
        flat_write_mask = slot_mask.flatten()                                  # (num_decodes*W,)

        flat_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        k_win = flat_key[flat_slots]                                           # (num_decodes*W, H, D)
        k_quantized = self._quant_dequant_hif4(k_win)

        write_mask = flat_write_mask.unsqueeze(-1).unsqueeze(-1)               # (num_decodes*W, 1, 1)
        updated_k = torch.where(write_mask, k_quantized, k_win)
        flat_key[flat_slots] = updated_k

    # =========== end HiF4 windowed-decode quantization =========== #

    # ============================ Legacy C4 quant helpers ============================ #
    # NOTE: the methods below (_prepare_c4_scales, _quantize_kv_to_fp4_sink x2,
    # _quantize_kv_to_fp4, _quantize_evicted_kv_slots{,_eager,_with_sink}) are
    # legacy C4 (INT4 symmetric, clamp/round to [-6, 6]) pseudo-quant helpers.
    # They are NOT on the active HiF4/MXFP4 forward path and are kept only as a
    # reference for the sink-masking / eviction pattern that _quantize_hif4_with_sink
    # and the windowed methods were derived from. Do not wire them into forward.

    def _prepare_c4_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C4 scales to this TP rank and pre-compute BF16
        BNSD antiquant tensors (scale-only, no offset). Legacy; unused on the
        active path.
        """
        if hasattr(layer, "_c4_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c4_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c4_v_scale = _shard_and_reshape(layer.v_cache_scale.data)

        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        layer._c4_k_aq_scale = layer._c4_k_scale.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c4_v_aq_scale = layer._c4_v_scale.to(torch.bfloat16).view(bnsd).contiguous()

        layer._c4_k_inv_scale_bf16 = (1.0 / layer._c4_k_scale).to(torch.bfloat16)
        layer._c4_v_inv_scale_bf16 = (1.0 / layer._c4_v_scale).to(torch.bfloat16)

        layer._c4_scales_prepared = True

    def _quantize_kv_to_fp4_sink(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
        query_start_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pseudo-quantize K/V, keeping the first `sink_size` tokens of each
        sequence unquantized. Legacy; superseded by _quantize_hif4_with_sink.
        """
        self._prepare_c4_scales(layer, key.device)
        dtype = key.dtype
        sink_size = ATTENTION_SINK_SIZE

        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        # Per-token relative index within its request (via query_start_loc).
        global_idx = torch.arange(num_actual_tokens, device=key.device)
        seq_ids = torch.bucketize(global_idx, query_start_loc, right=True) - 1
        token_start_locs = query_start_loc[seq_ids]
        relative_idx = global_idx - token_start_locs

        # Quantize everything except the first sink_size tokens of each request.
        quant_mask = (relative_idx >= sink_size).view(-1, 1, 1)

        k_qdq_all = (torch.clamp(
            torch.round(actual_key * layer._c4_k_inv_scale_bf16),
            -6,
            6,
        ) / layer._c4_k_inv_scale_bf16).to(dtype)

        v_qdq_all = (torch.clamp(
            torch.round(actual_value * layer._c4_v_inv_scale_bf16),
            -6,
            6,
        ) / layer._c4_v_inv_scale_bf16).to(dtype)

        k_qdq = torch.where(quant_mask, k_qdq_all, actual_key)
        v_qdq = torch.where(quant_mask, v_qdq_all, actual_value)

        return k_qdq, v_qdq

    def _quantize_kv_to_fp4_sink(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
        query_start_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pseudo-quantize K/V, keeping the first `sink_size` tokens of each
        sequence unquantized. Variant using query_start_loc[1:]. Legacy.
        """
        self._prepare_c4_scales(layer, key.device)
        dtype = key.dtype
        sink_size = ATTENTION_SINK_SIZE
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        global_idx = torch.arange(num_actual_tokens, device=key.device)
        seq_ids = torch.bucketize(global_idx, query_start_loc[1:], right=True)
        seq_start_locs = query_start_loc[seq_ids]
        relative_idx = global_idx - seq_start_locs

        quant_mask = (relative_idx >= sink_size).view(-1, 1, 1)

        k_qdq_all = (torch.clamp(
            torch.round(actual_key * layer._c4_k_inv_scale_bf16),
            -6,
            6,
        ) / layer._c4_k_inv_scale_bf16).to(dtype)

        v_qdq_all = (torch.clamp(
            torch.round(actual_value * layer._c4_v_inv_scale_bf16),
            -6,
            6,
        ) / layer._c4_v_inv_scale_bf16).to(dtype)

        k_qdq = torch.where(quant_mask, k_qdq_all, actual_key)
        v_qdq = torch.where(quant_mask, v_qdq_all, actual_value)

        return k_qdq, v_qdq

    def _quantize_kv_to_fp4(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pseudo-quantize K/V: clamp(round(x * inv_scale), -6, 6) / inv_scale. Legacy."""
        self._prepare_c4_scales(layer, key.device)
        dtype = key.dtype
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_qdq = (torch.clamp(
            torch.round(actual_key * layer._c4_k_inv_scale_bf16),
            -6,
            6,
        ) / layer._c4_k_inv_scale_bf16).to(dtype)
        v_qdq = (torch.clamp(
            torch.round(actual_value * layer._c4_v_inv_scale_bf16),
            -6,
            6,
        ) / layer._c4_v_inv_scale_bf16).to(dtype)
        return k_qdq, v_qdq

    def _quantize_evicted_kv_slots(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
        _seq_lens_buffer: torch.Tensor,
    ) -> None:
        if self.key_cache is None or self.value_cache is None:
            return

        num_decodes = attn_metadata.num_decodes
        if num_decodes <= 0:
            return

        block_tables = attn_metadata.block_tables  # (batch_size, max_blocks_per_seq)
        block_size = self.key_cache.shape[1]
        device = self.key_cache.device

        m = 0
        n = HIGH_PRECISION_WINDOW_SIZE

        active_blocks = block_tables[:num_decodes]

        approx_seq_lens = _seq_lens_buffer[:num_decodes]

        evicted_pos = approx_seq_lens - 1 - n  # (num_decodes,)
        valid_evict_mask = (evicted_pos >= 0) & (evicted_pos >= m)  # (num_decodes,)

        # Clamp negatives/out-of-range to 0 to avoid OOB indexing; masked out on write.
        safe_evicted_pos = torch.clamp(evicted_pos, min=0)
        target_block_indices = safe_evicted_pos // block_size  # (num_decodes,)
        offset_in_blocks = safe_evicted_pos % block_size       # (num_decodes,)

        row_indices = torch.arange(num_decodes, device=device)
        phys_blocks = active_blocks[row_indices, target_block_indices]  # (num_decodes,)

        evicted_slots = phys_blocks * block_size + offset_in_blocks  # (num_decodes,)

        # Map unallocated slots (phys_blocks == -1) to a safe 0 position.
        safe_slots_mask = valid_evict_mask & (phys_blocks >= 0)
        final_slots = torch.where(safe_slots_mask, evicted_slots, torch.zeros_like(evicted_slots))

        # In-place pseudo-quantization of the evicted KV slots.
        flat_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        flat_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)

        k_evicted = flat_key[final_slots]
        v_evicted = flat_value[final_slots]

        k_quantized = (torch.clamp(
            torch.round(k_evicted * layer._c4_k_inv_scale_bf16), -6, 6
        ) / layer._c4_k_inv_scale_bf16).to(k_evicted.dtype)

        v_quantized = (torch.clamp(
            torch.round(v_evicted * layer._c4_v_inv_scale_bf16), -6, 6
        ) / layer._c4_v_inv_scale_bf16).to(v_evicted.dtype)

        write_mask = safe_slots_mask.unsqueeze(-1).unsqueeze(-1)  # (num_decodes, 1, 1)

        k_final = torch.where(write_mask, k_quantized, k_evicted)
        v_final = torch.where(write_mask, v_quantized, v_evicted)

        # Scatter write-back in-place.
        flat_key[final_slots] = k_final
        flat_value[final_slots] = v_final

    def _quantize_evicted_kv_slots_eager(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
    ) -> None:
        """Eager mode (no attention sink): in-place pseudo-quantize the single
        old KV slot that just slid out of the high-precision window
        (HIGH_PRECISION_WINDOW_SIZE). Legacy.
        """
        if self.key_cache is None or self.value_cache is None:
            return

        seq_lens_list = attn_metadata.seq_lens_list
        if not seq_lens_list:
            return

        num_decodes = attn_metadata.num_decodes
        if num_decodes <= 0:
            return

        block_tables = attn_metadata.block_tables
        if block_tables is None:
            return

        block_size = self.key_cache.shape[1]
        n = HIGH_PRECISION_WINDOW_SIZE

        evicted_slots = []
        for req_idx in range(num_decodes):
            seq_len = seq_lens_list[req_idx]
            # Logical token position that just slid out of the window.
            evicted_pos = seq_len - 1 - n

            if evicted_pos < 0:
                continue

            block_idx = evicted_pos // block_size
            offset_in_block = evicted_pos % block_size

            phys_block = block_tables[req_idx, block_idx].item()
            if phys_block >= 0:
                flat_slot = phys_block * block_size + offset_in_block
                evicted_slots.append(flat_slot)

        if not evicted_slots:
            return

        device = self.key_cache.device
        evicted_indices = torch.tensor(evicted_slots, dtype=torch.long, device=device)

        flat_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        flat_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)

        k_evicted = flat_key[evicted_indices]
        v_evicted = flat_value[evicted_indices]

        k_quantized = (torch.clamp(
            torch.round(k_evicted * layer._c4_k_inv_scale_bf16), -6, 6
        ) / layer._c4_k_inv_scale_bf16).to(k_evicted.dtype)

        v_quantized = (torch.clamp(
            torch.round(v_evicted * layer._c4_v_inv_scale_bf16), -6, 6
        ) / layer._c4_v_inv_scale_bf16).to(v_evicted.dtype)

        flat_key[evicted_indices] = k_quantized
        flat_value[evicted_indices] = v_quantized

    def _quantize_evicted_kv_slots_with_sink(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
    ) -> None:
        """Eager mode with attention-sink protection: the first
        ATTENTION_SINK_SIZE tokens always stay full-precision. Legacy.
        """
        if self.key_cache is None or self.value_cache is None:
            return

        seq_lens_list = attn_metadata.seq_lens_list
        if not seq_lens_list:
            return

        num_decodes = attn_metadata.num_decodes
        if num_decodes <= 0:
            return

        block_tables = attn_metadata.block_tables
        if block_tables is None:
            return

        block_size = self.key_cache.shape[1]
        m = ATTENTION_SINK_SIZE
        n = HIGH_PRECISION_WINDOW_SIZE

        evicted_slots = []
        for req_idx in range(num_decodes):
            seq_len = seq_lens_list[req_idx]
            evicted_pos = seq_len - 1 - n

            # Skip if still inside the high-precision window or in the sink region.
            if evicted_pos < 0 or evicted_pos < m:
                continue

            block_idx = evicted_pos // block_size
            offset_in_block = evicted_pos % block_size

            phys_block = block_tables[req_idx, block_idx].item()
            if phys_block >= 0:
                flat_slot = phys_block * block_size + offset_in_block
                evicted_slots.append(flat_slot)

        if not evicted_slots:
            return

        device = self.key_cache.device
        evicted_indices = torch.tensor(evicted_slots, dtype=torch.long, device=device)

        flat_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        flat_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)

        k_evicted = flat_key[evicted_indices]
        v_evicted = flat_value[evicted_indices]

        k_quantized = (torch.clamp(
            torch.round(k_evicted * layer._c4_k_inv_scale_bf16), -6, 6
        ) / layer._c4_k_inv_scale_bf16).to(k_evicted.dtype)

        v_quantized = (torch.clamp(
            torch.round(v_evicted * layer._c4_v_inv_scale_bf16), -6, 6
        ) / layer._c4_v_inv_scale_bf16).to(v_evicted.dtype)

        flat_key[evicted_indices] = k_quantized
        flat_value[evicted_indices] = v_quantized

    # ======================== end legacy C4 quant helpers ======================= #


class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention backend implementation for INT8 KV cache (C8/QuaRot) models.

    This subclass handles static per-channel INT8 KV cache quantization.
    It is activated via class surgery in AscendC8KVCacheAttentionMethod.create_weights
    (vllm_ascend/quantization/methods/kv_c8.py)
    so that C8 attention layers automatically use this forward path.
    """

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        self._prepare_c8_scales(layer, query.device)
        float_key, float_value = None, None
        if self.vllm_config.kv_transfer_config is None:
            if key is not None and value is not None:
                if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                    float_key, float_value = key, value
                key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                query, key, value, _ = self._reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
            # pooling model branch
            if attn_metadata.model_runner_type == "pooling":
                attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                return self._forward_c8_decode(query, attn_metadata, output, layer)
            elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
            else:
                return self._forward_c8_fused_infer_attention(
                    query,
                    float_key if float_key is not None else key,
                    float_value if float_value is not None else value,
                    attn_metadata,
                    output,
                    layer,
                )
        else:
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly and self.is_kv_producer:
                output_padded = None
                if key is not None and value is not None:
                    output_padded = output
                    query, key, value, output_padded = self.reshape_and_cache(
                        query, key, value, kv_cache, attn_metadata, output
                    )
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if output_padded is not None:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
                else:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            elif not self.is_kv_producer:
                if key is not None and value is not None:
                    key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                    query, key, value, _ = self._reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                    return self._forward_c8_decode(query, attn_metadata, output, layer)

    def _nz_5d_view(self, cache: torch.Tensor, block_size: int) -> torch.Tensor:
        """View a KV cache tensor in NZ 5D layout: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)."""
        NZ_FMT_LAST_DIM = 32
        return cache.view(-1, self.num_kv_heads, self.head_size // NZ_FMT_LAST_DIM, block_size, NZ_FMT_LAST_DIM)

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        layer._c8_k_inv_scale = 1.0 / layer._c8_k_scale
        layer._c8_v_inv_scale = 1.0 / layer._c8_v_scale

        nz_bnsd = (self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale_nz_bnsd = layer._c8_k_scale.view(nz_bnsd).contiguous()
        layer._c8_v_aq_scale_nz_bnsd = layer._c8_v_scale.view(nz_bnsd).contiguous()

        layer._c8_scales_prepared = True

    def _dequant_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list,
        target_dtype: torch.dtype,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 KV blocks and dequantize."""
        batch_size = block_table.shape[0]
        max_blocks_per_seq = block_table.shape[1]

        # NZ 5D view: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)
        block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        key_nz = self._nz_5d_view(key, block_size)
        value_nz = self._nz_5d_view(value, block_size)

        # Gather: (batch*max_blocks, H, D//nz, S, nz)
        gathered_k = key_nz[flat_ids]
        gathered_v = value_nz[flat_ids]
        # NZ→ND conversion: permute (S, H, D//nz, nz) → reshape (S, H, D)
        gathered_k = (
            gathered_k.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
        )
        gathered_v = (
            gathered_v.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
        )

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, self.num_kv_heads, self.head_size)[valid_mask]
        dense_v = gathered_v.view(-1, self.num_kv_heads, self.head_size)[valid_mask]

        # Scale-only dequant for NZ (symmetric)
        dense_k = dense_k.to(target_dtype) * layer._c8_k_scale
        dense_v = dense_v.to(target_dtype) * layer._c8_v_scale
        return dense_k, dense_v

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V from float to INT8 using static per-channel C8 scales."""
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale + layer._c8_k_offset),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale + layer._c8_v_offset),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 decode via FIA V1 BNSD with native paged INT8 KV + perchannel antiquant."""
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
        assert block_size % 32 == 0, f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
        batch_size = len(attn_metadata.seq_lens_list)

        key = self._nz_5d_view(self.key_cache, block_size)
        value = self._nz_5d_view(self.value_cache, block_size)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale_nz_bnsd,
            value_antiquant_scale=layer._c8_v_aq_scale_nz_bnsd,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            antiquant_mode=0,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            inner_precise=1,
            sparse_mode=0,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_c8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 ChunkedPrefill: decode via FIA V1 BNSD paged INT8 (zero gather),
        prefill via FIA V1 TND with float KV (new) or gather+dequant (continuing).
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
            assert block_size % 32 == 0, (
                f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
            )
            kv_k = self._nz_5d_view(self.key_cache, block_size)
            kv_v = self._nz_5d_view(self.value_cache, block_size)

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale_nz_bnsd,
                value_antiquant_scale=layer._c8_v_aq_scale_nz_bnsd,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                antiquant_mode=0,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                inner_precise=1,
                sparse_mode=0,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
                paged_k = self._nz_5d_view(self.key_cache, blk_size)
                paged_v = self._nz_5d_view(self.value_cache, blk_size)

                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype, layer
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            # block_table is None for prefill; FIA ignores block_size in this case.
            # Use cache block_size for consistency rather than a magic number.
            cache_block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_c8_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ):
        """C8 FIA V1 TND for prefill states (PrefillNoCache uses float KV directly,
        PrefillCacheHit gathers + dequants paged INT8 KV).
        """
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, list) else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(key, value, block_table, seq_lens, query.dtype, layer)
                block_table = None
                # block_table is None after dequant; FIA ignores block_size.
                # Use cache block_size for consistency rather than a magic number.
                block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
                actual_seq_lengths_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)
            else:
                key = (key.to(query.dtype) - layer._c8_k_offset) * layer._c8_k_scale
                value = (value.to(query.dtype) - layer._c8_v_offset) * layer._c8_v_scale

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output

    def _reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping

            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER

            # NZ write path: 5D view + npu_scatter_pa_kv_cache
            block_size = self.vllm_config.cache_config.block_size
            k_cache_layer = self._nz_5d_view(self.key_cache, block_size)
            v_cache_layer = self._nz_5d_view(self.value_cache, block_size)

            torch_npu.npu_scatter_pa_kv_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=k_cache_layer,
                value_cache=v_cache_layer,
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots,
            )

            notify_kv_cache_written()
        return query, key, value, output
