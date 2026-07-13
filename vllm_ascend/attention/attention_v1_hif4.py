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
import torch.nn.functional as F 
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

# default max value of sliding window size
SWA_INT_MAX = 2147483647
_ATTN_KEYS_BUFFER = None
# HIGH_PRECISION_WINDOW_SIZE = 64
# ATTENTION_SINK_SIZE = 64

HIGH_PRECISION_WINDOW_SIZE = 64
ATTENTION_SINK_SIZE = 0

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
            logger.info_once(f"update_graph_params run in sinks")
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
            # logger.info_once(f"update_graph_params run in FIA")
            
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
                    ) = param

                    # logger.info_once(f"update_graph_params run in with torch.npu.stream(update_stream):")
                    
                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        latest_cpu_seq_lens = attn_metadata[draft_step][key].seq_lens
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        block_tables = attn_metadata[draft_step][key].block_tables
                        attn_count = attn_count + 1
                        if not attn_metadata[draft_step][key].causal:
                            sparse_mode = 0
                    else:
                        latest_cpu_seq_lens = attn_metadata[key].seq_lens
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
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
                    
                    # 用于验证是否生效的，具体做法就是直接令值为 -1 ，在使用的时候构造 0 作为分母，必然报错
                    #     # evicted_kv_seq_lens_buffer[:num_reqs].copy_(latest_cpu_seq_lens*0+(-1), non_blocking=True)
                    #     # logger.info_once(f"evicted_kv_seq_lens_buffer: {evicted_kv_seq_lens_buffer}, latest_cpu_seq_lens: {latest_cpu_seq_lens}, attn_metadata[key].seq_lens: {attn_metadata[key].seq_lens}")
                    
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
            # logger.info_once(f"actual_seq_lengths_q: {actual_seq_lengths_q}, actual_seq_lengths_kv:{actual_seq_lengths_kv} ")
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
            )  # type: ignore
        else:
            attn_params = attn_params + (None, None, None, None, self._seq_lens_buffer)
            # attn_params = attn_params + (None, None, None, None)  # type: ignore
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
        # is_decode = attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        
        # logger.info_once(f"is_decode: {is_decode}")
        
        # self._prepare_c4_scales(layer, query.device)
        
        if key is not None and value is not None:
            output_padded = output
            # Prefill阶段: 全部伪量化再写入cache
            # Decode阶段: 延迟量化——新KV不量化直接写入, 稍后对滑出窗口的旧KV做原地伪量化
            
            # if not is_decode:
            #     key, value = self._quantize_kv_to_fp4(key, value, layer, attn_metadata.num_actual_tokens)
            # key, value = self._quantize_kv_to_fp4(key, value, layer, attn_metadata.num_actual_tokens)
            
            # key, value : [num_tokens, num_kv_heads, head_size]
            # logger.info_once(f"key.shape: {key.shape}, attn_metadata.num_actual_tokens:{attn_metadata.num_actual_tokens}")
            query = self._quant_dequant_hif4(query)
            key = self._quant_dequant_hif4(key)
            value = self._quant_dequant_hif4(value)

            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )

            # Decode阶段: 新token已写入cache, 现在对刚滑出窗口的那个旧KV做原地伪量化
            # logger.info_once(f"key.shape : {key.shape}, query.shape: {query.shape}")
            # if is_decode:
            #     self._quantize_evicted_kv_slots_eager(layer, attn_metadata)

                # self._quantize_evicted_kv_slots(layer, attn_metadata, self._seq_lens_buffer)

        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output

        logger.info_once(f"run in here")
        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def _prepare_c4_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c4_scales_prepared"):
            return
        logger.info_once(f"run in _prepare_c4_scales")
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
        # layer._c4_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c4_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        # layer._c4_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        layer._c4_k_aq_scale = layer._c4_k_scale.to(torch.bfloat16).view(bnsd).contiguous()
        # layer._c4_k_aq_offset = layer._c4_k_offset.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c4_v_aq_scale = layer._c4_v_scale.to(torch.bfloat16).view(bnsd).contiguous()
        # layer._c4_v_aq_offset = layer._c4_v_offset.to(torch.bfloat16).view(bnsd).contiguous()

        layer._c4_k_inv_scale_bf16 = (1.0 / layer._c4_k_scale).to(torch.bfloat16)
        # layer._c4_k_offset_bf16 = layer._c4_k_offset.to(torch.bfloat16)
        layer._c4_v_inv_scale_bf16 = (1.0 / layer._c4_v_scale).to(torch.bfloat16)
        # layer._c4_v_offset_bf16 = layer._c4_v_offset.to(torch.bfloat16)

        layer._c4_scales_prepared = True

    def _quantize_kv_to_fp4(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pseudo-quantize K/V: clamp(round(x * inv_scale), -6, 6) / inv_scale."""
        # logger.info_once(f"run in _quantize_kv_to_fp4")
        self._prepare_c4_scales(layer, key.device)
        dtype = key.dtype
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        # logger.info(f"actual_key.shape: {actual_key.shape}, layer._c4_k_inv_scale_bf16: {layer._c4_k_inv_scale_bf16.shape}")
        
        k_qdq = (torch.clamp(
            torch.round(actual_key * layer._c4_k_inv_scale_bf16),
            -6,
            6,
        ) / layer._c4_k_inv_scale_bf16).to(dtype)
        # msek = torch.nn.MSELoss()(k_qdq, actual_key)
        v_qdq = (torch.clamp(
            torch.round(actual_value * layer._c4_v_inv_scale_bf16),
            -6,
            6,
        ) / layer._c4_v_inv_scale_bf16).to(dtype)
        # msev = torch.nn.MSELoss()(v_qdq, actual_value)
        # logger.info(f"msek: {msek}, msev: {msev}")
        return k_qdq, v_qdq


    def _quant_dequant_hif4(self, x:torch.Tensor, quant_type="hifx4", axe: int=-1):
        logger.info_once(f"start  _quant_dequant_hif4")

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

    # 量化
    def _quantize_hif4_kernel(self, x: torch.Tensor, qdim:int):

        x = x.unflatten(qdim, (-1, 8, 2, 4))  # [1024,1024] -> [16, 8, 2, 4, 1024]
        man_bits = 3
        x_unsigned = torch.abs(x)
        sign = torch.sign(x)

        max_lv3 = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]  # 提取最内层的4个元素最大值 torch.Size([16, 8, 2, 1, 1024])
        max_lv2 = torch.max(max_lv3, dim=qdim - 1, keepdim=True)[0]  # 提取中间层8个元素的最大值 torch.Size([16, 8, 1, 1, 1024])
        max_lv1 = torch.max(max_lv2, dim=qdim - 2, keepdim=True)[0]  # 提取最外层整个64元素块的最大值 torch.Size([16, 1, 1, 1, 1024])

        div7 = torch.ones_like(
            max_lv1) / 7.0  # 算法允许后续局部缩放最多缩小4倍（Level 2 和 Level 3 各缩小 2 倍），$7 / 4 = 1.75$。这保证了经过所有缩放后，最大的数值能落在 $[0, 2)$ 的范围内，方便进行尾数量化。
        # div7 orch.Size([16, 1, 1, 1, 1024])
        div7 = div7.to(torch.bfloat16).to(x.dtype)
        scale_factor = max_lv1 * div7  # 基础scale
        # scale_factor = (scale_factor).to(torch.bfloat16).to(x.dtype).clip(min=2 ** (-48), max=49152)

        ## change scale_factor to bf16(rint) 模拟硬件行为
        e_sf = torch.floor(torch.log2(scale_factor))
        mant_sf = scale_factor / 2 ** e_sf * 2 ** 7
        scale_factor = torch.round(mant_sf) / 2 ** 7 * 2 ** e_sf

        # scale_factor to e6m2 (8-bit 浮点数: 1符号位, 6指数位, 2尾数位)
        e_sf = torch.floor(torch.log2(scale_factor))
        scale_factor = torch.round(scale_factor * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)
        #     [[[[0.0625, 0.0625, 0.0625,  ..., 0.0938, 0.0547, 0.0781]]]]
        # 计算局部动态移位
        rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)
        # level2 尾数量化缩放
        scale_lv2 = (max_lv2 * rec_sf)
        scale_lv2 = torch.exp2((scale_lv2.clip(0,
                                            4) / 4).floor())  # max_lv2子块最大值>=4*scale_factor 则scale_lv2 = 2**1 = 2 ,否则为 2**0 = 1。分配1bit大小决定子块是否需要缩小2倍
        #     [[[2., 2., 2.,  ..., 2., 2., 2.]]],
        # level3 缩放
        scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())  # 同上 评估max_lv3子块最大值是否>=2
        #     [[[2., 2., 2.,  ..., 2., 1., 1.]],

        # 尾数量化
        mant = x_unsigned / scale_lv2 / scale_lv3 * rec_sf  # 全局缩放因子  scale_lv2 / scale_lv3 * rec_sf  shape: torch.Size([16, 8, 2, 4, 1024])
        mant = torch.floor(mant * 2 ** (man_bits - 1) + 0.5) / 2 ** (man_bits - 1)  # 将数值映射到指定的尾数精度man_bits（3）上
        # mant[mant >= 2] = 2 - 2 ** (-man_bits + 1)  # 最大值不超过2-0.25=1.75
        upper_bound = 2 - 2 ** (-man_bits + 1)
        # 同样直接传入 Python 标量
        mant = torch.clamp_max(mant, max=upper_bound)
        # x_unsigned 用hif4格式存储， 三阶存储
        # [0.5000, 1.7500, 0.2500,  ..., 1.0000, 0.5000, 1.7500],
        # out = torch.ones_like(x) * scale_factor
        # 反量化  符号位 * 尾数 * 三层缩放因子
        # S1E2M1 自带符号位 真实计算中mant带sign
        out = sign * mant * scale_lv2 * scale_lv3 * scale_factor  # torch.Size([16, 8, 2, 4, 1024])

        out = out.flatten(qdim - 3, qdim) # -3, 0
        return out



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
        
        # 用来验证 update param 是否生效，update 中直接赋值 -1 ，这里作为分母直接报错
        # tmp = 1.0 / (_seq_lens_buffer + 1.0)
        # tmp_int = torch.round(tmp).to(torch.int32)
        # logger.info_once(f"tmp: {len(tmp)}, ")
        
        # kv_cache: shape =
        #         [2, num_blocks, block_size, num_kv_heads, head_size]
        block_tables = attn_metadata.block_tables  # (batch_size, max_blocks_per_seq)
        block_size = self.key_cache.shape[1]
        device = self.key_cache.device

        m = ATTENTION_SINK_SIZE             # 例如 4
        n = HIGH_PRECISION_WINDOW_SIZE      # 例如 1024

        active_blocks = block_tables[:num_decodes]
        logger.info_once(f"num_decodes: {num_decodes}, active_blocks.shape: {active_blocks.shape}, block_tables.shape:{block_tables.shape}")
        logger.info_once(f"_seq_lens_buffer: {_seq_lens_buffer.device}, ")
        
        approx_seq_lens = _seq_lens_buffer[:num_decodes]
        
        # 用来验证 update param 是否生效，update 中直接赋值 -1 ，这里作为分母直接报错
        # approx_seq_lens = _seq_lens_buffer[:num_decodes] + tmp_int[:num_decodes]
        
        logger.info_once(f"approx_seq_lens: {len(approx_seq_lens)}")
        
        # 被踢出窗口的逻辑 Token 位置：seq_len - 1 - n
        # evicted_pos = approx_seq_lens - 1 - n # (num_decodes,)
        # valid_evict_mask = (evicted_pos >= 0) & (evicted_pos >= m)  # (num_decodes,)
        
        # 不做高精保护
        evicted_pos = approx_seq_lens# (num_decodes,)
        valid_evict_mask = evicted_pos >= 0  # (num_decodes,)


        # 强制用 clamp 把负数和越界位置收束到 0，防止索引越界，后面通过 mask 屏蔽写回
        safe_evicted_pos = torch.clamp(evicted_pos, min=0)
        target_block_indices = safe_evicted_pos // block_size  # (num_decodes,)
        offset_in_blocks = safe_evicted_pos % block_size       # (num_decodes,)

        # 从 block_table 捞出物理块号 (纯 NPU Tensor 静态索引，极其高效)
        row_indices = torch.arange(num_decodes, device=device)
        phys_blocks = active_blocks[row_indices, target_block_indices]  # (num_decodes,)

        # 计算出在展平后的 KV Cache 中的最终一维物理 Slot 地址
        evicted_slots = phys_blocks * block_size + offset_in_blocks  # (num_decodes,)

        # 如果 phys_blocks 是 -1（即该位置尚未分配），将其映射到安全的 0 位置
        safe_slots_mask = valid_evict_mask & (phys_blocks >= 0)
        final_slots = torch.where(safe_slots_mask, evicted_slots, torch.zeros_like(evicted_slots))

        # 对 num_decodes 个位置进行原地伪量化
        flat_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        flat_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)

        # 仅取出滑出位置的 KV 值，形状为 (num_decodes, num_kv_heads, head_size)
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

        # Scatter 原地回写
        flat_key[final_slots] = k_final
        flat_value[final_slots] = v_final
    
    def _quantize_evicted_kv_slots_eager(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
    ) -> None:
        """纯 Eager 模式：不引入 Attention Sink。
        准确地对刚滑出高精窗口（HIGH_PRECISION_WINDOW_SIZE）的单个旧 KV 槽位进行原地伪量化。
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
            # 刚滑出窗口的逻辑 Token 位置
            evicted_pos = seq_len - 1 - n
            
            # Eager 模式下直接通过 if 过滤，不满足条件的不处理
            if evicted_pos < 0:
                continue

            block_idx = evicted_pos // block_size
            offset_in_block = evicted_pos % block_size
            
            # 安全取出物理块号
            phys_block = block_tables[req_idx, block_idx].item()
            if phys_block >= 0:  # 确保块已被分配
                flat_slot = phys_block * block_size + offset_in_block
                evicted_slots.append(flat_slot)

        # 如果当前 batch 没有任何请求需要释放高精槽位，直接返回
        if not evicted_slots:
            return
        # logger.info_once(f"seq_lens_list: {seq_lens_list}, evicted_slots: {evicted_slots}, evicted_pos:{evicted_pos}, block_tables: {block_tables}, phys_block: {phys_block}")
        # 转换为 Tensor 用于高级索引
        device = self.key_cache.device
        evicted_indices = torch.tensor(evicted_slots, dtype=torch.long, device=device)

        # 展平 View（无 Copy 风险，方便一维索引）
        flat_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
        flat_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)

        # 仅取出需要量化的 KV
        k_evicted = flat_key[evicted_indices]
        v_evicted = flat_value[evicted_indices]

        # 原地量化 & 反量化
        k_quantized = (torch.clamp(
            torch.round(k_evicted * layer._c4_k_inv_scale_bf16), -6, 6
        ) / layer._c4_k_inv_scale_bf16).to(k_evicted.dtype)

        v_quantized = (torch.clamp(
            torch.round(v_evicted * layer._c4_v_inv_scale_bf16), -6, 6
        ) / layer._c4_v_inv_scale_bf16).to(v_evicted.dtype)

        # Scatter 回写
        flat_key[evicted_indices] = k_quantized
        flat_value[evicted_indices] = v_quantized

        # logger.info_once("Eager evict without sink executed successfully.")

    def _quantize_evicted_kv_slots_with_sink(
        self,
        layer: AttentionLayer,
        attn_metadata: AscendMetadata,
    ) -> None:
        """纯 Eager 模式：叠加 Attention Sink 保护。
        前 ATTENTION_SINK_SIZE 个 Token 永远保持高精度，不参与伪量化。
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
        m = ATTENTION_SINK_SIZE             # 例如 4
        n = HIGH_PRECISION_WINDOW_SIZE      # 例如 1024

        evicted_slots = []
        for req_idx in range(num_decodes):
            seq_len = seq_lens_list[req_idx]
            # 刚滑出高精窗口的逻辑 Token 位置
            evicted_pos = seq_len - 1 - n
            
            # 核心变更：如果还没有滑出高精窗，或者该位置属于前 m 个 Sink Token，则跳过不量化
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

        # 转换为 Tensor 进行回写
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

        logger.info_once(f"Eager evict with sink (size={m}) executed successfully.")
    # def _quantize_evicted_kv_slots(
    #     self,
    #     layer: AttentionLayer,
    #     attn_metadata: AscendMetadata,
    # ) -> None:
    #     """Pseudo-quantize the KV cache slots that just fell out of the
    #     high-precision window during decode.

    #     For each decode request whose seq_len > HIGH_PRECISION_WINDOW_SIZE,
    #     the token at logical position (seq_len - 1 - HIGH_PRECISION_WINDOW_SIZE)
    #     has just been pushed out of the window and must be quantized in-place.

    #     In ChunkedPrefill mode, only the first `num_decodes` requests are decode
    #     requests (batch is reordered: decodes first, prefills after).  Prefill
    #     requests are skipped because their KV entries are already quantized at
    #     write time.

    #     The block_tables are used to locate the physical KV cache slot for each
    #     evicted position.
    #     """
    #     if self.key_cache is None or self.value_cache is None:
    #         # logger.info_once(f"self.key_cache is None or self.value_cache is None")
    #         return

    #     seq_lens_list = attn_metadata.seq_lens_list
    #     # logger.info_once(f"len(seq_lens_list): {len(seq_lens_list)}")
    #     if seq_lens_list is None or len(seq_lens_list) == 0:
    #         return

    #     block_tables = attn_metadata.block_tables  # (batch_size, max_blocks_per_seq)
    #     block_size = self.key_cache.shape[1]

    #     # Only process decode requests (first num_decodes entries in the batch)
    #     num_decodes = attn_metadata.num_decodes
    #     # logger.info_once(f"num_decodes: {num_decodes}")
    #     if num_decodes <= 0:
    #         return

    #     # Collect the flat slot indices for all evicted positions
    #     evicted_slots = []
    #     for req_idx in range(num_decodes):
    #         seq_len = seq_lens_list[req_idx]
    #         # After writing the new token, seq_len is the total length including it.
    #         # The evicted position is (seq_len - 1 - HIGH_PRECISION_WINDOW_SIZE).
    #         evicted_pos = seq_len - 1 - HIGH_PRECISION_WINDOW_SIZE
    #         if evicted_pos < 0:
    #             # Window not yet full, nothing to evict
    #             continue
    #         # Map logical position to physical slot via block_table
    #         block_idx = evicted_pos // block_size
    #         offset_in_block = evicted_pos % block_size
    #         if block_tables is not None:
    #             phys_block = block_tables[req_idx, block_idx].item()
    #         else:
    #             # Should not happen for decode, but guard
    #             continue
    #         flat_slot = phys_block * block_size + offset_in_block
    #         # logger.info_once(f"phys_block: {phys_block}, block_size: { block_size}, offset_in_block: {offset_in_block}, flat_slot: {flat_slot}")
    #         evicted_slots.append(flat_slot)

    #     if len(evicted_slots) == 0:
    #         return
    #     # logger.info_once(f"len(evicted_slots): {len(evicted_slots)}， evicted_slots： {evicted_slots}")

    #     # Pseudo-quantize the evicted KV cache entries in-place
    #     evicted_indices = torch.tensor(evicted_slots, dtype=torch.long, device=self.key_cache.device)
    #     # key_cache shape: (num_blocks, block_size, num_kv_heads, head_size)
    #     # evicted_indices address the first two dims as flat index
    #     k_evicted = self.key_cache.view(-1, self.num_kv_heads, self.head_size)[evicted_indices]
    #     v_evicted = self.value_cache.view(-1, self.num_kv_heads, self.head_size)[evicted_indices]
    #     # logger.info_once(f"self.key_cache.shape: {self.key_cache.shape}， self.value_cache.shape:  {self.value_cache.shape}, evicted_indices: {evicted_indices}, self.num_kv_heads: {self.num_kv_heads}, self.head_size: {self.head_size}")

    #     k_quantized = (torch.clamp(
    #         torch.round(k_evicted * layer._c4_k_inv_scale_bf16),
    #         -6,
    #         6,
    #     ) / layer._c4_k_inv_scale_bf16).to(k_evicted.dtype)
    #     v_quantized = (torch.clamp(
    #         torch.round(v_evicted * layer._c4_v_inv_scale_bf16),
    #         -6,
    #         6,
    #     ) / layer._c4_v_inv_scale_bf16).to(v_evicted.dtype)

    #     # Write back in-place
    #     # [num_blocks, block_size, num_kv_heads, head_size]
    #     flat_key = self.key_cache.view(-1, self.num_kv_heads, self.head_size)
    #     flat_value = self.value_cache.view(-1, self.num_kv_heads, self.head_size)
    #     flat_key[evicted_indices] = k_quantized
    #     flat_value[evicted_indices] = v_quantized
    #     logger.info_once(f"run in _quantize_evicted_kv_slots")
        
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
