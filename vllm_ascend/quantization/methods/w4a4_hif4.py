#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""W4A4 HiF4 quantization scheme (bring-up / accuracy path).

HiF4 (microscaling S1E2M1 with a three-level e6m2 block scale) has no native
CANN operator that consumes a HiF4 weight together with a HiF4 activation in a
single 4x4 quant matmul. This module implements an accuracy-first bring-up path:

  * Weights are dequantized to BF16 once at load time
    (``process_weights_after_loading``) and stored as ordinary BF16 tensors.
  * Activations are pseudo-quantized (quant -> dequant) to HiF4 every forward
    (``apply``), reusing the same three-level math as the KV-cache HiF4 kernel
    in ``attention/attention_v1.py::_quantize_hif4_kernel`` (which is bit-exact
    with msModelSlim's ``ir/api/impl/hifx_quantization.py``).
  * The linear itself is a plain BF16 matmul (``torch.mm`` /
    ``npu_grouped_matmul`` via the unquant MoE path).

The result runs end-to-end and reproduces HiF4 quantization error on both sides;
throughput is BF16-class. It is the stepping stone toward a future fused HiF4
operator, not the final perf path.

Disk format (msModelSlim ``ascendv1_saver``, quant_type string ``W4A4_HIFP4``,
group_size=64) -- see ``hif4_disk_format`` memory and msModelSlim
``core/.../save/utils/pack.py``:

  * ``weight``      : uint8 ``[out, in//2]``, two 4-bit elements per byte
                      ``byte = low_nibble | (high_nibble << 4)``.
                      nibble = ``bit3(sign) | bit0-2(E1M2 index)`` into
                      ``{0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75}``.
  * ``weight_scale``: float32 ``[out, in//64, 1]`` but stored as a **packed
                      int32** reinterpreted as float32:
                      ``int32 = (e6m2_byte << 24) | (lv2_byte << 16) | lv3_u16``
                      - e6m2_byte : lv1 base scale, exp=(b>>2)-31, mant=b&3,
                                    scale_factor = 2**exp * (1 + mant/4).
                      - lv2_byte  : 8 one-bit flags (big-endian), 1.0 or 2.0,
                                    one flag per 8-element sub-block.
                      - lv3_u16   : 16 one-bit flags, 1.0 or 2.0, one per
                                    4-element sub-block. (64 = 8*2*4.)
"""

from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.utils import maybe_trans_nz

from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme

# ---------------------------------------------------------------------------
# HiF4 format constants (mirror attention/attention_v1.py::_quantize_hif4_kernel
# and msModelSlim ir/api/impl/hifx_quantization.py)
# ---------------------------------------------------------------------------
_HIF4_BLOCK = 64          # scalars per HiF4 group / disk group_size
_HIF4_LV2 = 8             # lv2 sub-blocks of size 8 within a 64-block
_HIF4_LV3 = 2             # lv3 sub-blocks of size 4 within an 8-block   (8*2*4=64)
_HIF4_INNER = 4
_HIF4_MAN_BITS = 3        # mantissa rounding bits (-> S1E2M1, clamp to 1.75)

# E1M2 magnitude LUT used on disk (NOT FP4-E2M1's {0,0.5,1,1.5,2,3,4,6}).
_E1M2_LUT = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75)
# Big-endian bit weights used by msModelSlim's pack_hif4_scale_e1_to_uint8.
_BIT8 = (128, 64, 32, 16, 8, 4, 2, 1)
_BIT16 = (32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1)


# ---------------------------------------------------------------------------
# Activation: HiF4 pseudo-quantization (quant -> dequant, stays high precision)
# ---------------------------------------------------------------------------
def _e6m2_round(scale_factor: torch.Tensor) -> torch.Tensor:
    """Round a positive scale to the e6m2 grid (1 sign, 6 exp, 2 mantissa bits).

    Mirrors the HW scale-quantization step of the KV-cache HiF4 kernel and of
    msModelSlim's ``calculate_hifx_qparam``.
    """
    # Clamp to msModelSlim's range [2**-48, 49152] *before* log2: using bf16 tiny
    # (≈2**-126) here makes ``scale / 2**e_sf * 2**7`` overflow to inf in bf16,
    # which then poisons the whole 64-block with NaN. 2**-48 is small enough that
    # the intermediate stays finite. Do the math in float32 for safety.
    sf = scale_factor.float()
    sf = torch.clamp(sf, min=2 ** (-48), max=49152.0)
    e_sf = torch.floor(torch.log2(sf))
    mant_sf = sf / 2**e_sf * 2**7
    sf = torch.round(mant_sf) / 2**7 * 2**e_sf
    e_sf = torch.floor(torch.log2(sf))
    sf = torch.round(sf * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)
    return sf.to(scale_factor.dtype)


def hif4_pseudo_quant(x: torch.Tensor, qdim: int = -1) -> torch.Tensor:
    """Pseudo-quantize (quant -> dequant) ``x`` to HiF4 along ``qdim``.

    Bit-exact with ``attention/attention_v1.py::_quantize_hif4_kernel`` and with
    msModelSlim's ``hif4_per_block_quantize`` + ``hif4_per_block_dequantize``.
    Input and output share the same dtype; the output carries HiF4 quantization
    noise but stays in high precision, ready for a plain BF16 matmul.
    """
    dtype = x.dtype
    ndim = x.dim()
    qdim_abs = qdim if qdim >= 0 else ndim + qdim  # unflatten inserts 4 dims here

    # Factor the quantized dim into (outer, lv2=8, lv3=2, inner=4) = a 64-block.
    x_blk = x.unflatten(qdim, (-1, _HIF4_LV2, _HIF4_LV3, _HIF4_INNER))

    x_unsigned = torch.abs(x_blk)
    sign = torch.sign(x_blk)

    # Three-level max along the block dims: inner 4 -> lv3(2) -> lv2(8) -> 64.
    d_inner, d_lv3, d_lv2 = qdim_abs + 3, qdim_abs + 2, qdim_abs + 1
    max_lv3 = torch.max(x_unsigned, dim=d_inner, keepdim=True)[0]   # over inner 4
    max_lv2 = torch.max(max_lv3, dim=d_lv3, keepdim=True)[0]        # over lv3 (2)
    max_lv1 = torch.max(max_lv2, dim=d_lv2, keepdim=True)[0]        # over lv2 (8)

    # div7 = 1/7: lets lv2/lv3 each shrink by up to 2x so the largest value lands
    # in [0, 2) after all scaling, ready for the mantissa grid.
    div7 = (torch.ones_like(max_lv1) / 7.0).to(torch.bfloat16).to(dtype)
    scale_factor = _e6m2_round(max_lv1 * div7)                      # base (lv1) scale
    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(dtype)

    # lv2 sub-block: 2x if its max >= 4 * scale_factor else 1x
    scale_lv2 = max_lv2 * rec_sf
    scale_lv2 = torch.exp2((scale_lv2.clip(0, 4) / 4).floor())
    # lv3 sub-block: 2x if its max >= 2 * (scale_factor * scale_lv2) else 1x
    scale_lv3 = (max_lv3 * rec_sf / scale_lv2).clip(0, 2)
    scale_lv3 = torch.exp2((scale_lv3 / 2).floor())

    # 3-bit mantissa round on the normalized value -> S1E2M1 (values in [0, 1.75])
    man_bits = _HIF4_MAN_BITS
    mant = x_unsigned / scale_lv2 / scale_lv3 * rec_sf
    mant = torch.floor(mant * 2 ** (man_bits - 1) + 0.5) / 2 ** (man_bits - 1)
    upper_bound = 2 - 2 ** (-man_bits + 1)
    mant = torch.clamp_max(mant, max=upper_bound)

    # dequant: sign * mant * three-level scale
    out = sign * mant * scale_lv2 * scale_lv3 * scale_factor
    return out.flatten(qdim_abs, qdim_abs + 3)


# ---------------------------------------------------------------------------
# Weight: dequantize the on-disk HiF4 format back to BF16
# ---------------------------------------------------------------------------
def _decode_hif4_scale(scale_packed: torch.Tensor):
    """Decode the packed-int32 (stored as float32) three-level scale.

    Args:
        scale_packed: float32 tensor of shape ``(..., n_blocks)`` (2-D for a single
                      linear/expert, 3-D ``[E, out, n_blocks]`` for grouped MoE)
                      whose bits are ``(e6m2_byte << 24) | (lv2_byte << 16) | lv3_u16``.
                      A trailing size-1 dim (msModelSlim's on-disk ``[out, n_blocks, 1]``
                      layout) is squeezed automatically.

    Returns:
        scale_factor: ``(..., n_blocks)`` float, the lv1 e6m2 base scale.
        lv2:          ``(..., n_blocks, 8)`` float in {1.0, 2.0}, one per 8-elem.
        lv3:          ``(..., n_blocks, 16)`` float in {1.0, 2.0}, one per 4-elem.
    """
    s32 = scale_packed.view(torch.int32)
    # Drop a trailing size-1 dim if present (on-disk layout is [out, n_blocks, 1]).
    if s32.dim() >= 2 and s32.shape[-1] == 1:
        s32 = s32.squeeze(-1)

    f_bits = ((s32 >> 24) & 0xFF).long()         # e6m2 byte
    l2_bits = ((s32 >> 16) & 0xFF).long()        # 8 lv2 flags packed
    l3_bits = (s32 & 0xFFFF).long()              # 16 lv3 flags packed

    # e6m2 decode: exp=(b>>2)-31, mant=b&3 -> 2^exp * (1 + mant/4)
    exp = ((f_bits >> 2) - 31).float()
    mant = (f_bits & 3).float()
    scale_factor = torch.pow(2.0, exp) * (1.0 + mant / 4.0)     # (..., n_blocks)

    w8 = torch.tensor(_BIT8, dtype=torch.int64, device=s32.device)
    w16 = torch.tensor(_BIT16, dtype=torch.int64, device=s32.device)
    lv2 = ((l2_bits.unsqueeze(-1) & w8) != 0).float() + 1.0     # (..., n_blocks, 8)
    lv3 = ((l3_bits.unsqueeze(-1) & w16) != 0).float() + 1.0    # (..., n_blocks, 16)
    return scale_factor, lv2, lv3


def hif4_dequantize_weight(
    weight_packed: torch.Tensor,
    scale_packed: torch.Tensor,
    input_size: int,
    output_size: int,
) -> torch.Tensor:
    """Dequantize a packed HiF4 weight (uint8 + packed-int32 scale) to BF16.

    Args:
        weight_packed: uint8 ``[output_size, input_size // 2]`` (two nibbles/byte).
        scale_packed:  float32 ``[output_size, input_size // 64, 1]`` (packed int32).
        input_size:    full reduction dim (must be a multiple of 64).
        output_size:   output dim.

    Returns:
        BF16 weight of shape ``(input_size, output_size)`` (transposed to match
        the ``x @ W`` convention used by ``apply``).
    """
    assert input_size % _HIF4_BLOCK == 0, "HiF4 requires input_size % 64 == 0"

    # 1) Unpack two nibbles per byte -> signed E1M2 quantized values (out, input_size).
    e1m2 = torch.tensor(_E1M2_LUT, dtype=torch.float32, device=weight_packed.device)
    low = (weight_packed & 0xF).long()
    high = (weight_packed >> 4).long()
    mag = torch.stack([e1m2[low & 0x7], e1m2[high & 0x7]], dim=-1).reshape(output_size, input_size)
    sign = torch.stack(
        [
            torch.where(low & 0x8 != 0, -1.0, 1.0),
            torch.where(high & 0x8 != 0, -1.0, 1.0),
        ],
        dim=-1,
    ).reshape(output_size, input_size)
    qval = mag * sign                                          # (out, in) in [-1.75, 1.75]

    # 2) Decode the three-level scale and broadcast onto the 64-block structure.
    n_blocks = input_size // _HIF4_BLOCK
    scale_factor, lv2, lv3 = _decode_hif4_scale(scale_packed)  # each (out, n_blocks, ...)

    qblk = qval.reshape(output_size, n_blocks, _HIF4_BLOCK)    # (out, n_blocks, 64)
    sf = scale_factor.unsqueeze(-1).expand(output_size, n_blocks, _HIF4_BLOCK)
    lv2e = lv2.repeat_interleave(_HIF4_LV3 * _HIF4_INNER, dim=-1)   # 8 flags -> each x8
    lv3e = lv3.repeat_interleave(_HIF4_INNER, dim=-1)               # 16 flags -> each x4
    deq = qblk * sf * lv2e * lv3e                             # (out, n_blocks, 64)

    deq = deq.reshape(output_size, input_size).to(torch.bfloat16)
    return deq.transpose(0, 1).contiguous()                   # (in, out) for x @ W


@register_scheme("W4A4_HIFP4", "linear")
class AscendW4A4HiF4DynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A4 HiF4 (BF16-dequant weight + HiF4 pseudo-quant activation).

    Loaded parameters (msModelSlim ``W4A4_HIFP4`` checkpoint):

    * ``weight``:      packed FP4 (E1M2) weight, uint8 ``(output_size, input_size // 2)``
                       (two nibbles per byte; the packed layout is kept on both param and disk).
    * ``weight_scale``: three-level scale packed into float32 ``(output_size, input_size // 64)``
                       (2-D, matches the on-disk layout; a trailing size-1 dim is also accepted
                       by ``_decode_hif4_scale`` for the older 3-D on-disk layout).

    After :meth:`process_weights_after_loading`, ``layer.weight`` holds the BF16
    dequantized weight of shape ``(input_size, output_size)`` and the packed
    storage is released.
    """

    model_dtype = None

    def __init__(self):
        vllm_config = get_current_vllm_config()
        # group_size for HiF4 is fixed at 64 on disk; the config value is kept
        # only for API compatibility.
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", _HIF4_BLOCK)

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        # Two E1M2 elements packed per byte -> input_size // 2 along the reduction dim.
        # Both the parameter and the on-disk tensor keep the packed layout, so no
        # _packed_dim/_packed_factor declaration is needed: the weight_loader narrows
        # the (already-packed) reduction dim by the TP partition directly. This
        # matches the W4A4 MXFP4 linear scheme.
        return {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        # One packed-int32 (stored as float32) three-level scale per 64-input block.
        # Shape is 2-D [out, in//64], matching the on-disk layout exactly (the dense
        # linear weight_loader in vllm linear.py does a strict param.size()==loaded.size()
        # assert). _decode_hif4_scale also accepts a trailing size-1 dim for old layouts.
        return {
            "weight_scale": torch.empty(output_size, input_size // _HIF4_BLOCK, dtype=torch.float32)
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])

        # Activation: HiF4 pseudo-quant (quant -> dequant), stays BF16.
        # Set env HIF4_NO_ACT_QUANT=1 to bypass activation quant (diagnostic only:
        # isolates whether a quality issue comes from the activation pseudo-quant
        # or from the weight dequant).
        import os as _os
        if _os.environ.get("HIF4_NO_ACT_QUANT"):
            x_q = x
        else:
            x_q = hif4_pseudo_quant(x, qdim=-1)

        # Weight: already dequantized to BF16 at load time. Plain matmul.
        output = torch.mm(x_q, layer.weight)
        if bias is not None:
            output = output + bias.to(output.dtype)

        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)
        return output

    def process_weights_after_loading(self, layer):
        """Dequantize the packed HiF4 weight to BF16 once and store it.

        After this call ``layer.weight`` is a BF16 tensor of shape
        ``(input_size, output_size)`` and the original packed storage is replaced.
        ``weight_scale`` is dropped once dequantized.
        """
        weight_packed = layer.weight.data            # (out, in//2) uint8
        scale_packed = layer.weight_scale.data       # (out, in//64) float32 (packed int32)

        output_size, half_k = weight_packed.shape
        input_size = half_k * 2

        w_bf16 = hif4_dequantize_weight(weight_packed, scale_packed, input_size, output_size)
        layer.weight = torch.nn.Parameter(w_bf16, requires_grad=False)
        if hasattr(layer, "weight_scale"):
            del layer.weight_scale


@register_scheme("W4A4_HIFP4", "moe")
class AscendW4A4HiF4DynamicFusedMoEMethod(AscendMoEScheme):
    """MoE method for Ascend W4A4 HiF4 (BF16-dequant experts + HiF4 pseudo-quant activation).

    Expert weights are dequantized to BF16 at load time and the MoE forward is
    routed through the unquantized ``npu_grouped_matmul`` path
    (``quant_type=QuantType.NONE`` -> ``unquant_apply_mlp``), with activations
    HiF4 pseudo-quantized on the way in. Trades MXFP4 4x4 throughput for a
    correct, runnable HiF4 MoE.
    """

    model_dtype = None
    quant_type: QuantType = QuantType.NONE  # -> unquant_apply_mlp (BF16 grouped matmul)

    def __init__(self):
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", _HIF4_BLOCK)
        ascend_config = get_ascend_config()
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )

    @staticmethod
    def get_weight(
        num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        return {
            "w13_weight": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, hidden_sizes // 2, dtype=torch.uint8
            ),
            "w2_weight": torch.empty(
                num_experts, hidden_sizes, intermediate_size_per_partition // 2, dtype=torch.uint8
            ),
        }

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        # 3-D grouped scales [E, out, in//64] (no trailing size-1 dim). On disk each
        # expert's scale is the 2-D [out, in//64] slice; a 3-D per-expert tensor would
        # be misclassified by vLLM's weight_loader full-load heuristic (ndim==3).
        return {
            "w13_weight_scale": torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_sizes // _HIF4_BLOCK,
                dtype=torch.float32,
            ),
            "w2_weight_scale": torch.empty(
                num_experts,
                hidden_sizes,
                intermediate_size_per_partition // _HIF4_BLOCK,
                dtype=torch.float32,
            ),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = True,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        assert router_logits.shape[1] == num_logical_experts, "Number of global experts mismatch (excluding redundancy)"
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            num_experts=num_logical_experts,
        )

        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(x.dtype)

        # Activation: HiF4 pseudo-quant on the way into the experts.
        x_q = hif4_pseudo_quant(x, qdim=-1)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x_q,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=layer.w13_weight,        # BF16 after process_weights_after_loading
                w2=layer.w2_weight,         # BF16 after process_weights_after_loading
                quant_type=self.quant_type,  # QuantType.NONE -> unquant_apply_mlp
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
            )
        )

    def process_weights_after_loading(self, layer):
        """Dequantize both MoE expert matrices (w13, w2) from HiF4 to BF16.

        ``hif4_dequantize_weight`` returns ``(in, out)`` per expert, so after stacking
        the layout is ``[E, K, N]`` (K=input/reduction dim, N=output). That is exactly
        the layout ``npu_grouped_matmul`` expects: ``weight.dim == x.dim`` on the K
        axis (need_trans=False path in ``unquant_apply_mlp``). Only the NZ memory-format
        cast is needed -- the same final step ``AscendUnquantizedFusedMoEMethod`` does
        after its own ``transpose(1,2)`` (its on-disk layout starts as ``[E, N, K]``,
        so it transposes; ours starts as ``[E, K, N]``, so it must not).
        """

        def _dequant_expert(weight_packed: torch.Tensor, scale_packed: torch.Tensor) -> torch.Tensor:
            num_experts, output_size, half_k = weight_packed.shape
            input_size = half_k * 2
            out = torch.empty(num_experts, input_size, output_size, dtype=torch.bfloat16, device=weight_packed.device)
            for e in range(num_experts):
                out[e] = hif4_dequantize_weight(
                    weight_packed[e], scale_packed[e], input_size, output_size
                )
            return out.contiguous()   # [E, K, N]

        # No transpose: dequant already yields [E, K, N] = what npu_grouped_matmul wants
        # (weight K-axis must match x's K-axis). Only cast to NZ.
        w13 = _dequant_expert(layer.w13_weight.data, layer.w13_weight_scale.data)
        w2 = _dequant_expert(layer.w2_weight.data, layer.w2_weight_scale.data)
        layer.w13_weight = torch.nn.Parameter(maybe_trans_nz(w13), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(maybe_trans_nz(w2), requires_grad=False)
        for name in ("w13_weight_scale", "w2_weight_scale"):
            if hasattr(layer, name):
                delattr(layer, name)
