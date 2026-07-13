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

from collections.abc import Callable
from typing import Any

import torch
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.device.mxfp_compat import (
    FLOAT8_E8M0FNU_DTYPE,
    ensure_mxfp4_linear_available,
    ensure_mxfp4_moe_available,
)
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod


from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme


@register_scheme("W4A4_HIFP4", "linear")
class AscendW4A4HiFPDynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A4_HIFP4 (Microscaling FP4) quantization.
    """

    model_dtype = None

    def __init__(self):
        ensure_mxfp4_linear_available("W4A4_HIFP4 linear quantization")
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 64)

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}

        params_dict["weight_scale"] = torch.empty(output_size, input_size // self.group_size, dtype=torch.uint8)
        return params_dict

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # reshape x for Qwen VL models
        original_shape = x.shape
        output_dtype = x.dtype
        qdim = -1
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        # quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(
        #     x, dst_type=torch_npu.float4_e2m1fn_x2, round_mode="round"
        # )
        
        # 激活、权重scale的伪量化
        quantized_x = quant_dequant_hif4(x)
        print(f"--------------layer quanzition quantized_x:{quantized_x}")
        # 解析weight_scale
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)

        # 注意w的转置
        output = torch.ops.vllm.unquantized_gemm(quantized_x, layer.weight, bias)
        # pertoken_scale = dynamic_scale
        # output = torch_npu.npu_quant_matmul(
        #     quantized_x,
        #     layer.weight,
        #     layer.weight_scale,
        #     scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        #     pertoken_scale=pertoken_scale,
        #     pertoken_scale_dtype=FLOAT8_E8M0FNU_DTYPE,
        #     bias=bias,
        #     output_dtype=output_dtype,
        #     x1_dtype=torch_npu.float4_e2m1fn_x2,
        #     x2_dtype=torch_npu.float4_e2m1fn_x2,
        #     group_sizes=[1, 1, self.group_size],
        # )
        # reshape output for Qwen VL models
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)

        return output.to(output_dtype)

    def process_weights_after_loading(self, layer):
        """Process weights after loading for HiFP4 inference.

        This method transforms weights for NPU HiFP4 computation:
        - weight: (output_size, input_size) -> (input_size, output_size)
        - weight_scale: (n_dim, k_dim) -> (k_dim//2, n_dim, 2)
        """

        # n_dim, k_dim = layer.weight_scale.data.shape
        # layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
        # layer.weight.data = layer.weight.data.transpose(0, 1)
        # layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)
        print(f"------------before process_weights_after_loading layer.weight: {layer.weight.shape}")
        scale_factor, scale_lv2, scale_lv3 = unpack_hif4_scale_from_fp32(layer.weight_scale)
        quantized_w = layer.weight * scale_lv2 * scale_lv3 * scale_factor
        print(f"------------after process_weights_after_loading layer.weight: {layer.weight.shape}  quantized_w:{quantized_w.shape}")
        quantized_w = dqw.flatten(qdim - 3, qdim)
        layer.weight.data = quantized_w

@register_scheme("W4A4_HIFP4", "moe")
class AscendW4A4HiFPDynamicFusedMoEMethod(AscendUnquantizedFusedMoEMethod):
    """FusedMoe method for Ascend W4A4_HIFP4."""

    model_dtype = None
    quant_type = None

    def __init__(self):
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 64)
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb

    @staticmethod
    def get_weight(
        num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}

        # intermediate_size_per_partition 768 /  hidden_sizes 2048
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // 2, dtype=torch.uint8
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // 2, dtype=torch.uint8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        # intermediate_size_per_partition:768, hidden_sizes:2048 num_experts:128
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.uint8
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.uint8
        )
        return param_dict

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
        quantized_x = quant_dequant_hif4(x)

        return super().apply()  # TODO 参数对应


    def process_weights_after_loading(self, layer):
        # msit load weight
        # w13_weight_scale  w2_weight_scale 反量化 ——> layer.w13_weight layer.w2_weight

        # w13: up-gate w2: down
        print(f"-------------wsh moe process_weights_after_loading layer.w13_weight_scale:{layer.w13_weight_scale.shape}")
        w13_scale_factor, w13_scale_lv2, w13_scale_lv3 = unpack_hif4_scale_from_fp32(layer.w13_weight_scale)
        layer.w13_weight.data =  layer.w13_weight * w13_scale_lv2 * w13_scale_lv3 * w13_scale_factor
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        
        w2_scale_factor, w2_scale_lv2, w2_scale_lv3 = unpack_hif4_scale_from_fp32(layer.w13_weight_scale)
        layer.w13_weight.data =  layer.w2_weight * w2_scale_lv2 * w2_scale_lv3 * w2_scale_factor
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)

        # [ 专家， 输入， 输出]
        # 原 w4a4_mxfp4
        # g_num, n_size, k_size = layer.w13_weight_scale.shape
        # layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
        # g_num, n_size, k_size = layer.w2_weight_scale.shape
        # layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g_num, n_size, k_size // 2, 2)
        # layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        # layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        # layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2)
        # layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2)


def quant_dequant_hif4(x: torch.Tensor, quant_type: str = "hifx4", axe: int = -1):
    logger.info_once("[layer quant] HiF4 strategy active (_quant_dequant_hif4)")
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

    qdq_out = self.quantize_hif4_kernel(x, -1)
    qdq_out *= attention_mask
    return qdq_out.to(dtype_ori)

def quantize_hif4_kernel(x: torch.Tensor, qdim: int):
    x = x.unflatten(qdim, (-1, 8, 2, 4))  # head_size -> [16, 8, 2, 4] for 1024
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

def unpack_hif4_scale_from_fp32(packed_fp32: torch.Tensor, bias=31):
    """
    从打包的 float32 恢复 scale_factor, scale_lv2, scale_lv3
    """
    device = packed_fp32.device
    
    # 1. 视作 int32 进行位操作
    packed_int32 = packed_fp32.view(torch.int32)
    
    # 2. 提取各部分的位域
    f_u8 = (packed_int32 >> 24) & 0xFF
    l2_u8 = (packed_int32 >> 16) & 0xFF
    l3_u16 = packed_int32 & 0xFFFF
    
    # ----------------------------------------------------
    # 3. 恢复 scale_factor (反量化 E6M2)
    # ----------------------------------------------------
    exp_bits = (f_u8 >> 2) & 0x3F     # 高 6 位是指数
    mantissa_bits = f_u8 & 0x03       # 低 2 位是尾数
    
    e_sf = exp_bits.to(torch.float32) - bias
    mantissa = mantissa_bits.to(torch.float32) / 4.0
    scale_factor = (1.0 + mantissa) * torch.pow(2.0, e_sf)
    
    # ----------------------------------------------------
    # 辅助函数：将整型恢复为 8 位的 1.0 或 2.0 (大端序)
    # ----------------------------------------------------
    def unpack_bytes_to_bits(tensor, num_bits):
        # 创建大端序的位移矩阵 [num_bits-1, ..., 0]
        shifts = torch.arange(num_bits - 1, -1, -1, device=device, dtype=tensor.dtype)
        # 通过右移和与运算提取每一位
        bits = (tensor.unsqueeze(-1) >> shifts) & 1
        # 将 0/1 映射回 1.0/2.0
        return torch.where(bits == 1, 2.0, 1.0)

    # ----------------------------------------------------
    # 4. 恢复 scale_lv2 和 scale_lv3
    # ----------------------------------------------------
    # l2_u8 每个元素代表 8 个标志位
    scale_lv2 = unpack_bytes_to_bits(l2_u8, num_bits=8)
    # 展开最后一维以符合原始多维张量结构
    scale_lv2 = scale_lv2.reshape(list(packed_fp32.shape) + [-1])
    
    # l3_u16 每个元素包含 16 个标志位
    scale_lv3 = unpack_bytes_to_bits(l3_u16, num_bits=16)
    scale_lv3 = scale_lv3.reshape(list(packed_fp32.shape) + [-1])
    
    return scale_factor, scale_lv2, scale_lv3
