import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.logger import logger

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

from .base import AscendAttentionScheme
from .registry import register_scheme

from vllm.logger import logger

def _fa_quant_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor):
    """Weight loader for MLA-based C8 (FAKQuant) models."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        shard_size = loaded_weight.shape[0] // tp_size
        loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank, shard_size)
        assert param.size() == loaded_weight.size(), (
            "[vllm-ascend/FAKQuant] Attempted to load weight "
            f"({loaded_weight.size()}) into parameter ({param.size()}) "
            f"when TP size is {tp_size} and TP rank is {tp_rank}."
        )

        param.data.copy_(loaded_weight)


@register_scheme("FAKQuant", "attention")
class AscendFAQuantAttentionMethod:
    def __init__(self):
        vllm_config = get_current_vllm_config()
        config = vllm_config.model_config.hf_config
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)

    def create_weights(self, layer: torch.nn.Module) -> None:
        extra_module_names = ["fa_q", "fa_k", "fa_v"]
        for name in extra_module_names:
            setattr(layer, name, torch.nn.Module())
        params_dict = {}
        dtype = torch.get_default_dtype()
        params_dict["fa_q.scale"] = torch.empty((layer.num_heads, 1), dtype=dtype)
        params_dict["fa_k.scale"] = torch.empty((layer.num_kv_heads, 1), dtype=dtype)
        params_dict["fa_v.scale"] = torch.empty((layer.num_kv_heads, 1), dtype=dtype)
        params_dict["fa_q.offset"] = torch.empty((layer.num_heads, 1), dtype=torch.int8)
        params_dict["fa_k.offset"] = torch.empty((layer.num_kv_heads, 1), dtype=torch.int8)
        params_dict["fa_v.offset"] = torch.empty((layer.num_kv_heads, 1), dtype=torch.int8)

        for name, weight in params_dict.items():
            module_name, weight_name = name.rsplit(".", 1)
            module = getattr(layer, module_name)
            weight_param = torch.nn.Parameter(weight, requires_grad=False)
            module.register_parameter(weight_name, weight_param)
            # When loading weights, segment them according to TP
            weight_param.weight_loader = _fa_quant_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        fa_k_scale = torch.squeeze(layer.fa_k.scale).unsqueeze(0)
        layer.fak_descale_float = torch.nn.Parameter(fa_k_scale.to(torch.float), requires_grad=False)
        layer.fak_descale = torch.nn.Parameter(fa_k_scale, requires_grad=False)
        if get_ascend_device_type() == AscendDeviceType.A5:
            layer.fak_descale_reciprocal = 1.0 / torch.nn.Parameter(fa_k_scale.to(torch.float), requires_grad=False)
        else:
            layer.fak_descale_reciprocal = 1.0 / torch.nn.Parameter(fa_k_scale, requires_grad=False)
        fa_k_offset = torch.squeeze(layer.fa_k.offset).unsqueeze(0)
        layer.fak_offset = torch.nn.Parameter(fa_k_offset.to(layer.fak_descale.dtype), requires_grad=False)

        repeated_quant_kscale = fa_k_scale.repeat(self.kv_lora_rank)
        layer.quant_kscale = repeated_quant_kscale.view(1, self.kv_lora_rank)
        layer.quant_kscale = 1.0 / torch.nn.Parameter(layer.quant_kscale.to(torch.float), requires_grad=False)


@register_scheme("INT8_DYNAMIC", "attention")
class AscendSFAQuantAttentionMethod:
    def __init__(self):
        vllm_config = get_current_vllm_config()
        config = vllm_config.model_config.hf_config
        self.index_head_dim = config.index_head_dim

    def create_weights(self, layer: torch.nn.Module) -> None:
        extra_module_names = ["indexer"]
        for name in extra_module_names:
            setattr(layer, name, torch.nn.Module())
        params_dict = {}
        params_dict["indexer.q_rot"] = torch.empty((self.index_head_dim, self.index_head_dim), dtype=torch.float32)
        params_dict["indexer.k_rot"] = torch.empty((self.index_head_dim, self.index_head_dim), dtype=torch.float32)
        for name, weight in params_dict.items():
            module_name, weight_name = name.split(".")
            module = getattr(layer, module_name)
            weight_param = torch.nn.Parameter(weight, requires_grad=False)
            module.register_parameter(weight_name, weight_param)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass


def _c8_kv_scale_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Weight loader for dense-attention C8 KV cache scales/offsets."""
    loaded_weight = loaded_weight.squeeze()
    if param.data.shape != loaded_weight.shape:
        param.data = loaded_weight.to(param.dtype).clone()
    else:
        param.data.copy_(loaded_weight)


class AscendC8KVCacheAttentionMethod(AscendAttentionScheme):
    """C8 INT8 KV cache quantization for dense-attention models (e.g. Qwen3)."""

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix
        vllm_config = get_current_vllm_config()
        self.is_kv_producer = False
        if vllm_config.kv_transfer_config is not None:
            self.is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Returns int8 if the P node is not a PD detachment node.
        if not self.is_kv_producer:
            logger.info_once(
                "[vllm-ascend/C8_KV] KV cache producer is disabled; setting kv_cache_torch_dtype to torch.int8."
            )
            layer.kv_cache_torch_dtype = torch.int8
        # Upgrade impl to the C8-specific subclass so the C8 forward path is always used.
        if hasattr(layer, "impl"):
            from vllm_ascend.attention.attention_v1 import AscendC8AttentionBackendImpl

            layer.impl.__class__ = AscendC8AttentionBackendImpl
        dtype = torch.get_default_dtype()
        layer.k_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        layer.k_cache_scale.weight_loader = _c8_kv_scale_weight_loader
        layer.k_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
        layer.k_cache_offset.weight_loader = _c8_kv_scale_weight_loader
        layer.v_cache_scale = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        layer.v_cache_scale.weight_loader = _c8_kv_scale_weight_loader
        layer.v_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
        layer.v_cache_offset.weight_loader = _c8_kv_scale_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.k_cache_scale.data = layer.k_cache_scale.data.flatten()
        layer.k_cache_offset.data = layer.k_cache_offset.data.flatten()
        layer.v_cache_scale.data = layer.v_cache_scale.data.flatten()
        layer.v_cache_offset.data = layer.v_cache_offset.data.flatten()

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "AscendC8KVCacheAttentionMethod.apply should not be called. "
            "C8 KV cache quantization is handled by the attention backend."
        )


class AscendC4KVCacheAttentionMethod(AscendAttentionScheme):
    """C4 FP KV cache quantization for dense-attention models (e.g. Qwen3)."""

    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Override kv_cache_torch_dtype so Attention.get_kv_cache_spec returns int8 automatically.
        
        # layer.kv_cache_torch_dtype = torch.int8
        
        # Upgrade impl to the C8-specific subclass so the C8 forward path is always used.
        
        # if hasattr(layer, "impl"):
        #     from vllm_ascend.attention.attention_v1 import AscendC8AttentionBackendImpl

        #     layer.impl.__class__ = AscendC8AttentionBackendImpl
        logger.info_once(f"AscendC4KVCacheAttentionMethod create_weights")
        layer.k_cache_scale = torch.nn.Parameter(torch.ones(512, dtype=torch.float32), requires_grad=False)
        layer.k_cache_scale.weight_loader = _c8_kv_scale_weight_loader
        # layer.k_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        # layer.k_cache_offset.weight_loader = _c8_kv_scale_weight_loader
        layer.v_cache_scale = torch.nn.Parameter(torch.ones(512, dtype=torch.float32), requires_grad=False)
        layer.v_cache_scale.weight_loader = _c8_kv_scale_weight_loader
        # layer.v_cache_offset = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        # layer.v_cache_offset.weight_loader = _c8_kv_scale_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.k_cache_scale.data = layer.k_cache_scale.data.flatten()
        # layer.k_cache_offset.data = layer.k_cache_offset.data.flatten()
        layer.v_cache_scale.data = layer.v_cache_scale.data.flatten()
        # layer.v_cache_offset.data = layer.v_cache_offset.data.flatten()
        layer.qk_rot = torch.load("/home/z00909726/block_rht_matrix.pt")

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "AscendC8KVCacheAttentionMethod.apply should not be called. "
            "C8 KV cache quantization is handled by the attention backend."
        )
        # logger.info_once(f"query.shape :{query.shape}, key.shape: {key.shape}, value.shape: {value.shape}, layer: {layer}")
        # return layer.impl.forward(query, key, value, kv_cache, attn_metadata, attn_type, scale, output)



# class AscendC4FakeAttentionMethod(AscendAttentionScheme):
#     """KV Cache FP4 per-channel 伪量化方案（无窗口，每次 forward 对待写入的 K/V 量化）"""

#     def __init__(self, quant_description: dict, prefix: str):
#         self.quant_description = quant_description
#         self.prefix = prefix

#     def create_weights(self, layer: torch.nn.Module) -> None:
#         # 伪量化不改变实际存储类型，保持默认的高精度 (如 BF16/FP16)
#         dtype = torch.bfloat16 if torch.get_default_dtype() == torch.bfloat16 else torch.float32
#         layer.k_fake_scale = torch.nn.Parameter(
#             torch.ones((layer.num_kv_heads, layer.head_dim), dtype=dtype), requires_grad=False
#         )
#         layer.v_fake_scale = torch.nn.Parameter(
#             torch.ones((layer.num_kv_heads, layer.head_dim), dtype=dtype), requires_grad=False
#         )

#     def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
#         # 将其 view 成 [1, N, D] 方便后续对 [Num_tokens, N, D] 的数据进行广播计算
#         layer.k_fake_scale.data = layer.k_fake_scale.data.view(1, layer.num_kv_heads, layer.head_dim)
#         layer.v_fake_scale.data = layer.v_fake_scale.data.view(1, layer.num_kv_heads, layer.head_dim)

#     def apply(
#         self,
#         layer: torch.nn.Module,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         kv_cache,
#         attn_metadata,
#         attn_type,
#         scale,
#         output,
#     ) -> torch.Tensor:
#         # 对当前 forward 待写入的 key/value 进行 FP4 per-channel 伪量化
#         # key/value 形状: [num_tokens, num_kv_heads, head_size]
#         if key is not None:
#             key = simulate_fp4_per_channel(key, layer.k_fake_scale)
#         if value is not None:
#             value = simulate_fp4_per_channel(value, layer.v_fake_scale)

#         # 正常执行 attention，量化后的 K/V 会被写入 kv_cache
#         return layer.impl.forward(query, key, value, kv_cache, attn_metadata, attn_type, scale, output)
    
# class AscendC4Window32FakeAttentionMethod(AscendAttentionScheme):
#     """32窗口周期性 KV Cache 伪量化方案"""

#     def __init__(self, quant_description: dict, prefix: str):
#         self.quant_description = quant_description
#         self.prefix = prefix

#     def create_weights(self, layer: torch.nn.Module) -> None:
#         # 1. 伪量化不改变实际存储类型，保持默认的高精度 (如 BF16/FP16)
#         # 2. 在 layer 上注册统计 forward 次数的计数器 count
#         if not hasattr(layer, "forward_count"):
#             layer.register_buffer("forward_count", torch.tensor(0, dtype=torch.long))

#         # 3. 创建 per-channel 级别的 scale，形状为 [num_kv_heads, head_dim] (即 N * D)
#         dtype = torch.bfloat16 if torch.get_default_dtype() == torch.bfloat16 else torch.float32
#         layer.k_fake_scale = torch.nn.Parameter(
#             torch.ones((layer.num_kv_heads, layer.head_dim), dtype=dtype), requires_grad=False
#         )
#         layer.v_fake_scale = torch.nn.Parameter(
#             torch.ones((layer.num_kv_heads, layer.head_dim), dtype=dtype), requires_grad=False
#         )

#     def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
#         # 将其 view 成 [1, N, D] 方便后续对 [Num_tokens, N, D] 的数据进行广播计算
#         layer.k_fake_scale.data = layer.k_fake_scale.data.view(1, layer.num_kv_heads, layer.head_dim)
#         layer.v_fake_scale.data = layer.v_fake_scale.data.view(1, layer.num_kv_heads, layer.head_dim)

#     def apply(
#         self,
#         layer: torch.nn.Module,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         kv_cache,
#         attn_metadata,
#         attn_type,
#         scale,
#         output,
#     ) -> torch.Tensor:
#         """
#         在每步 Attention 计算时触发的拦截器
#         """
#         # 1. 步长计数器自增
#         layer.forward_count += 1

#         # 2. 检查当前步长是否是 32 的整数倍
#         if layer.forward_count > 0 and layer.forward_count % 32 == 0:
            
#             # 从全局的元数据中获取当前 Request 分配的 Slot Mapping
#             # slot_mapping 记录了当前 Batch 中所有 Token 在 kv_cache 里的绝对一维物理索引
#             slot_mapping = getattr(attn_metadata, "slot_mapping", None)
            
#             if slot_mapping is not None and slot_mapping.numel() > 0:
#                 # 3. 提取最近 32 个操作的物理槽位
#                 # 注意：slot_mapping 是当前 Batch 所有处于活动状态的 token 索引。
#                 # 取最后 32 个位置，代表最近写入或处理的 32 个 KV Cache 槽位。
#                 recent_slots = slot_mapping[-32:] if slot_mapping.numel() >= 32 else slot_mapping
                
#                 # 4. 根据底层具体存储格式（vLLM V1 通常将 K 和 V 组织在同一个大 Tensor 的不同维度，或者分开独立的 Tensor）
#                 # kv_cache 通常是一个元组 (k_cache, v_cache) 或者统一大张量。
#                 # 昇腾适配层通常映射为独立的物理块，此处以分离的 (k_cache, v_cache) 为主逻辑：
#                 if isinstance(kv_cache, tuple) or isinstance(kv_cache, list):
#                     k_cache, v_cache = kv_cache[0], kv_cache[1]
#                 else:
#                     # 如果是统合的大 Cache 内存，需根据具体存储 stride 拆分，通常昇腾为独立元组
#                     k_cache, v_cache = kv_cache, kv_cache 

#                 # 5. 从物理 Cache 中把这 32 个历史 KV 捞出来
#                 # 物理 Cache 的形状一般为 [提取总槽位数, num_kv_heads, head_dim]
#                 # 用进阶索引（Advanced Indexing）直接切片出最近 32 个 Token 对应的高精度数据
#                 recent_k = k_cache[recent_slots]
#                 recent_v = v_cache[recent_slots]

#                 # 6. 对捞出来的最近 32 个高精度 KV 进行 FP4 伪量化计算
#                 # quantized_k = simulate_fp4_per_channel(recent_k, layer.k_fake_scale)
#                 # quantized_v = simulate_fp4_per_channel(recent_v, layer.v_fake_scale)

#                 # 7. 原地（In-place）写回到物理大 Cache 对应的槽位中
#                 # 此时由于类型没有变（仍是原始高精度），写回操作可以直接通过高精度的 copy_ 或赋值完成
#                 k_cache[recent_slots] = quantized_k
#                 v_cache[recent_slots] = quantized_v

#         # 8. 周期性窗口处理完毕（或未到触发时机），直接调用原生的 backend 算子执行实际的 Attention 计算
#         # 此时硬件算子从物理存储中读取的 KV 已经是被我们周期性截断污染过的数据了
#         return layer.impl.forward(query, key, value, kv_cache, attn_metadata, attn_type, scale, output)