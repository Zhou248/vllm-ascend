import torch
import math

def hadamard_matrix(n: int) -> torch.Tensor:
    """生成标准2的幂次阶哈达玛矩阵 H_n，H @ H.T = n * I"""
    assert (n & (n - 1)) == 0, "n必须是2的整数次幂"
    h = torch.tensor([[1.0]], dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h

def random_hadamard_block(block_size: int = 32, device: torch.device = None, dtype=torch.float32) -> torch.Tensor:
    """
    生成单个 32×32 随机哈达玛矩阵（随机符号翻转RHT）
    R = S @ H, S是对角±1随机矩阵
    R^T @ R = block_size * I
    """
    assert block_size == 32, "固定块大小32"
    # 标准哈达玛
    H = hadamard_matrix(block_size).to(device=device, dtype=dtype)
    # 随机±1对角符号
    sign = torch.randint(0, 2, (block_size,), device=device, dtype=dtype) * 2 - 1
    S = torch.diag(sign)
    # 随机哈达玛块 + 正交归一缩放
    scale = 1.0 / math.sqrt(block_size)
    R = scale * (S @ H)
    return R

def block_diag_random_hadamard(num_blocks: int, block_size: int = 32, device: torch.device = None, dtype=torch.float32) -> torch.Tensor:
    """
    构造分块对角随机哈达玛矩阵
    :param num_blocks: 对角线上有多少个32×32块
    :param block_size: 固定32
    :return: shape = [N*32, N*32] 分块对角矩阵，仅对角线为随机哈达玛块，其余0
    """
    blocks = []
    for _ in range(num_blocks):
        blk = random_hadamard_block(block_size, device, dtype)
        blocks.append(blk)
    # 拼接成分块对角
    mat = torch.block_diag(*blocks)
    return mat.to(device)