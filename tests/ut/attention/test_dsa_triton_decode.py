import torch

from vllm_ascend.attention.dsa_triton_decode import dequant_blocks_vec


def test_dequant_blocks_preserves_bfloat16_rope_bytes():
    block_size = 2
    rope = torch.tensor(
        [[1.0, -0.5] + [0.0] * 62, [3.0, 0.25] + [0.0] * 62],
        dtype=torch.bfloat16,
    )
    packed = torch.zeros((1, block_size, 1, 640), dtype=torch.uint8)
    packed[:, :, :, :128] = rope.view(torch.uint8).reshape(1, block_size, 1, 128)
    # Zero FP8 values with unit E8M0 scales.
    packed[:, :, :, 576:583] = 127

    result = dequant_blocks_vec(packed.view(torch.float8_e4m3fn), torch.tensor([[0]]))

    torch.testing.assert_close(result[0, :, 448:], rope)


def test_dequant_blocks_applies_scale_per_64_values():
    nope = torch.ones((1, 448), dtype=torch.float8_e4m3fn)
    packed = torch.zeros((1, 1, 1, 640), dtype=torch.uint8)
    packed[:, :, :, 128:576] = nope.view(torch.uint8).reshape(1, 1, 1, 448)
    packed[:, :, :, 576:583] = torch.arange(127, 134, dtype=torch.uint8)

    result = dequant_blocks_vec(packed.view(torch.float8_e4m3fn), torch.tensor([[0]]))
    expected = torch.cat(
        [torch.full((64,), float(2**exponent), dtype=torch.bfloat16) for exponent in range(7)]
    )

    torch.testing.assert_close(result[0, 0, :448], expected)
