import torch

from models.attention.mha import MultiHeadAttention


def test_multi_head_attention_example():
    torch.manual_seed(123)
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ]
    )
    batch = torch.stack((inputs, inputs), dim=0)

    _, context_length, d_in = batch.shape
    mha = MultiHeadAttention(d_in, 2, context_length, 0.0, num_heads=2)

    context_vecs = mha(batch)

    assert context_vecs.shape == (2, 6, 2)
    assert torch.isfinite(context_vecs).all()
    torch.testing.assert_close(context_vecs[0], context_vecs[1])
