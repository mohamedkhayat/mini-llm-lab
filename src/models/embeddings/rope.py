import torch


# should make this a function later on
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    # i reprents pairs, so we have d/2 pairs
    # formula; is theta_i = 1 / base^(2i / d) i  = [1, 2, 3, ... d/2 -1] or just 2i = [2, 4, ... d -2]
    # [:, (head_dim // 2)] is just make sure at most we have d/2 pairs
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim)
    )
    # the formula is angle = m * theta_i, m is just position index
    positions = torch.arange(context_length)

    # inv freq is shape (head_dim //2,)
    # positions is shape (context_length,)
    # we need to reshape them to compute angles
    # inv frq to 1 x head dim and pos to context_length x 1 and broadcasting will handle the rest
    angles = inv_freq.unsqueeze(0) * positions.unsqueeze(1)

    # need to reshape angles from context_length, head_dim / 2 to context_length, head_dim

    angles = torch.concat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def compute_rope(x, cos, sin):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # shape is context_size, head_dim
    # needs to be 1, 1, seq_len, head_dim for broadcasting to work
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # this is an efficient vectorized implementation of rotation matrix
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated
