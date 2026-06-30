import torch
import torch.nn as nn
import constants

class RoPE(nn.Module):
    """
    Rotary Position Embedding for attention Q/K tensors.

    Expected input shape:
        x: (batch, heads, seq_len, head_dim)

    Returns:
        x_rotated: same shape as x
    """

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self):
        super().__init__()

        # inv_freq has shape (head_dim / 2,)
        #
        # For head_dim = 8, this corresponds to frequency indices:
        #   [0, 2, 4, 6]
        #
        # inv_freq[i] = 1 / constants.rope_base^(i / head_dim)
        inv_freq = 1.0 / (
            constants.rope_base ** (torch.arange(0, constants.attention_head_dim, 2).float() / constants.attention_head_dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache.
        self._build_cache(constants.num_patches)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # angles shape: (seq_len, head_dim / 2)
        #
        # angles[pos, i] = pos * inv_freq[i]
        angles = torch.outer(positions, self.inv_freq)

        # Shapes for broadcasting over batch and heads:
        # cos/sin: (1, 1, seq_len, head_dim / 2)
        self.register_buffer(
            "cos_cached",
            angles.cos()[None, None, :, :],
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            angles.sin()[None, None, :, :],
            persistent=False,
        )

        constants.num_patches = seq_len

    def forward(self, x: torch.Tensor, seq_dim: int = -2) -> torch.Tensor:
        """
        Apply RoPE to x.

        Default assumes x shape:
            (batch, heads, seq_len, head_dim)

        where seq_dim = -2.
        """

        if x.shape[-1] != constants.attention_head_dim:
            raise ValueError(
                f"Expected last dim {constants.attention_head_dim}, got {x.shape[-1]}"
            )

        seq_len = x.shape[seq_dim]

        if seq_len > constants.num_patches:
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype)

        # Split even and odd coordinates.
        #
        # x_even: coordinates 0, 2, 4, ...
        # x_odd:  coordinates 1, 3, 5, ...
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply 2D rotation to each pair:
        #
        # [x_even'] = [ cos  -sin ] [x_even]
        # [x_odd' ]   [ sin   cos ] [x_odd ]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos

        # Interleave even and odd dimensions back together.
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_rotated_even
        x_out[..., 1::2] = x_rotated_odd

        return x_out
