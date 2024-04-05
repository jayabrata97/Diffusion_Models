import torch
import torch.nn as nn


def time_embedding(time_steps, t_emb_dim):
    """Converts time steps tensor to an embedding using the sinusoidal time embedding formula from transformers
    Args:
        time_steps: 1D tensor of length batch size
        t_emb_dim: Embedding dimension
    Returns:
        B X D embedding representation of B time steps
    """
    assert t_emb_dim % 2 == 0
    factor = 10000 ** (
        (
            torch.arange(
                start=0,
                end=t_emb_dim // 2,
                dtype=torch.float32,
                device=time_steps.device,
            )
            / (t_emb_dim / 2)
        )
    )
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

    return t_emb


class DownBlock(nn.Module):
    """
    Down conv block with attention
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        down_sample=True,
        num_heads=4,
        num_layers=1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )
        self.t_emb_layers = nn.ModuleList(
            [
                nn.Sequential(nn.SiLU(), nn.Linear(t_emb_dim, out_channels))
                for _ in range(num_layers)
            ]
        )
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(8, out_channels) for _ in range(num_layers)]
        )
        self.attentions = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )
