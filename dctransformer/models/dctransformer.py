from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_modules import DCTransformerBlock


class DCTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        dim: int = 128,
        window_size: int = 4,
        num_groups: int = 6,
        num_blocks_in_group: int = 4,
        mode: str = "color",
    ) -> None:
        super().__init__()
        if mode not in {"color", "gray"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = 64
        self.num_groups = num_groups
        self.num_blocks_in_group = num_blocks_in_group
        self.window_size = window_size

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(),
        )
        if self.mode == "color":
            self.head_cb = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PReLU(),
                nn.ConvTranspose2d(
                    in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
                ),
            )
            self.head_cr = nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PReLU(),
                nn.ConvTranspose2d(
                    in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
                ),
            )
            self.head_concat = nn.Sequential(
                nn.Conv2d(in_channels=dim * 3, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.PReLU(),
            )

        self.bottleneck = nn.ModuleList(
            [
                DCTransformerBlock(dim=dim, num_blocks=self.num_blocks_in_group, win_size=self.window_size)
                for _ in range(self.num_groups)
            ]
        )

        self.tail_y = nn.Conv2d(dim, self.out_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.tail_cb = nn.Conv2d(dim, self.out_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.tail_cr = nn.Conv2d(dim, self.out_dim, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x_y, x_cb=None, x_cr=None):
        if self.mode == "color":
            if x_cb is None or x_cr is None:
                raise ValueError("Color mode expects x_cb and x_cr inputs.")
            fea_y = self.head(x_y)
            fea_cb = self.head_cb(x_cb)
            fea_cr = self.head_cr(x_cr)
            fea = self.head_concat(torch.cat([fea_y, fea_cb, fea_cr], dim=1))
        else:
            fea = self.head(x_y)

        for grp in self.bottleneck:
            fea = grp(fea) + fea

        out_y = self.tail_y(fea)
        out_y = out_y + x_y

        if self.mode == "color":
            out_cb = self.tail_cb(fea)
            out_cr = self.tail_cr(fea)
            out_cb = out_cb + F.interpolate(x_cb, size=out_cb.shape[-2:], mode="bilinear")
            out_cr = out_cr + F.interpolate(x_cr, size=out_cb.shape[-2:], mode="bilinear")
            return out_y, out_cb, out_cr

        return out_y
