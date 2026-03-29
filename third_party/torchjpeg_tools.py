"""
Source Lib: https://gitlab.com/torchjpeg/torchjpeg
"""


from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def _setup_local_torchjpeg_source() -> None:
    third_party_dir = Path(__file__).resolve().parent
    local_src = third_party_dir / "torchjpeg" / "src"
    if local_src.exists():
        src_str = str(local_src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_setup_local_torchjpeg_source()

from torchjpeg.dct import block_idct, fdct, to_ycbcr
from torchjpeg.metrics._psnr import psnr


def _channel_to_coeff_blocks(channel: Tensor) -> tuple[Tensor, tuple[int, int]]:
    """Convert one channel (H, W) to block coefficients (1, H/8, W/8, 8, 8)."""
    if channel.dim() != 2:
        raise ValueError(f"channel must be 2D, got shape={tuple(channel.shape)}")

    h, w = channel.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8

    ch = channel.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    if pad_h > 0 or pad_w > 0:
        ch = F.pad(ch, (0, pad_w, 0, pad_h), mode="replicate")

    coeff_map = fdct(ch.squeeze(0)).squeeze(0)  # (H',W')
    blocks = coeff_map.unfold(0, 8, 8).unfold(1, 8, 8).contiguous()  # (H'/8,W'/8,8,8)
    return blocks.unsqueeze(0), (h, w)


def _downsample_420(channel: Tensor) -> Tensor:
    h, w = channel.shape
    out_h = (h + 1) // 2
    out_w = (w + 1) // 2
    x = channel.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(out_h, out_w), mode="area")
    return x.squeeze(0).squeeze(0)


def _read_coefficients_from_bgr_image(bgr: np.ndarray):
    """
    Lightweight Python fallback for torchjpeg.codec.read_coefficients.

    This returns the same tuple layout:
    (dimensions, quantization, y_coefficients, cbcr_coefficients)
    but coefficients are reconstructed from decoded pixels, so they are
    approximation-compatible rather than bit-exact libjpeg coefficients.
    """
    if bgr.ndim == 2:
        gray = torch.from_numpy(bgr.astype("float32"))
        y_centered = gray - 128.0
        y_blocks, (h, w) = _channel_to_coeff_blocks(y_centered)

        dim = torch.tensor([[h, w]], dtype=torch.int32)
        quant = torch.ones((1, 8, 8), dtype=torch.float32)
        return dim, quant, y_blocks, None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32")
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
    ycbcr = to_ycbcr(rgb_t, data_range=1.0)

    # Keep same transform convention as current training target generation:
    # Y is centered; Cb/Cr stay in their direct converted domain.
    y_blocks, (hy, wy) = _channel_to_coeff_blocks(ycbcr[0] - 128.0)
    cb_blocks, (hc, wc) = _channel_to_coeff_blocks(_downsample_420(ycbcr[1]))
    cr_blocks, _ = _channel_to_coeff_blocks(_downsample_420(ycbcr[2]))

    cbcr_blocks = torch.cat([cb_blocks, cr_blocks], dim=0)

    dim = torch.tensor([[hy, wy], [hc, wc], [hc, wc]], dtype=torch.int32)
    quant = torch.ones((3, 8, 8), dtype=torch.float32)
    return dim, quant, y_blocks, cbcr_blocks


def read_coefficients_from_encoded_bytes(encoded: np.ndarray):
    bgr = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError("Failed to decode JPEG bytes in memory.")
    return _read_coefficients_from_bgr_image(bgr)


def read_coefficients(path: str):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return _read_coefficients_from_bgr_image(bgr)
