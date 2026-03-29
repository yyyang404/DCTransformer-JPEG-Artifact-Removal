from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from third_party.torchjpeg_tools import block_idct


def zigzag(n: int) -> dict[int, tuple[int, int]]:
    """Return zigzag traversal order for n x n indices."""

    def compare(xy: tuple[int, int]) -> tuple[int, int]:
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)

    xs = range(n)
    return {
        index: pos
        for index, pos in enumerate(sorted(((x, y) for x in xs for y in xs), key=compare))
    }


def reshape_one_channel(coefficients: Tensor) -> Tensor:
    """
    Convert block DCT coefficients (N, H/8, W/8, 8, 8) to image tensor (N, 1, H, W).
    Output is normalized to [0, 1].
    """
    bs, h_div8, w_div8, block_h, block_w = coefficients.shape
    spatial_blocks = block_idct(coefficients) + 128.0
    blocks = spatial_blocks.reshape(bs, h_div8 * w_div8, block_h * block_w).transpose(1, 2)
    image = F.fold(
        blocks,
        output_size=(h_div8 * block_h, w_div8 * block_w),
        kernel_size=(block_h, block_w),
        stride=(block_h, block_w),
    )
    return image.clamp(0, 255) / 255.0


def reshape_image_from_frequencies(frequency_map: Tensor) -> Tensor:
    """
    Convert frequency map tensor (N, 64|192, H, W) to image tensor (N, 1|3, 8H, 8W).
    """
    n, c, h, w = frequency_map.shape
    if c == 64:
        blocked_map = frequency_map.reshape(n, 8, 8, h, w).permute(0, 3, 4, 1, 2)
        return reshape_one_channel(blocked_map)
    if c == 192:
        blocked_map_y = frequency_map[:, :64, ...].reshape(n, 8, 8, h, w).permute(0, 3, 4, 1, 2)
        blocked_map_cb = frequency_map[:, 64:128, ...].reshape(n, 8, 8, h, w).permute(0, 3, 4, 1, 2)
        blocked_map_cr = frequency_map[:, 128:, ...].reshape(n, 8, 8, h, w).permute(0, 3, 4, 1, 2)
        return torch.cat(
            [
                reshape_one_channel(blocked_map_y),
                reshape_one_channel(blocked_map_cb),
                reshape_one_channel(blocked_map_cr),
            ],
            dim=1,
        )
    raise ValueError(
        f"reshape_image_from_frequencies expects 64 or 192 channels, got {c}"
    )


def rgb2ycbcr(img: np.ndarray, only_y: bool = True) -> np.ndarray:
    """Matlab-style RGB to YCbCr conversion for uint8 or float images."""
    in_img_type = img.dtype
    img = img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.0

    if only_y:
        result = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        result = (
            np.matmul(
                img,
                [
                    [65.481, -37.797, 112.0],
                    [128.553, -74.203, -93.786],
                    [24.966, 112.0, -18.214],
                ],
            )
            / 255.0
            + [16, 128, 128]
        )

    if in_img_type == np.uint8:
        result = result.round()
    else:
        result /= 255.0
    return result.astype(in_img_type)


def to_rgb(x: Tensor, data_range: float = 255.0) -> Tensor:
    """Convert Tensor from YCbCr to RGB."""
    if data_range not in (1.0, 255.0):
        raise ValueError(f"data_range must be 1.0 or 255.0, got {data_range}")

    rgb_from_ycbcr = torch.tensor(
        [
            1.0,
            0.0,
            1.40200,
            1.0,
            -0.344136286,
            -0.714136286,
            1.0,
            1.77200,
            0.0,
        ],
        device=x.device,
    ).view(3, 3).transpose(0, 1)

    if data_range == 255.0:
        bias = torch.tensor([-179.456, 135.458816, -226.816], device=x.device).view(3, 1, 1)
    else:
        bias = torch.tensor([-0.70374902, 0.531211043, -0.88947451], device=x.device).view(3, 1, 1)

    out = torch.einsum("cv,...cxy->...vxy", [rgb_from_ycbcr, x])
    out += bias
    return out.contiguous()


def _compute_symmetric_pads(h: int, w: int, stride: int) -> tuple[int, int, int, int]:
    new_h = h if h % stride == 0 else h + stride - (h % stride)
    new_w = w if w % stride == 0 else w + stride - (w % stride)
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


def pad_to(x: Tensor | np.ndarray, stride: int) -> tuple[Tensor | np.ndarray, tuple[int, int, int, int]]:
    """Pad input symmetrically to be divisible by stride. Returns (padded, (l, r, t, b))."""
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = np.expand_dims(x, axis=2)
        h, w = x.shape[:2]
        pad_left, pad_right, pad_top, pad_bottom = _compute_symmetric_pads(h, w, stride)
        out = np.pad(
            x,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return out, (pad_left, pad_right, pad_top, pad_bottom)

    if isinstance(x, torch.Tensor):
        h, w = x.shape[-2:]
        pad_left, pad_right, pad_top, pad_bottom = _compute_symmetric_pads(h, w, stride)
        out = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        return out, (pad_left, pad_right, pad_top, pad_bottom)

    raise TypeError(f"Unsupported type for pad_to: {type(x)}")


def unpad(x: Tensor | np.ndarray, pads: tuple[int, int, int, int]) -> Tensor | np.ndarray:
    """Remove padding produced by pad_to/pad_to_rb. pads=(left, right, top, bottom)."""
    pad_left, pad_right, pad_top, pad_bottom = pads

    h_end = -pad_bottom if pad_bottom > 0 else None
    w_end = -pad_right if pad_right > 0 else None

    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            return x[pad_top:h_end, pad_left:w_end]
        return x[pad_top:h_end, pad_left:w_end, ...]

    if isinstance(x, torch.Tensor):
        return x[..., pad_top:h_end, pad_left:w_end]

    raise TypeError(f"Unsupported type for unpad: {type(x)}")


def save_checkpoint(save_path: str, epoch: int, model: nn.Module, optimizer, scheduler) -> None:
    if isinstance(model, nn.DataParallel):
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_dict": model_dict,
            "optimizer_dict": optimizer.state_dict(),
            "scheduler_dict": scheduler.state_dict(),
        },
        save_path,
    )
    print(f"Checkpoint saved successfully to: {save_path}")


def load_checkpoint(load_path: str, device: str | torch.device | None = None):
    if device is not None:
        ckpt = torch.load(load_path, map_location=torch.device(device))
        print(f"checkpoint load success from: {load_path}, device: {device}")
        return ckpt
    return torch.load(load_path)


def pad_to_rb(x: Tensor, stride: int) -> tuple[Tensor, tuple[int, int, int, int]]:
    """Pad only right/bottom to make H,W divisible by stride. Returns (padded, (l, r, t, b))."""
    h, w = x.shape[-2:]
    new_h = h if h % stride == 0 else h + stride - (h % stride)
    new_w = w if w % stride == 0 else w + stride - (w % stride)
    pad_left, pad_top = 0, 0
    pad_right = new_w - w
    pad_bottom = new_h - h
    out = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
    return out, (pad_left, pad_right, pad_top, pad_bottom)


def unpad_rb(x: Tensor, pads: tuple[int, int, int, int]) -> Tensor:
    return unpad(x, pads)


def tensor2single(img: Tensor) -> np.ndarray:
    out = img.detach().squeeze().float().cpu().numpy()
    if out.ndim == 3:
        out = np.transpose(out, (1, 2, 0))
    return out


def single2uint(img: np.ndarray) -> np.ndarray:
    return np.uint8((img.clip(0, 1) * 255.0).round())


def imsave(img: np.ndarray, img_path: str) -> None:
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
