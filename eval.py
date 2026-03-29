#!/usr/bin/env python3
"""
Unified evaluation entry for DCTransformer.

Usage:
    python eval.py --config configs/eval.color.yaml
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dctransformer.models.dctransformer import DCTransformer
from dctransformer.data.dataset import ColorFolderEvaluationDataset, EvaluationDataset
from dctransformer.utils.frequency_metrics import evaluate_coefficients_restoration
from dctransformer.utils.pixel_metrics import calculate_psnr, calculate_psnrb, calculate_ssim
from dctransformer.utils.runtime_utils import (
    load_state_dict_flexible,
    load_yaml_config,
    maybe_wrap_data_parallel,
    parse_device,
    setup_seed,
)
from dctransformer.data.image_ops import (
    imsave,
    load_checkpoint,
    mkdir,
    pad_to_rb,
    reshape_image_from_frequencies,
    tensor2single,
    to_rgb,
    unpad,
    unpad_rb,
    single2uint,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "seed": 42,
        "device": "auto",
        "use_data_parallel": True,
        "device_ids": [0, 1, 2, 3],
        "show_progress": True,
    },
    "model": {
        "type": "dctransformer",
        "mode": "color",  # color | gray
        "input_dim": 64,
        "dim": 160,
        "num_groups": 6,
        "num_blocks_in_group": 4,
    },
    "paths": {
        "ckpt_path": "",
        "output_dir": "./test_results",
    },
    "eval": {
        "double_jpeg": False,
        "sets": [],  # if empty: default by mode
        "qfs": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
        "save_images": True,
        "save_input_images": False,
        "save_gt_images": False,
        "print_per_image": False,
        "calc_freq_metrics": False,
        "single_refs": {
            "live1_pattern": "./testsets_mod8/live1/*.bmp",
            "bsds500_pattern": "./testsets_mod8/bsds500/*.jpg",
            "urban100_pattern": "./testsets_mod8/Urban100/*.png",
            "icb_pattern": "./testsets_mod8/icb-color8bit/*.ppm",
            "classic5_pattern": "./testsets_mod8/classic5/*.bmp",
            "set5_pattern": "./testsets_mod8/Set5/*.png",
        },
        "gray_color2gray_overrides": {
            "classic5": False,
            "set5": False,
        },
        "double": {
            "qf_pairs": [
                [10, 90, 0, 0],
                [90, 10, 0, 0],
                [75, 95, 0, 0],
                [95, 75, 0, 0],
                [10, 90, 4, 4],
                [90, 10, 4, 4],
                [75, 95, 4, 4],
                [95, 75, 4, 4],
            ],
            "jpeg_root": "./dataset/double_jpeg",
            "gt_pattern": "./testsets_mod8/live1/*.bmp",
            "output_subdir": "double_eval",
        },
    },
}


def build_model(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = cfg["model"]
    if model_cfg["type"] != "dctransformer":
        raise ValueError("Unified eval.py currently supports model.type=dctransformer only.")
    model = DCTransformer(
        mode=model_cfg["mode"],
        in_dim=int(model_cfg["input_dim"]),
        dim=int(model_cfg["dim"]),
        num_blocks_in_group=int(model_cfg["num_blocks_in_group"]),
        num_groups=int(model_cfg["num_groups"]),
    ).to(device)
    return model


def load_model_weights(cfg: dict[str, Any], model: nn.Module, device: torch.device) -> int:
    ckpt_path = str(cfg["paths"]["ckpt_path"]).strip()
    if not ckpt_path:
        raise ValueError("paths.ckpt_path is required for evaluation.")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = load_checkpoint(ckpt_path, device)
    load_state_dict_flexible(model, ckpt)
    epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
    return epoch


def align_img_hw(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w, ...], b[:h, :w, ...]


def tensor_to_uint_img(tensor: torch.Tensor) -> np.ndarray:
    return single2uint(tensor2single(tensor))


def get_default_sets(mode: str) -> list[str]:
    if mode == "gray":
        return ["classic5", "live1", "bsds500"]
    return ["urban100"]


def get_single_ref_pattern(eval_cfg: dict[str, Any], set_name: str) -> str:
    refs = eval_cfg["single_refs"]
    key = f"{set_name.lower()}_pattern"
    if key not in refs:
        raise ValueError(f"Missing eval.single_refs.{key} in config.")
    return refs[key]


def get_color2gray(mode: str, eval_cfg: dict[str, Any], set_name: str) -> bool:
    if mode != "gray":
        return False
    overrides = eval_cfg.get("gray_color2gray_overrides", {})
    if set_name in overrides:
        return bool(overrides[set_name])
    return True


def get_eval_summary_path(output_dir: str) -> Path:
    return Path(output_dir) / "eval_summary.txt"


def reset_eval_summary(output_dir: str) -> Path:
    summary_path = get_eval_summary_path(output_dir)
    if summary_path.exists():
        summary_path.unlink()
    return summary_path


def append_eval_summary(output_dir: str, text: str) -> Path:
    summary_path = get_eval_summary_path(output_dir)
    prefix = "\n\n" if summary_path.exists() and summary_path.stat().st_size > 0 else ""
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(prefix)
        f.write(text.rstrip())
        f.write("\n")
    return summary_path


def format_single_set_result(
    set_name: str,
    qfs: list[int],
    jpeg_psnr: list[float],
    jpeg_ssim: list[float],
    jpeg_psnrb: list[float],
    ours_psnr: list[float],
    ours_ssim: list[float],
    ours_psnrb: list[float],
    jpeg_js: list[float] | None,
    ours_js: list[float] | None,
    jpeg_bra: list[float] | None,
    ours_bra: list[float] | None,
) -> str:
    lines = [
        f"Evaluation on dataset: {set_name}",
        "qf \t jpeg_psnr \t jpeg_ssim \t jpeg_psnrb \t ours_psnr \t ours_ssim \t ours_psnrb",
    ]
    for i, qf in enumerate(qfs):
        lines.append(
            f"{qf} \t "
            f"{jpeg_psnr[i]:.3f} \t {jpeg_ssim[i]:.3f} \t {jpeg_psnrb[i]:.3f} \t "
            f"{ours_psnr[i]:.3f} (+{(ours_psnr[i]-jpeg_psnr[i]):.3f}) \t "
            f"{ours_ssim[i]:.3f} (+{(ours_ssim[i]-jpeg_ssim[i]):.3f}) \t "
            f"{ours_psnrb[i]:.3f} (+{(ours_psnrb[i]-jpeg_psnrb[i]):.3f})"
        )

    if jpeg_js is not None and ours_js is not None and jpeg_bra is not None and ours_bra is not None:
        lines.append("qf \t jpeg_js \t jpeg_bra \t ours_js \t ours_bra")
        for i, qf in enumerate(qfs):
            lines.append(
                f"{qf} \t {jpeg_js[i]:.4f} \t {jpeg_bra[i]:.4f} \t "
                f"{ours_js[i]:.4f} \t {ours_bra[i]:.4f}"
            )
    return "\n".join(lines)


def evaluate_single_set(
    cfg: dict[str, Any],
    model: nn.Module,
    model_win_size: int,
    device: torch.device,
    set_name: str,
) -> None:
    eval_cfg = cfg["eval"]
    mode = cfg["model"]["mode"]
    qfs = [int(qf) for qf in eval_cfg["qfs"]]
    output_dir = cfg["paths"]["output_dir"]
    save_images = bool(eval_cfg["save_images"])
    save_input_images = bool(eval_cfg.get("save_input_images", False))
    save_gt_images = bool(eval_cfg.get("save_gt_images", False))
    show_per_image = bool(eval_cfg.get("print_per_image", False))
    calc_freq = bool(eval_cfg.get("calc_freq_metrics", False))

    ref_pattern = get_single_ref_pattern(eval_cfg, set_name)
    refs_path = sorted(glob.glob(ref_pattern))
    if not refs_path:
        print(f"[skip] set={set_name}: no refs found by pattern: {ref_pattern}")
        return

    color2gray = get_color2gray(mode, eval_cfg, set_name)
    print(
        f"[eval] set={set_name} images={len(refs_path)} "
        f"qfs={qfs} total_passes={len(refs_path) * len(qfs)}"
    )

    avg_jpeg_psnr: list[float] = []
    avg_jpeg_ssim: list[float] = []
    avg_jpeg_psnrb: list[float] = []
    avg_ours_psnr: list[float] = []
    avg_ours_ssim: list[float] = []
    avg_ours_psnrb: list[float] = []
    avg_jpeg_js: list[float] | None = [] if calc_freq else None
    avg_ours_js: list[float] | None = [] if calc_freq else None
    avg_jpeg_bra: list[float] | None = [] if calc_freq else None
    avg_ours_bra: list[float] | None = [] if calc_freq else None

    for qf_idx, qf in enumerate(qfs, start=1):
        print(f"[eval] set={set_name} qf={qf} ({qf_idx}/{len(qfs)})")
        out_subdir = os.path.join(output_dir, f"{mode}_{set_name}_qf_{qf}")
        mkdir(out_subdir)

        dataset = EvaluationDataset(
            in_paths=refs_path,
            color_scale=mode,
            color2gray=color2gray,
            qf=qf,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        iterator = tqdm(
            loader,
            total=len(loader),
            desc=f"{set_name} qf={qf}",
            dynamic_ncols=True,
            leave=False,
            disable=not bool(cfg["runtime"]["show_progress"]),
        )

        jpeg_psnr_vals: list[float] = []
        jpeg_ssim_vals: list[float] = []
        jpeg_psnrb_vals: list[float] = []
        ours_psnr_vals: list[float] = []
        ours_ssim_vals: list[float] = []
        ours_psnrb_vals: list[float] = []
        jpeg_js_vals: list[float] = []
        ours_js_vals: list[float] = []
        jpeg_bra_vals: list[float] = []
        ours_bra_vals: list[float] = []

        with torch.no_grad():
            for itr, batch in enumerate(iterator):
                if mode == "gray":
                    qm, x, y, img_gt, pad_mul16 = batch
                    x, y, img_gt = x.to(device), y.to(device), img_gt.to(device)

                    x_padded, pad = pad_to_rb(x, model_win_size)
                    x_pred = model(x_padded)
                    x_pred = unpad_rb(x_pred, pad)

                    img_l_tensor = unpad(reshape_image_from_frequencies(x), pad_mul16)
                    img_e_tensor = unpad(reshape_image_from_frequencies(x_pred), pad_mul16)
                    img_h_tensor = img_gt

                    img_l = tensor_to_uint_img(img_l_tensor)
                    img_e = tensor_to_uint_img(img_e_tensor)
                    img_h = tensor_to_uint_img(img_h_tensor)

                    img_l, img_h = align_img_hw(img_l, img_h)
                    img_e, img_h = align_img_hw(img_e, img_h)

                    if calc_freq:
                        jpeg_js, jpeg_bra = evaluate_coefficients_restoration(x[0], y[0])
                        ours_js, ours_bra = evaluate_coefficients_restoration(x_pred[0], y[0])
                        jpeg_js_vals.append(float(jpeg_js.cpu().numpy()))
                        ours_js_vals.append(float(ours_js.cpu().numpy()))
                        jpeg_bra_vals.append(float(jpeg_bra.cpu().numpy()))
                        ours_bra_vals.append(float(ours_bra.cpu().numpy()))

                else:
                    qm, [in_y, in_cb, in_cr], y, img_gt, pad_mul16 = batch
                    in_y, in_cb, in_cr, y, img_gt = (
                        in_y.to(device),
                        in_cb.to(device),
                        in_cr.to(device),
                        y.to(device),
                        img_gt.to(device),
                    )

                    in_y_padded, pad = pad_to_rb(in_y, model_win_size)
                    in_cb_padded, _ = pad_to_rb(in_cb, model_win_size // 2)
                    in_cr_padded, _ = pad_to_rb(in_cr, model_win_size // 2)

                    pred_y, pred_cb, pred_cr = model(in_y_padded, in_cb_padded, in_cr_padded)
                    pred_y = unpad_rb(pred_y, pad)
                    pred_cb = unpad_rb(pred_cb, pad)
                    pred_cr = unpad_rb(pred_cr, pad)

                    if calc_freq:
                        color_jpeg_js = 0.0
                        color_jpeg_bra = 0.0
                        color_ours_js = 0.0
                        color_ours_bra = 0.0
                        for idx, (inp, pred) in enumerate(zip([in_y, in_cb, in_cr], [pred_y, pred_cb, pred_cr])):
                            if idx >= 1:
                                inp = torch.nn.functional.interpolate(inp, size=pred.shape[-2:], mode="bicubic")
                            jpeg_js, jpeg_bra = evaluate_coefficients_restoration(
                                inp[0], y[0, idx * 64 : (idx + 1) * 64, ...]
                            )
                            ours_js, ours_bra = evaluate_coefficients_restoration(
                                pred[0], y[0, idx * 64 : (idx + 1) * 64, ...]
                            )
                            color_jpeg_js += float(jpeg_js.cpu().numpy())
                            color_jpeg_bra += float(jpeg_bra.cpu().numpy())
                            color_ours_js += float(ours_js.cpu().numpy())
                            color_ours_bra += float(ours_bra.cpu().numpy())
                        jpeg_js_vals.append(color_jpeg_js / 3.0)
                        jpeg_bra_vals.append(color_jpeg_bra / 3.0)
                        ours_js_vals.append(color_ours_js / 3.0)
                        ours_bra_vals.append(color_ours_bra / 3.0)

                    y_chn = reshape_image_from_frequencies(in_y)
                    cb_chn = torch.nn.functional.interpolate(
                        reshape_image_from_frequencies(in_cb),
                        size=y_chn.shape[-2:],
                        mode="bilinear",
                    )
                    cr_chn = torch.nn.functional.interpolate(
                        reshape_image_from_frequencies(in_cr),
                        size=y_chn.shape[-2:],
                        mode="bilinear",
                    )
                    pred_y_chn = reshape_image_from_frequencies(pred_y)
                    pred_cb_chn = torch.nn.functional.interpolate(
                        reshape_image_from_frequencies(pred_cb),
                        size=pred_y_chn.shape[-2:],
                        mode="bilinear",
                    )
                    pred_cr_chn = torch.nn.functional.interpolate(
                        reshape_image_from_frequencies(pred_cr),
                        size=pred_y_chn.shape[-2:],
                        mode="bilinear",
                    )

                    ycbcr_img = torch.cat([y_chn, cb_chn, cr_chn], dim=1)
                    pred_ycbcr_img = torch.cat([pred_y_chn, pred_cb_chn, pred_cr_chn], dim=1)

                    img_l_tensor = unpad(to_rgb(ycbcr_img, data_range=1.0).contiguous(), pad_mul16)
                    img_e_tensor = unpad(to_rgb(pred_ycbcr_img, data_range=1.0).contiguous(), pad_mul16)
                    img_h_tensor = img_gt

                    img_l = tensor_to_uint_img(img_l_tensor)
                    img_e = tensor_to_uint_img(img_e_tensor)
                    img_h = tensor_to_uint_img(img_h_tensor)

                    img_l, img_h = align_img_hw(img_l, img_h)
                    img_e, img_h = align_img_hw(img_e, img_h)

                jpeg_psnr = calculate_psnr(img_l, img_h)
                jpeg_ssim = calculate_ssim(img_l, img_h)
                jpeg_psnrb = calculate_psnrb(img_h, img_l)
                ours_psnr = calculate_psnr(img_e, img_h)
                ours_ssim = calculate_ssim(img_e, img_h)
                ours_psnrb = calculate_psnrb(img_h, img_e)

                jpeg_psnr_vals.append(float(jpeg_psnr))
                jpeg_ssim_vals.append(float(jpeg_ssim))
                jpeg_psnrb_vals.append(float(jpeg_psnrb))
                ours_psnr_vals.append(float(ours_psnr))
                ours_ssim_vals.append(float(ours_ssim))
                ours_psnrb_vals.append(float(ours_psnrb))

                if save_images:
                    fname = f"{dataset.img_names[itr]}.png"
                    imsave(img_e, os.path.join(out_subdir, fname))
                    if save_input_images:
                        imsave(img_l, os.path.join(out_subdir, f"{dataset.img_names[itr]}__jpeg.png"))
                    if save_gt_images:
                        imsave(img_h, os.path.join(out_subdir, f"{dataset.img_names[itr]}__gt.png"))

                if show_per_image:
                    print(
                        f"{set_name} {dataset.img_names[itr]}.png qf_{qf} "
                        f"psnr:{ours_psnr:.3f} ssim:{ours_ssim:.3f} psnrb:{ours_psnrb:.3f}"
                    )

        avg_jpeg_psnr.append(float(np.mean(jpeg_psnr_vals)))
        avg_jpeg_ssim.append(float(np.mean(jpeg_ssim_vals)))
        avg_jpeg_psnrb.append(float(np.mean(jpeg_psnrb_vals)))
        avg_ours_psnr.append(float(np.mean(ours_psnr_vals)))
        avg_ours_ssim.append(float(np.mean(ours_ssim_vals)))
        avg_ours_psnrb.append(float(np.mean(ours_psnrb_vals)))

        if calc_freq and avg_jpeg_js is not None and avg_ours_js is not None and avg_jpeg_bra is not None and avg_ours_bra is not None:
            avg_jpeg_js.append(float(np.mean(jpeg_js_vals)))
            avg_ours_js.append(float(np.mean(ours_js_vals)))
            avg_jpeg_bra.append(float(np.mean(jpeg_bra_vals)))
            avg_ours_bra.append(float(np.mean(ours_bra_vals)))

    summary_text = format_single_set_result(
        set_name=set_name,
        qfs=qfs,
        jpeg_psnr=avg_jpeg_psnr,
        jpeg_ssim=avg_jpeg_ssim,
        jpeg_psnrb=avg_jpeg_psnrb,
        ours_psnr=avg_ours_psnr,
        ours_ssim=avg_ours_ssim,
        ours_psnrb=avg_ours_psnrb,
        jpeg_js=avg_jpeg_js,
        ours_js=avg_ours_js,
        jpeg_bra=avg_jpeg_bra,
        ours_bra=avg_ours_bra,
    )
    print()
    print(summary_text)
    summary_path = append_eval_summary(output_dir, summary_text)
    print(f"[saved] summary -> {summary_path}")


def evaluate_double_jpeg(cfg: dict[str, Any], model: nn.Module, model_win_size: int, device: torch.device) -> None:
    if cfg["model"]["mode"] != "color":
        raise ValueError("eval.double_jpeg=true currently supports model.mode=color only.")

    eval_cfg = cfg["eval"]
    dcfg = eval_cfg["double"]
    output_dir = cfg["paths"]["output_dir"]
    save_images = bool(eval_cfg["save_images"])
    show_per_image = bool(eval_cfg.get("print_per_image", False))

    gt_paths = sorted(glob.glob(dcfg["gt_pattern"]))
    if not gt_paths:
        print(f"[skip] no GT images found by pattern: {dcfg['gt_pattern']}")
        return

    summary_lines = [
        "Double JPEG evaluation",
        "qf_pair \t jpeg_psnr \t ours_psnr \t increase",
    ]
    print("\nDouble JPEG evaluation")
    print("qf_pair \t jpeg_psnr \t ours_psnr \t increase")
    with torch.no_grad():
        for pair in dcfg["qf_pairs"]:
            q1, q2, sh, sw = [int(x) for x in pair]
            jpeg_pattern = os.path.join(dcfg["jpeg_root"], f"shift{sh}{sw}_jpg", f"{q1}_{q2}", "*.jpg")
            jpeg_paths = sorted(glob.glob(jpeg_pattern))
            if not jpeg_paths:
                print(f"{pair} \t [skip: no jpeg inputs]")
                continue

            size = min(len(jpeg_paths), len(gt_paths))
            dataset = ColorFolderEvaluationDataset(
                jpeg_paths=jpeg_paths[:size],
                png_paths=gt_paths[:size],
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            iterator = tqdm(loader, total=len(loader), disable=not bool(cfg["runtime"]["show_progress"]))

            jpeg_psnr_vals: list[float] = []
            ours_psnr_vals: list[float] = []
            out_subdir = os.path.join(output_dir, dcfg.get("output_subdir", "double_eval"), f"{q1}_{q2}_shift{sh}{sw}")
            if save_images:
                mkdir(out_subdir)

            for _, (qm, [in_y, in_cb, in_cr], img_gt, filename) in enumerate(iterator):
                in_y, in_cb, in_cr = in_y.to(device), in_cb.to(device), in_cr.to(device)

                in_y_padded, pad = pad_to_rb(in_y, model_win_size)
                in_cb_padded, _ = pad_to_rb(in_cb, model_win_size // 2)
                in_cr_padded, _ = pad_to_rb(in_cr, model_win_size // 2)

                pred_y, pred_cb, pred_cr = model(in_y_padded, in_cb_padded, in_cr_padded)
                pred_y = unpad_rb(pred_y, pad)
                pred_cb = unpad_rb(pred_cb, pad)
                pred_cr = unpad_rb(pred_cr, pad)

                y_chn = reshape_image_from_frequencies(in_y)
                cb_chn = torch.nn.functional.interpolate(
                    reshape_image_from_frequencies(in_cb),
                    size=y_chn.shape[-2:],
                    mode="bilinear",
                )
                cr_chn = torch.nn.functional.interpolate(
                    reshape_image_from_frequencies(in_cr),
                    size=y_chn.shape[-2:],
                    mode="bilinear",
                )
                pred_y_chn = reshape_image_from_frequencies(pred_y)
                pred_cb_chn = torch.nn.functional.interpolate(
                    reshape_image_from_frequencies(pred_cb),
                    size=pred_y_chn.shape[-2:],
                    mode="bilinear",
                )
                pred_cr_chn = torch.nn.functional.interpolate(
                    reshape_image_from_frequencies(pred_cr),
                    size=pred_y_chn.shape[-2:],
                    mode="bilinear",
                )

                img_l = tensor_to_uint_img(to_rgb(torch.cat([y_chn, cb_chn, cr_chn], dim=1), data_range=1.0))
                img_e = tensor_to_uint_img(
                    to_rgb(torch.cat([pred_y_chn, pred_cb_chn, pred_cr_chn], dim=1), data_range=1.0)
                )
                img_h = tensor_to_uint_img(img_gt[..., sh:, sw:])

                img_l, img_h = align_img_hw(img_l, img_h)
                img_e, img_h = align_img_hw(img_e, img_h)

                jpeg_psnr = calculate_psnr(img_l, img_h)
                ours_psnr = calculate_psnr(img_e, img_h)
                jpeg_psnr_vals.append(float(jpeg_psnr))
                ours_psnr_vals.append(float(ours_psnr))

                if save_images:
                    name = filename[0] if isinstance(filename, (list, tuple)) else str(filename)
                    imsave(img_e, os.path.join(out_subdir, f"{name}.png"))

                if show_per_image:
                    print(
                        f"pair={pair} img={filename[0] if isinstance(filename, (list, tuple)) else filename} "
                        f"psnr={ours_psnr:.3f}"
                    )

            avg_jpeg = float(np.mean(jpeg_psnr_vals))
            avg_ours = float(np.mean(ours_psnr_vals))
            line = f"{pair} \t {avg_jpeg:.4f} \t {avg_ours:.4f} \t +{(avg_ours - avg_jpeg):.4f}"
            print(line)
            summary_lines.append(line)
    summary_path = append_eval_summary(output_dir, "\n".join(summary_lines))
    print(f"[saved] summary -> {summary_path}")


def run_eval(cfg: dict[str, Any]) -> None:
    setup_seed(int(cfg["runtime"]["seed"]))
    device = parse_device(cfg["runtime"])
    print("DEVICE:", device)

    mkdir(cfg["paths"]["output_dir"])
    summary_path = reset_eval_summary(cfg["paths"]["output_dir"])
    print(f"summary file: {summary_path}")

    model = build_model(cfg, device)
    last_epoch = load_model_weights(cfg, model, device)
    model = maybe_wrap_data_parallel(model, cfg["runtime"], device)
    model.eval()
    if isinstance(model, torch.nn.DataParallel):
        print(f"Eval mode: DataParallel on cuda devices {list(model.device_ids)}")
    elif device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"Eval mode: single-GPU on cuda:{device_index} (visible GPUs: {torch.cuda.device_count()})")
    else:
        print("Eval mode: CPU")
    print(f"ckpt epoch: {last_epoch}")

    model_win_size = model.module.window_size if isinstance(model, torch.nn.DataParallel) else model.window_size

    eval_cfg = cfg["eval"]
    if bool(eval_cfg["double_jpeg"]):
        evaluate_double_jpeg(cfg, model, model_win_size, device)
    else:
        sets = list(eval_cfg.get("sets", []))
        if not sets:
            sets = get_default_sets(cfg["model"]["mode"])
        print(f"eval sets: {sets}")
        for set_name in sets:
            evaluate_single_set(cfg, model, model_win_size, device, set_name.lower())

def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DCTransformer evaluation entry")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.color.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    cfg = load_yaml_config(args.config, DEFAULT_CONFIG)
    run_eval(cfg)


if __name__ == "__main__":
    main()
