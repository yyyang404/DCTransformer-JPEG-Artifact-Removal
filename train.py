#!/usr/bin/env python3
"""
Unified training entry for DCTransformer.

Usage:
    python train.py --config configs/train.color.yaml
    CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --config configs/train.color.yaml
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import os
import random
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import yaml


EARLY_DEFAULT_CONFIG_PATH = "configs/train.color.yaml"
EARLY_RUNTIME_DEFAULTS: dict[str, Any] = {
    "device": "auto",
    "use_data_parallel": True,
    "device_ids": [0, 1, 2, 3],
}
RUN_TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
CHECKPOINT_INDEX_NAME = "checkpoint_index.json"
BEST_CHECKPOINT_ALIAS = "best.ckpt"
BEST_CHECKPOINT_INFO = "best_checkpoint.txt"
CHECKPOINT_NAME_RE = re.compile(
    r"^(?P<mode>.+)_epoch(?P<epoch>\d+)_tloss(?P<train_loss>\d+(?:\.\d+)?)"
    r"(?:_(?:avg)?psnr(?P<metric>\d+(?:\.\d+)?))?\.ckpt$"
)


def _get_cli_config_path(argv: list[str]) -> str:
    config_path = EARLY_DEFAULT_CONFIG_PATH
    for idx, arg in enumerate(argv):
        if arg == "--config" and idx + 1 < len(argv):
            return argv[idx + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return config_path


def _load_early_runtime_cfg(config_path: str) -> dict[str, Any]:
    runtime_cfg = dict(EARLY_RUNTIME_DEFAULTS)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        if isinstance(user_cfg, dict) and isinstance(user_cfg.get("runtime"), dict):
            runtime_cfg.update(user_cfg["runtime"])
    except Exception:
        pass
    return runtime_cfg


def _maybe_bind_visible_devices_from_config() -> None:
    if Path(sys.argv[0]).name != "train.py":
        return
    if os.environ.get("_DCTRANSFORMER_GPU_BOUND") == "1":
        return

    runtime_cfg = _load_early_runtime_cfg(_get_cli_config_path(sys.argv[1:]))
    device_name = str(runtime_cfg.get("device", "auto")).strip().lower()
    if device_name == "cpu":
        return

    raw_device_ids = runtime_cfg.get("device_ids", [])
    device_ids = [int(x) for x in raw_device_ids]
    if not device_ids:
        return

    if bool(runtime_cfg.get("use_data_parallel", True)):
        visible_ids = device_ids
    else:
        visible_ids = [device_ids[0]]

    desired_visible = ",".join(str(x) for x in visible_ids)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = desired_visible
    env["_DCTRANSFORMER_GPU_BOUND"] = "1"
    env["_DCTRANSFORMER_BOUND_GPU_COUNT"] = str(len(visible_ids))
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_maybe_bind_visible_devices_from_config()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dctransformer.models.dctransformer import DCTransformer
from third_party.torchjpeg_tools import psnr
from dctransformer.utils.runtime_utils import (
    load_state_dict_flexible,
    load_yaml_config,
    maybe_wrap_data_parallel,
    parse_device,
    setup_seed,
)
from dctransformer.data.dataset import (
    ColorFolderEvaluationDataset,
    DCTCollocatedMapDataset,
    DCTCollocatedMapDataset_DoubleJPEG,
    EvaluationDataset,
    SpeedyDCTCollocatedMapDataset,
)
from dctransformer.models.domain_losses import (
    ColorDualDomainLoss,
    ColorFreqEnhancedLoss,
    ColorPixelDomainLoss,
    GrayDualDomainLoss,
    GrayFreqEnhancedLoss,
    GrayPixelDomainLoss,
)
from dctransformer.data.image_ops import (
    load_checkpoint,
    mkdir,
    pad_to_rb,
    reshape_image_from_frequencies,
    save_checkpoint,
    to_rgb,
    unpad,
    unpad_rb,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "seed": 42,
        "device": "auto",
        "use_data_parallel": True,
        "device_ids": [0, 1, 2, 3],
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
        "save_ckpt_path": "./local_save",
        "resume_ckpt_path": "",
        "load_model_path": "",
    },
    "checkpoint": {
        "save_top_k": 10,
        "keep_last_k": 2,
    },
    "train": {
        "enabled": True,
        "double_jpeg": False,
        "epochs": 300,
        "batch_size": 96,
        "num_workers": 8,
        "patch_size": 320,
        "loader_type": "standard",  # standard | speedy
        "qf_range": [10, 100, 10],
        "custom_qf": False,
        "custom_shift": False,
        "sample_per_epoch": 0,
        "limit_train_images": 0,
        "max_train_iters": 0,
        "perf_log_interval": 50,
        "save_every_epoch_when_no_eval": True,
    },
    "data": {
        "train_patterns": [],
    },
    "loss": {
        "name": "dual-c",  # dual-c | dual-g | freq_wise-c | freq_wise-g | pixel-c | pixel-g | l1
        "balance_ratio": 255.0,
        "freq_mode": "low",
        "ycbcr_ratio": [2.0, 1.0],
    },
    "optimizer": {
        "name": "adamw",  # adam | adamw
        "base_lr": 2e-4,
        "grad_clip": 0.2,
    },
    "scheduler": {
        "name": "warmup+cosine",  # multistep | cosine | warmup+cosine
        "multistep_milestones": [30],
        "multistep_gamma": 0.5,
        "cosine_eta_min": 1e-4,
        "warmup_ratio": 0.2,
        "warmup_start_lr": 1e-5,
        "cosine_max_lr": 1e-5,
        "cosine_min_lr": 1e-6,
    },
    "eval": {
        "enabled": True,
        "every_n_epochs": 1,
        "set": "live1",  # live1 | bsds500 | classic5
        "sets": [],
        "qfs": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "single_refs": {
            "live1_pattern": "./testsets_mod8/live1/*.bmp",
            "bsds500_pattern": "./testsets_mod8/bsds500/*.jpg",
            "classic5_pattern": "./testsets_mod8/classic5/*.bmp",
        },
    },
    "wandb": {
        "enabled": False,
        "project": "dctransformer",
        "dir": "./local_save/wandb",
        "entity": "",
        "run_name": "",
        "group": "",
        "tags": [],
        "notes": "",
        "mode": "online",  # online | offline | disabled
        "log_interval": 50,
    },
}


def normalize_runtime_after_gpu_binding(cfg: dict[str, Any]) -> dict[str, Any]:
    bound_gpu_count = os.environ.get("_DCTRANSFORMER_BOUND_GPU_COUNT")
    if not bound_gpu_count:
        return cfg

    count = max(1, int(bound_gpu_count))
    cfg["runtime"]["device_ids"] = list(range(count))

    device_name = str(cfg["runtime"].get("device", "auto")).strip().lower()
    if device_name == "auto" or device_name.startswith("cuda"):
        cfg["runtime"]["device"] = "cuda:0"
    return cfg


def dump_effective_config(cfg: dict[str, Any], save_ckpt_path: str) -> Path:
    mkdir(save_ckpt_path)
    out = Path(save_ckpt_path) / "effective_config.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
    return out


def infer_checkpoint_entry_from_filename(filename: str) -> dict[str, Any] | None:
    if filename == BEST_CHECKPOINT_ALIAS:
        return None
    match = CHECKPOINT_NAME_RE.match(filename)
    if match is None:
        return None
    metric = match.group("metric")
    return {
        "filename": filename,
        "epoch": int(match.group("epoch")),
        "train_loss": float(match.group("train_loss")),
        "metric": float(metric) if metric is not None else None,
    }


def load_checkpoint_index(save_ckpt_path: str) -> dict[str, Any]:
    save_dir = Path(save_ckpt_path)
    index_path = save_dir / CHECKPOINT_INDEX_NAME
    entries: list[dict[str, Any]] = []
    if index_path.exists():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and isinstance(payload.get("entries"), list):
                entries.extend(x for x in payload["entries"] if isinstance(x, dict))
        except (OSError, json.JSONDecodeError):
            entries = []

    known = {
        str(entry.get("filename", "")).strip()
        for entry in entries
        if str(entry.get("filename", "")).strip()
    }
    for ckpt_path in sorted(save_dir.glob("*.ckpt")):
        entry = infer_checkpoint_entry_from_filename(ckpt_path.name)
        if entry is None or entry["filename"] in known:
            continue
        entries.append(entry)
        known.add(entry["filename"])

    filtered_entries: list[dict[str, Any]] = []
    for entry in entries:
        filename = str(entry.get("filename", "")).strip()
        if not filename:
            continue
        if filename == BEST_CHECKPOINT_ALIAS:
            continue
        if not (save_dir / filename).exists():
            continue
        filtered_entries.append(
            {
                "filename": filename,
                "epoch": int(entry.get("epoch", -1)),
                "train_loss": float(entry.get("train_loss", 0.0)),
                "metric": None if entry.get("metric") is None else float(entry.get("metric")),
            }
        )

    filtered_entries.sort(
        key=lambda item: (int(item["epoch"]), str(item["filename"])),
        reverse=True,
    )
    best_entry = max(
        (entry for entry in filtered_entries if entry["metric"] is not None),
        key=lambda item: (float(item["metric"]), int(item["epoch"]), str(item["filename"])),
        default=None,
    )
    return {
        "entries": filtered_entries,
        "best_filename": "" if best_entry is None else str(best_entry["filename"]),
    }


def save_checkpoint_index(save_ckpt_path: str, index: dict[str, Any]) -> Path:
    save_dir = Path(save_ckpt_path)
    mkdir(str(save_dir))
    index_path = save_dir / CHECKPOINT_INDEX_NAME
    payload = {
        "best_filename": str(index.get("best_filename", "")),
        "entries": list(index.get("entries", [])),
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
    return index_path


def update_best_checkpoint_markers(save_ckpt_path: str, best_entry: dict[str, Any] | None) -> None:
    save_dir = Path(save_ckpt_path)
    alias_path = save_dir / BEST_CHECKPOINT_ALIAS
    info_path = save_dir / BEST_CHECKPOINT_INFO

    if alias_path.exists() or alias_path.is_symlink():
        alias_path.unlink()

    if best_entry is None:
        info_path.write_text("best_checkpoint: none\n", encoding="utf-8")
        return

    try:
        alias_path.symlink_to(best_entry["filename"])
    except OSError:
        pass

    info_path.write_text(
        "\n".join(
            [
                f"filename: {best_entry['filename']}",
                f"epoch: {best_entry['epoch']}",
                f"metric: {best_entry['metric']:.6f}",
                f"train_loss: {best_entry['train_loss']:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def apply_checkpoint_retention(
    cfg: dict[str, Any],
    ckpt_path: str,
    epoch: int,
    train_loss: float,
    eval_psnrs: list[float],
) -> None:
    save_ckpt_path = str(cfg["paths"]["save_ckpt_path"])
    ckpt_cfg = cfg["checkpoint"]
    save_top_k = max(0, int(ckpt_cfg.get("save_top_k", 10)))
    keep_last_k = max(0, int(ckpt_cfg.get("keep_last_k", 2)))
    index = load_checkpoint_index(save_ckpt_path)

    current_entry = {
        "filename": Path(ckpt_path).name,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "metric": float(np.mean(eval_psnrs)) if eval_psnrs else None,
    }

    entries = [entry for entry in index["entries"] if entry["filename"] != current_entry["filename"]]
    entries.append(current_entry)

    if save_top_k <= 0 and keep_last_k <= 0:
        kept_entries = sorted(entries, key=lambda item: (int(item["epoch"]), str(item["filename"])), reverse=True)
        removed_entries: list[dict[str, Any]] = []
    else:
        keep_names: set[str] = set()
        if save_top_k > 0:
            ranked_entries = sorted(
                (entry for entry in entries if entry["metric"] is not None),
                key=lambda item: (float(item["metric"]), int(item["epoch"]), str(item["filename"])),
                reverse=True,
            )
            keep_names.update(entry["filename"] for entry in ranked_entries[:save_top_k])
        if keep_last_k > 0:
            recent_entries = sorted(
                entries,
                key=lambda item: (int(item["epoch"]), str(item["filename"])),
                reverse=True,
            )
            keep_names.update(entry["filename"] for entry in recent_entries[:keep_last_k])
        if not keep_names and entries:
            newest_entry = max(entries, key=lambda item: (int(item["epoch"]), str(item["filename"])))
            keep_names.add(str(newest_entry["filename"]))

        kept_entries = [entry for entry in entries if entry["filename"] in keep_names]
        removed_entries = [entry for entry in entries if entry["filename"] not in keep_names]

    for entry in removed_entries:
        stale_path = Path(save_ckpt_path) / str(entry["filename"])
        if stale_path.exists():
            stale_path.unlink()

    kept_entries.sort(key=lambda item: (int(item["epoch"]), str(item["filename"])), reverse=True)
    best_entry = max(
        (entry for entry in kept_entries if entry["metric"] is not None),
        key=lambda item: (float(item["metric"]), int(item["epoch"]), str(item["filename"])),
        default=None,
    )
    new_index = {
        "entries": kept_entries,
        "best_filename": "" if best_entry is None else str(best_entry["filename"]),
    }
    save_checkpoint_index(save_ckpt_path, new_index)
    update_best_checkpoint_markers(save_ckpt_path, best_entry)

    metric_text = "n/a" if current_entry["metric"] is None else f"{current_entry['metric']:.4f}"
    print(
        f"Checkpoint retention: saved={current_entry['filename']}, avg_psnr={metric_text}, "
        f"top_k={save_top_k}, keep_last_k={keep_last_k}, kept={len(kept_entries)}, removed={len(removed_entries)}"
    )
    if best_entry is not None:
        print(
            f"Current best checkpoint: {best_entry['filename']} "
            f"(epoch={best_entry['epoch']}, avg_psnr={best_entry['metric']:.4f})"
        )


def resolve_train_paths(cfg: dict[str, Any]) -> list[str]:
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]

    patterns = list(data_cfg.get("train_patterns", []))
    if not patterns:
        raise RuntimeError("No training patterns configured. Please set data.train_patterns in your YAML config.")

    in_paths: list[str] = []
    for pattern in patterns:
        in_paths.extend(glob.glob(pattern, recursive=True))
    in_paths = sorted(in_paths)

    limit_images = int(train_cfg.get("limit_train_images", 0))
    if limit_images > 0:
        in_paths = in_paths[:limit_images]

    if not in_paths:
        raise RuntimeError(
            "No training images found. "
            "Please update data.train_patterns in your YAML config."
        )

    return in_paths


def resolve_single_eval_sets(eval_cfg: dict[str, Any]) -> list[str]:
    raw_sets = eval_cfg.get("sets", [])
    if raw_sets:
        eval_sets = [str(x).lower() for x in raw_sets]
    else:
        eval_sets = [str(eval_cfg.get("set", "live1")).lower()]

    supported = {"live1", "bsds500", "classic5"}
    for eval_set in eval_sets:
        if eval_set not in supported:
            raise ValueError(f"Unsupported eval set: {eval_set}")
    return eval_sets


def resolve_single_eval_refs(eval_cfg: dict[str, Any], eval_set: str) -> list[str]:
    ref_cfg = eval_cfg["single_refs"]
    if eval_set == "live1":
        pattern = ref_cfg["live1_pattern"]
    elif eval_set == "bsds500":
        pattern = ref_cfg["bsds500_pattern"]
    elif eval_set == "classic5":
        pattern = ref_cfg["classic5_pattern"]
    else:
        raise ValueError(f"Unsupported eval.set: {eval_set}")
    return sorted(glob.glob(pattern))


def build_model(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    if model_type != "dctransformer":
        raise ValueError("Unified train.py currently supports model.type=dctransformer only.")

    model = DCTransformer(
        mode=model_cfg["mode"],
        in_dim=int(model_cfg["input_dim"]),
        dim=int(model_cfg["dim"]),
        num_blocks_in_group=int(model_cfg["num_blocks_in_group"]),
        num_groups=int(model_cfg["num_groups"]),
    ).to(device)
    return model


def print_model_summary(cfg: dict[str, Any], model: nn.Module) -> None:
    model_cfg = cfg["model"]
    summary_device = str(next(model.parameters()).device)
    if model_cfg["mode"] == "color":
        stats = summary(
            model,
            input_size=[(1, 64, 32, 32), (1, 64, 16, 16), (1, 64, 16, 16)],
            device=summary_device,
            verbose=0,
        )
        print(stats)
    else:
        stats = summary(
            model,
            input_size=(1, 64, 32, 32),
            device=summary_device,
            verbose=0,
        )
        print(stats)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_wandb_run_name(wandb_cfg: dict[str, Any]) -> str:
    base_name = str(wandb_cfg.get("run_name", "")).strip()
    if base_name:
        return f"{base_name}-{RUN_TIMESTAMP}"
    return RUN_TIMESTAMP


def init_wandb(
    cfg: dict[str, Any],
    model: nn.Module,
    effective_config_path: Path,
    num_train_images: int,
    train_loader_len: int,
    last_epoch: int,
):
    wandb_cfg = cfg.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb.enabled=true but wandb is not installed. "
            "Install it with `pip install wandb` or reinstall requirements.txt."
        ) from exc

    entity = str(wandb_cfg.get("entity", "")).strip() or None
    run_name = build_wandb_run_name(wandb_cfg)
    group = str(wandb_cfg.get("group", "")).strip() or None
    notes = str(wandb_cfg.get("notes", "")).strip() or None
    mode = str(wandb_cfg.get("mode", "online")).strip().lower()
    tags = [str(tag) for tag in wandb_cfg.get("tags", [])]
    wandb_dir = str(wandb_cfg.get("dir", "./local_save/wandb")).strip() or "./local_save/wandb"
    mkdir(wandb_dir)

    wandb.init(
        project=str(wandb_cfg["project"]),
        entity=entity,
        name=run_name,
        group=group,
        notes=notes,
        tags=tags,
        dir=wandb_dir,
        config=cfg,
        mode=mode,
    )

    total_params, trainable_params = count_parameters(model)
    if wandb.run is not None:
        wandb.run.summary["model/total_params"] = total_params
        wandb.run.summary["model/trainable_params"] = trainable_params
        wandb.run.summary["model/window_size"] = int(model.window_size)
        wandb.run.summary["train/num_images"] = int(num_train_images)
        wandb.run.summary["train/loader_len"] = int(train_loader_len)
        wandb.run.summary["train/resume_from_epoch"] = int(last_epoch)
        wandb.run.summary["paths/save_ckpt_path"] = str(cfg["paths"]["save_ckpt_path"])
        wandb.run.summary["paths/wandb_dir"] = str(wandb_dir)
        wandb.run.summary["paths/effective_config"] = str(effective_config_path)
        wandb.run.summary["wandb/run_name"] = str(run_name)
        wandb.run.summary["wandb/run_timestamp"] = RUN_TIMESTAMP
        print(f"W&B resolved run_name: {run_name}")
    return wandb


def build_eval_log_metrics(
    cfg: dict[str, Any],
    eval_mode: str,
    eval_loaders: list[tuple[Any, DataLoader]],
    eval_psnrs: list[float],
    jpeg_psnrs: list[float],
) -> dict[str, float]:
    if not eval_psnrs or not jpeg_psnrs:
        return {}

    model_mode = str(cfg["model"]["mode"]).lower()
    if eval_mode == "double":
        base_prefix = f"eval/{model_mode}_double_jpeg"
    else:
        base_prefix = f"eval/{model_mode}"

    if eval_mode == "double":
        metrics: dict[str, float] = {
            f"{base_prefix}/avg_model_psnr": float(np.mean(eval_psnrs)),
        }
        for (item, _), model_psnr, _ in zip(eval_loaders, eval_psnrs, jpeg_psnrs):
            q1, q2, sh, sw = item
            tag = f"qf_{q1}_{q2}_shift_{sh}_{sw}"
            metrics[f"{base_prefix}/{tag}/model_psnr"] = float(model_psnr)
        return metrics

    per_set: dict[str, dict[str, float]] = {}
    for (item, _), model_psnr, _ in zip(eval_loaders, eval_psnrs, jpeg_psnrs):
        eval_set, qf = item
        eval_set = str(eval_set).lower()
        qf = int(qf)
        per_set.setdefault(eval_set, {})[f"qf_{qf}"] = float(model_psnr)

    metrics = {}
    all_psnrs: list[float] = []
    for eval_set, qf_map in per_set.items():
        set_psnrs = list(qf_map.values())
        all_psnrs.extend(set_psnrs)
        metrics[f"{base_prefix}/{eval_set}/avg_model_psnr"] = float(np.mean(set_psnrs))
        for qf_tag, model_psnr in qf_map.items():
            metrics[f"{base_prefix}/{eval_set}/{qf_tag}/model_psnr"] = float(model_psnr)

    if all_psnrs:
        metrics[f"{base_prefix}/overall/avg_model_psnr"] = float(np.mean(all_psnrs))
    return metrics


def build_train_dataloader(cfg: dict[str, Any], in_paths: list[str]) -> DataLoader:
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    qf_range = tuple(train_cfg["qf_range"])

    if train_cfg["double_jpeg"]:
        dataset = DCTCollocatedMapDataset_DoubleJPEG(
            in_paths=in_paths,
            crop_size=[train_cfg["patch_size"], train_cfg["patch_size"]],
            color_scale=model_cfg["mode"],
            qf_range=qf_range,
            custom_qf=bool(train_cfg["custom_qf"]),
            custom_shift=bool(train_cfg["custom_shift"]),
        )
        return DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            drop_last=True,
            num_workers=int(train_cfg["num_workers"]),
        )

    loader_type = str(train_cfg["loader_type"]).lower()
    if loader_type == "speedy":
        dataset = SpeedyDCTCollocatedMapDataset(
            in_paths=in_paths,
            crop_size=[train_cfg["patch_size"], train_cfg["patch_size"]],
            color_scale=model_cfg["mode"],
            batch_size=int(train_cfg["batch_size"]),
            qf_range=qf_range,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            num_workers=int(train_cfg["num_workers"]),
        )

    dataset = DCTCollocatedMapDataset(
        in_paths=in_paths,
        crop_size=[train_cfg["patch_size"], train_cfg["patch_size"]],
        color_scale=model_cfg["mode"],
        qf_range=qf_range,
        custom_qf=bool(train_cfg["custom_qf"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(train_cfg["num_workers"]),
    )


def build_eval_loaders(cfg: dict[str, Any]) -> tuple[str, list[tuple[Any, DataLoader]]]:
    if not cfg["eval"]["enabled"]:
        return "none", []

    model_mode = cfg["model"]["mode"]
    is_double = bool(cfg["train"]["double_jpeg"])

    if is_double:
        if model_mode != "color":
            raise ValueError("double_jpeg mode currently supports color model only.")
        dcfg = cfg["eval"]["double"]
        gt_paths = sorted(glob.glob(dcfg["gt_pattern"]))
        loaders: list[tuple[Any, DataLoader]] = []
        for pair in dcfg["qf_pairs"]:
            q1, q2, sh, sw = [int(x) for x in pair]
            jpeg_pattern = os.path.join(dcfg["jpeg_root"], f"shift{sh}{sw}_jpg", f"{q1}_{q2}", "*.jpg")
            jpeg_paths = sorted(glob.glob(jpeg_pattern))
            if not jpeg_paths or not gt_paths:
                continue
            size = min(len(jpeg_paths), len(gt_paths))
            dataset = ColorFolderEvaluationDataset(
                jpeg_paths=jpeg_paths[:size],
                png_paths=gt_paths[:size],
            )
            loaders.append(((q1, q2, sh, sw), DataLoader(dataset, batch_size=1, shuffle=False)))
        return "double", loaders

    eval_sets = resolve_single_eval_sets(cfg["eval"])
    loaders = []
    eval_qfs = [int(qf) for qf in cfg["eval"]["qfs"]]
    for eval_set in eval_sets:
        refs = resolve_single_eval_refs(cfg["eval"], eval_set)
        if not refs:
            continue
        color2gray = model_mode == "gray" and eval_set != "classic5"
        for qf in eval_qfs:
            dataset = EvaluationDataset(
                in_paths=refs,
                color_scale=model_mode,
                color2gray=color2gray,
                qf=qf,
            )
            loaders.append(((eval_set, qf), DataLoader(dataset, batch_size=1, shuffle=False)))
    return "single", loaders


def build_criterion(cfg: dict[str, Any], device: torch.device) -> nn.Module:
    loss_cfg = cfg["loss"]
    model_mode = cfg["model"]["mode"]
    name = str(loss_cfg["name"]).lower()

    if name == "auto":
        name = "dual-c" if model_mode == "color" else "dual-g"

    if name == "freq_wise-g":
        return GrayFreqEnhancedLoss(mode=loss_cfg["freq_mode"], device=device)
    if name == "freq_wise-c":
        return ColorFreqEnhancedLoss(
            mode=loss_cfg["freq_mode"],
            y_cbcr_ratio=loss_cfg["ycbcr_ratio"],
            device=device,
        )
    if name == "dual-g":
        return GrayDualDomainLoss(balance_ratio=float(loss_cfg["balance_ratio"]))
    if name == "dual-c":
        return ColorDualDomainLoss(balance_ratio=float(loss_cfg["balance_ratio"]))
    if name == "pixel-g":
        return GrayPixelDomainLoss(loss_type="l1", itrp_mode="bilinear")
    if name == "pixel-c":
        return ColorPixelDomainLoss(loss_type="l1", itrp_mode="bilinear")
    if name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss.name: {loss_cfg['name']}")


def build_optimizer(cfg: dict[str, Any], model: nn.Module) -> optim.Optimizer:
    opt_cfg = cfg["optimizer"]
    name = str(opt_cfg["name"]).lower()
    lr = float(opt_cfg["base_lr"])
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer.name: {opt_cfg['name']}")


def build_scheduler(cfg: dict[str, Any], optimizer: optim.Optimizer) -> Any:
    sch_cfg = cfg["scheduler"]
    train_cfg = cfg["train"]
    name = str(sch_cfg["name"]).lower()
    total_epochs = int(train_cfg["epochs"])

    if name == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=[int(x) for x in sch_cfg["multistep_milestones"]],
            gamma=float(sch_cfg["multistep_gamma"]),
        )
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=float(sch_cfg["cosine_eta_min"]),
        )
    if name == "warmup+cosine":
        warm_up_iter = max(1, int(total_epochs * float(sch_cfg["warmup_ratio"])))
        t_max = total_epochs
        cosine_span = max(1, t_max - warm_up_iter)
        lr_start = float(sch_cfg["warmup_start_lr"])
        lr_cos_max = float(sch_cfg["cosine_max_lr"])
        lr_cos_min = float(sch_cfg["cosine_min_lr"])
        base_lr = float(cfg["optimizer"]["base_lr"])

        def lambda_warmup_cos(cur_iter: int) -> float:
            if cur_iter < warm_up_iter:
                return (lr_start + (cur_iter / warm_up_iter) * (lr_cos_max - lr_start)) / base_lr
            return (
                lr_cos_min
                + 0.5
                * (lr_cos_max - lr_cos_min)
                * (1.0 + math.cos((cur_iter - warm_up_iter) / cosine_span * math.pi))
            ) / base_lr

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_warmup_cos)

    raise ValueError(f"Unsupported scheduler.name: {sch_cfg['name']}")


def load_weights_if_needed(
    cfg: dict[str, Any],
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> int:
    last_epoch = 0
    paths_cfg = cfg["paths"]
    resume_ckpt = str(paths_cfg.get("resume_ckpt_path", "")).strip()
    load_model_path = str(paths_cfg.get("load_model_path", "")).strip()

    if load_model_path:
        raw_obj = torch.load(load_model_path, map_location=device)
        load_state_dict_flexible(model, raw_obj)
        return last_epoch

    if not resume_ckpt:
        return last_epoch

    ckpt = load_checkpoint(resume_ckpt, device)
    load_state_dict_flexible(model, ckpt)
    if isinstance(ckpt, dict):
        last_epoch = int(ckpt.get("epoch", -1))
    if "optimizer_dict" not in ckpt or "scheduler_dict" not in ckpt:
        print(
            f"WARNING: resume_ckpt={resume_ckpt} has no optimizer/scheduler state. "
            "Model weights loaded only."
        )
        return last_epoch

    optimizer.load_state_dict(ckpt["optimizer_dict"])

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    if isinstance(scheduler, MultiStepLR):
        ckpt["scheduler_dict"]["milestones"] = scheduler.state_dict()["milestones"]
        scheduler.load_state_dict(ckpt["scheduler_dict"])
        scheduler.last_epoch = last_epoch

    return last_epoch


def to_float_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(psnr(pred.contiguous(), gt.contiguous()).mean().cpu().item())


def align_hw(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = min(a.shape[-2], b.shape[-2])
    w = min(a.shape[-1], b.shape[-1])
    return a[..., :h, :w], b[..., :h, :w]


def evaluate_color(
    model: nn.Module,
    eval_loaders: list[tuple[Any, DataLoader]],
    model_win_size: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    model_psnrs: list[float] = []
    jpeg_psnrs: list[float] = []
    with torch.no_grad():
        for _, loader in eval_loaders:
            compressed_list: list[float] = []
            recovered_list: list[float] = []
            for _, (qm, [in_y, in_cb, in_cr], y, img_gt, pad_mul16) in enumerate(loader):
                in_y, in_cb, in_cr = in_y.to(device), in_cb.to(device), in_cr.to(device)

                in_y_padded, pad = pad_to_rb(in_y, model_win_size)
                in_cb_padded, _ = pad_to_rb(in_cb, model_win_size // 2)
                in_cr_padded, _ = pad_to_rb(in_cr, model_win_size // 2)

                pred_y, pred_cb, pred_cr = model(in_y_padded, in_cb_padded, in_cr_padded)
                pred_y = unpad_rb(pred_y, pad)
                pred_cb = unpad_rb(pred_cb, pad)
                pred_cr = unpad_rb(pred_cr, pad)

                in_y = in_y.cpu()
                in_cb = in_cb.cpu()
                in_cr = in_cr.cpu()
                pred_y = pred_y.cpu()
                pred_cb = pred_cb.cpu()
                pred_cr = pred_cr.cpu()

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

                loss_img = unpad(to_rgb(ycbcr_img, data_range=1.0), pad_mul16)
                recovered_img = unpad(to_rgb(pred_ycbcr_img, data_range=1.0), pad_mul16)
                gt_img = img_gt

                loss_img, gt_img = align_hw(loss_img, gt_img)
                recovered_img, gt_img = align_hw(recovered_img, gt_img)

                compressed_list.append(to_float_psnr(loss_img, gt_img))
                recovered_list.append(to_float_psnr(recovered_img, gt_img))

            model_psnrs.append(float(np.mean(recovered_list)))
            jpeg_psnrs.append(float(np.mean(compressed_list)))
    return model_psnrs, jpeg_psnrs


def evaluate_single_gray(
    model: nn.Module,
    eval_loaders: list[tuple[Any, DataLoader]],
    model_win_size: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    model_psnrs: list[float] = []
    jpeg_psnrs: list[float] = []
    with torch.no_grad():
        for _, loader in eval_loaders:
            compressed_list: list[float] = []
            recovered_list: list[float] = []
            for _, (qm, x, y, img_gt, pad_mul16) in enumerate(loader):
                if not torch.is_tensor(img_gt):
                    img_gt = torch.as_tensor(img_gt)
                x, img_gt = x.to(device), img_gt.to(device)
                x_padded, pad = pad_to_rb(x, model_win_size)
                x_pred = model(x_padded)
                x_pred = unpad_rb(x_pred, pad)

                loss_img = unpad(reshape_image_from_frequencies(x.cpu()), pad_mul16)
                recovered_img = unpad(reshape_image_from_frequencies(x_pred.cpu()), pad_mul16)
                gt_img = img_gt.cpu()

                loss_img, gt_img = align_hw(loss_img, gt_img)
                recovered_img, gt_img = align_hw(recovered_img, gt_img)

                compressed_list.append(to_float_psnr(loss_img, gt_img))
                recovered_list.append(to_float_psnr(recovered_img, gt_img))

            model_psnrs.append(float(np.mean(recovered_list)))
            jpeg_psnrs.append(float(np.mean(compressed_list)))
    return model_psnrs, jpeg_psnrs


def evaluate_double_color(
    cfg: dict[str, Any],
    model: nn.Module,
    eval_loaders: list[tuple[Any, DataLoader]],
    model_win_size: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    model_psnrs: list[float] = []
    jpeg_psnrs: list[float] = []

    dcfg = cfg["eval"]["double"]
    save_images = bool(dcfg.get("save_images", False))
    out_dir = dcfg.get("image_out_dir", "./double_eval")

    with torch.no_grad():
        for pair, loader in eval_loaders:
            q1, q2, sh, sw = pair
            compressed_list: list[float] = []
            recovered_list: list[float] = []
            for _, (qm, [in_y, in_cb, in_cr], img_gt, filename) in enumerate(loader):
                in_y, in_cb, in_cr = in_y.to(device), in_cb.to(device), in_cr.to(device)

                in_y_padded, pad = pad_to_rb(in_y, model_win_size)
                in_cb_padded, _ = pad_to_rb(in_cb, model_win_size // 2)
                in_cr_padded, _ = pad_to_rb(in_cr, model_win_size // 2)

                pred_y, pred_cb, pred_cr = model(in_y_padded, in_cb_padded, in_cr_padded)
                pred_y = unpad_rb(pred_y, pad)
                pred_cb = unpad_rb(pred_cb, pad)
                pred_cr = unpad_rb(pred_cr, pad)

                in_y = in_y.cpu()
                in_cb = in_cb.cpu()
                in_cr = in_cr.cpu()
                pred_y = pred_y.cpu()
                pred_cb = pred_cb.cpu()
                pred_cr = pred_cr.cpu()

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

                loss_img = to_rgb(torch.cat([y_chn, cb_chn, cr_chn], dim=1), data_range=1.0)
                recovered_img = to_rgb(
                    torch.cat([pred_y_chn, pred_cb_chn, pred_cr_chn], dim=1),
                    data_range=1.0,
                )
                gt_img = img_gt[..., sh:, sw:]

                loss_img, gt_img = align_hw(loss_img, gt_img)
                recovered_img, gt_img = align_hw(recovered_img, gt_img)

                compressed_list.append(to_float_psnr(loss_img, gt_img))
                recovered_list.append(to_float_psnr(recovered_img, gt_img))

                if save_images:
                    tag = f"{q1}_{q2}_shift{sh}{sw}"
                    out_subdir = Path(out_dir) / tag
                    mkdir(str(out_subdir))
                    name = filename[0] if isinstance(filename, (list, tuple)) else str(filename)
                    out_img = recovered_img[0, [2, 1, 0], ...].permute(1, 2, 0).numpy() * 255.0
                    cv2.imwrite(str(out_subdir / f"{name}.png"), out_img)

            model_psnrs.append(float(np.mean(recovered_list)))
            jpeg_psnrs.append(float(np.mean(compressed_list)))
    return model_psnrs, jpeg_psnrs


def train_epoch(
    cfg: dict[str, Any],
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int,
    global_step: int,
    wandb_module=None,
) -> tuple[float, int, dict[str, float]]:
    train_cfg = cfg["train"]
    mode = cfg["model"]["mode"]
    grad_clip = float(cfg["optimizer"]["grad_clip"])
    max_train_iters = int(train_cfg.get("max_train_iters", 0))
    perf_log_interval = int(train_cfg.get("perf_log_interval", 50))
    wandb_cfg = cfg.get("wandb", {})
    log_interval = int(wandb_cfg.get("log_interval", 10))

    losses: list[float] = []
    recent_losses: deque[float] = deque(maxlen=10)
    recent_data_times: deque[float] = deque(maxlen=10)
    recent_iter_times: deque[float] = deque(maxlen=10)
    recent_samples: deque[int] = deque(maxlen=10)
    epoch_stats = {
        "data_time": 0.0,
        "transfer_time": 0.0,
        "forward_time": 0.0,
        "backward_time": 0.0,
        "step_time": 0.0,
        "iter_time": 0.0,
        "samples": 0.0,
        "iters": 0.0,
        "timed_iters": 0.0,
    }
    window_stats = {
        "data_time": 0.0,
        "iter_time": 0.0,
        "samples": 0.0,
        "iters": 0.0,
    }
    tqdm_bar = tqdm(loader, total=len(loader), mininterval=30, miniters=100)
    last_iter_end = time.perf_counter()
    for itr, batch in enumerate(tqdm_bar):
        iter_idx = itr + 1
        should_log_perf = perf_log_interval > 0 and (iter_idx % perf_log_interval == 0 or itr == 0)
        sync_stage_timing = should_log_perf and device.type == "cuda"

        data_time = time.perf_counter() - last_iter_end
        iter_start = time.perf_counter()
        transfer_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        step_time = 0.0

        if mode == "color":
            qm, [x_y, x_cb, x_cr], y = batch
            if y.dim() == 5:  # speedy loader
                x_y = x_y[0, ...]
                x_cb = x_cb[0, ...]
                x_cr = x_cr[0, ...]
                y = y[0, ...]

            if sync_stage_timing:
                torch.cuda.synchronize()
            transfer_start = time.perf_counter()
            x_y, x_cb, x_cr, y = x_y.to(device), x_cb.to(device), x_cr.to(device), y.to(device)
            if sync_stage_timing:
                torch.cuda.synchronize()
            transfer_time = time.perf_counter() - transfer_start

            if sync_stage_timing:
                torch.cuda.synchronize()
            forward_start = time.perf_counter()
            pred_y, pred_cb, pred_cr = model(x_y, x_cb, x_cr)
            loss = criterion(pred_y, pred_cb, pred_cr, y)
            if sync_stage_timing:
                torch.cuda.synchronize()
            forward_time = time.perf_counter() - forward_start
            batch_size = int(y.shape[0])
        else:
            qm, x, y = batch
            if y.dim() == 5:  # speedy loader
                x = x[0, ...]
                y = y[0, ...]

            if sync_stage_timing:
                torch.cuda.synchronize()
            transfer_start = time.perf_counter()
            x, y = x.to(device), y.to(device)
            if sync_stage_timing:
                torch.cuda.synchronize()
            transfer_time = time.perf_counter() - transfer_start

            if sync_stage_timing:
                torch.cuda.synchronize()
            forward_start = time.perf_counter()
            pred_y = model(x)
            loss = criterion(pred_y, y)
            if sync_stage_timing:
                torch.cuda.synchronize()
            forward_time = time.perf_counter() - forward_start
            batch_size = int(y.shape[0])

        optimizer.zero_grad()

        if sync_stage_timing:
            torch.cuda.synchronize()
        backward_start = time.perf_counter()
        loss.backward()
        if sync_stage_timing:
            torch.cuda.synchronize()
        backward_time = time.perf_counter() - backward_start

        if sync_stage_timing:
            torch.cuda.synchronize()
        step_start = time.perf_counter()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        if sync_stage_timing:
            torch.cuda.synchronize()
        step_time = time.perf_counter() - step_start

        iter_time = time.perf_counter() - iter_start
        global_step += 1
        loss_value = float(loss.item())
        losses.append(loss_value)
        recent_losses.append(loss_value)
        recent_data_times.append(data_time)
        recent_iter_times.append(iter_time)
        recent_samples.append(batch_size)

        epoch_stats["data_time"] += data_time
        epoch_stats["iter_time"] += iter_time
        epoch_stats["samples"] += batch_size
        epoch_stats["iters"] += 1
        if should_log_perf:
            epoch_stats["transfer_time"] += transfer_time
            epoch_stats["forward_time"] += forward_time
            epoch_stats["backward_time"] += backward_time
            epoch_stats["step_time"] += step_time
            epoch_stats["timed_iters"] += 1

        window_stats["data_time"] += data_time
        window_stats["iter_time"] += iter_time
        window_stats["samples"] += batch_size
        window_stats["iters"] += 1

        tqdm_bar.set_description(f"Ep: {epoch_num} | Itr: {iter_idx}")
        if itr % 10 == 0:
            recent_iter_sum = max(sum(recent_iter_times), 1e-12)
            recent_sps = sum(recent_samples) / recent_iter_sum
            tqdm_bar.set_postfix(
                loss=f"{np.mean(recent_losses):.5f}",
                data=f"{np.mean(recent_data_times):.3f}s",
                iter=f"{np.mean(recent_iter_times):.3f}s",
                sps=f"{recent_sps:.1f}",
            )

        if should_log_perf:
            window_iter_avg = window_stats["iter_time"] / max(window_stats["iters"], 1.0)
            window_data_avg = window_stats["data_time"] / max(window_stats["iters"], 1.0)
            window_data_ratio = window_data_avg / max(window_iter_avg, 1e-12)
            window_sps = window_stats["samples"] / max(window_stats["iter_time"], 1e-12)
            mem_gb = 0.0
            if device.type == "cuda":
                mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f"[perf] epoch={epoch_num} iter={iter_idx}/{len(loader)} "
                f"loss={loss_value:.5f} "
                f"data={window_data_avg:.3f}s ({window_data_ratio * 100:.1f}%) "
                f"h2d={transfer_time:.3f}s forward={forward_time:.3f}s "
                f"backward={backward_time:.3f}s step={step_time:.3f}s "
                f"iter={window_iter_avg:.3f}s sps={window_sps:.1f} "
                f"max_mem={mem_gb:.2f}G"
            )
            window_stats = {
                "data_time": 0.0,
                "iter_time": 0.0,
                "samples": 0.0,
                "iters": 0.0,
            }

        if wandb_module is not None and log_interval > 0 and ((itr + 1) % log_interval == 0 or itr == 0):
            metrics = {
                "train/epoch": int(epoch_num),
                "train/iter": int(iter_idx),
                "train/loss_iter": loss_value,
                "train/loss_iter_avg": float(np.mean(losses[-min(len(losses), log_interval):])),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "perf/data_time": float(data_time),
                "perf/iter_time": float(iter_time),
                "perf/samples_per_sec": float(batch_size / max(iter_time, 1e-12)),
                "perf/data_ratio": float(data_time / max(iter_time, 1e-12)),
            }
            if should_log_perf:
                metrics.update(
                    {
                        "perf/transfer_time": float(transfer_time),
                        "perf/forward_time": float(forward_time),
                        "perf/backward_time": float(backward_time),
                        "perf/step_time": float(step_time),
                    }
                )
            wandb_module.log(metrics, step=global_step)

        if max_train_iters > 0 and (itr + 1) >= max_train_iters:
            break

        last_iter_end = time.perf_counter()

    if epoch_stats["iters"] <= 0:
        return 0.0, global_step, {
            "avg_data_time": 0.0,
            "avg_transfer_time": 0.0,
            "avg_forward_time": 0.0,
            "avg_backward_time": 0.0,
            "avg_step_time": 0.0,
            "avg_iter_time": 0.0,
            "avg_samples_per_sec": 0.0,
            "data_ratio": 0.0,
        }

    avg_iter_time = epoch_stats["iter_time"] / epoch_stats["iters"]
    timed_iters = max(epoch_stats["timed_iters"], 1.0)
    perf_stats = {
        "avg_data_time": float(epoch_stats["data_time"] / epoch_stats["iters"]),
        "avg_transfer_time": float(epoch_stats["transfer_time"] / timed_iters),
        "avg_forward_time": float(epoch_stats["forward_time"] / timed_iters),
        "avg_backward_time": float(epoch_stats["backward_time"] / timed_iters),
        "avg_step_time": float(epoch_stats["step_time"] / timed_iters),
        "avg_iter_time": float(avg_iter_time),
        "avg_samples_per_sec": float(epoch_stats["samples"] / max(epoch_stats["iter_time"], 1e-12)),
        "data_ratio": float((epoch_stats["data_time"] / epoch_stats["iters"]) / max(avg_iter_time, 1e-12)),
    }
    return float(np.mean(losses)), global_step, perf_stats


def format_ckpt_name(mode: str, epoch: int, train_loss: float, eval_psnrs: list[float]) -> str:
    if not eval_psnrs:
        return f"{mode}_epoch{epoch}_tloss{train_loss:.4f}.ckpt"
    if len(eval_psnrs) == 1:
        return f"{mode}_epoch{epoch}_tloss{train_loss:.4f}_psnr{eval_psnrs[0]:.2f}.ckpt"
    return f"{mode}_epoch{epoch}_tloss{train_loss:.4f}_avgpsnr{float(np.mean(eval_psnrs)):.2f}.ckpt"


def print_brief_config(
    cfg: dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    last_epoch: int,
    eval_mode: str,
    eval_loaders: list[tuple[Any, DataLoader]],
    num_train_images: int,
    effective_config_path: Path,
) -> None:
    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    ckpt_cfg = cfg["checkpoint"]
    loss_cfg = cfg["loss"]
    opt_cfg = cfg["optimizer"]
    sch_cfg = cfg["scheduler"]
    eval_cfg = cfg["eval"]
    wandb_cfg = cfg["wandb"]
    total_params, trainable_params = count_parameters(model)

    print("\n----------------------------------")
    print(
        f"start training: {train_cfg['epochs']} eps * {len(train_loader)} iters (full) "
        f"or {train_cfg['sample_per_epoch']}(sampled)"
    )
    print("--- Runtime ---")
    print(
        f"device={runtime_cfg['device']}, data_parallel={runtime_cfg['use_data_parallel']}, "
        f"device_ids={runtime_cfg['device_ids']}, seed={runtime_cfg['seed']}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '[all]')}"
    )
    print("--- Model ---")
    print(
        f"type={model_cfg['type']}, mode={model_cfg['mode']}, "
        f"in_dim={model_cfg['input_dim']}, dim={model_cfg['dim']}, "
        f"groups={model_cfg['num_groups']}, blocks/group={model_cfg['num_blocks_in_group']}, "
        f"window_size={model.window_size}"
    )
    print(f"total_params={total_params}, trainable_params={trainable_params}")
    print("--- Train ---")
    print(
        f"double_jpeg={train_cfg['double_jpeg']}, train_patterns={len(data_cfg['train_patterns'])}, "
        f"total_images={num_train_images}, loader={train_cfg['loader_type']}, "
        f"bs={train_cfg['batch_size']}, patch={train_cfg['patch_size']}, workers={train_cfg['num_workers']}, "
        f"perf_log_interval={train_cfg['perf_log_interval']}"
    )
    print(
        f"qf_range={train_cfg['qf_range']}, custom_qf={train_cfg['custom_qf']}, "
        f"custom_shift={train_cfg['custom_shift']}, max_train_iters={train_cfg['max_train_iters']}"
    )
    print(
        f"dataloader: batch_size={train_loader.batch_size}, num_workers={train_loader.num_workers}, "
        f"pin_memory={train_loader.pin_memory}, persistent_workers={train_loader.persistent_workers}, "
        f"prefetch_factor={getattr(train_loader, 'prefetch_factor', None)}"
    )
    print("--- Data ---")
    for pattern in data_cfg["train_patterns"]:
        print(f"train_pattern: {pattern}")
    print("--- Loss / Optim / Sched ---")
    print(f"loss={loss_cfg['name']}, balance_ratio={loss_cfg['balance_ratio']}, freq_mode={loss_cfg['freq_mode']}")
    print(f"optimizer={opt_cfg['name']}, base_lr={opt_cfg['base_lr']}, grad_clip={opt_cfg['grad_clip']}")
    print(f"scheduler={sch_cfg['name']}")
    print("--- Checkpoints ---")
    print(
        f"save_top_k={ckpt_cfg['save_top_k']}, keep_last_k={ckpt_cfg['keep_last_k']}, "
        f"best_alias={BEST_CHECKPOINT_ALIAS}"
    )
    print("--- Eval ---")
    print(
        f"enabled={eval_cfg['enabled']}, mode={eval_mode}, loaders={len(eval_loaders)}, "
        f"every_n_epochs={eval_cfg['every_n_epochs']}"
    )
    if eval_mode == "double":
        print(f"double_qf_pairs={len(eval_cfg['double']['qf_pairs'])}, gt_pattern={eval_cfg['double']['gt_pattern']}")
    else:
        print(f"sets={resolve_single_eval_sets(eval_cfg)}, qfs={eval_cfg['qfs']}")
    print("--- W&B ---")
    if wandb_cfg["enabled"]:
        print(
            f"enabled=True, project={wandb_cfg['project']}, run_name={build_wandb_run_name(wandb_cfg)}, "
            f"group={wandb_cfg['group'] or '[none]'}, mode={wandb_cfg['mode']}, "
            f"dir={wandb_cfg['dir']}, log_interval={wandb_cfg['log_interval']}"
        )
    else:
        print(f"enabled=False, dir={wandb_cfg['dir']}")
    print("--- Paths ---")
    print(f"resume_from_epoch={last_epoch}, resume_ckpt={cfg['paths']['resume_ckpt_path']}")
    print(f"load_model_path={cfg['paths']['load_model_path']}")
    print(f"save_ckpt_path={cfg['paths']['save_ckpt_path']}")
    print(f"effective_config_path={effective_config_path}")
    print("--- Effective Config ---")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False).rstrip())


def run_training(cfg: dict[str, Any]) -> None:
    setup_seed(int(cfg["runtime"]["seed"]))
    device = parse_device(cfg["runtime"])
    print("DEVICE:", device)
    if torch.cuda.is_available():
        device_index = device.index if device.index is not None else 0
        torch.cuda.set_device(device_index)
        print("Use", torch.cuda.device_count(), "gpus")

    mkdir(cfg["paths"]["save_ckpt_path"])
    effective_config_path = dump_effective_config(cfg, cfg["paths"]["save_ckpt_path"])
    checkpoint_index = load_checkpoint_index(cfg["paths"]["save_ckpt_path"])
    save_checkpoint_index(cfg["paths"]["save_ckpt_path"], checkpoint_index)
    existing_best = next(
        (
            entry
            for entry in checkpoint_index["entries"]
            if entry["filename"] == checkpoint_index.get("best_filename", "")
        ),
        None,
    )
    update_best_checkpoint_markers(cfg["paths"]["save_ckpt_path"], existing_best)
    if checkpoint_index["entries"]:
        print(
            f"Existing checkpoints: {len(checkpoint_index['entries'])}, "
            f"best={checkpoint_index.get('best_filename', '') or 'n/a'}"
        )

    model = build_model(cfg, device)
    print_model_summary(cfg, model)
    model_win_size = model.window_size

    in_paths = resolve_train_paths(cfg)
    print(f"train_patterns: {len(cfg['data']['train_patterns'])}, total images: {len(in_paths)}")

    train_loader = build_train_dataloader(cfg, in_paths)
    eval_mode, eval_loaders = build_eval_loaders(cfg)
    if cfg["eval"]["enabled"]:
        print(f"eval mode: {eval_mode}, eval loaders: {len(eval_loaders)}")
        if not eval_loaders:
            print("WARNING: eval.enabled=true but no evaluation data was found.")

    criterion = build_criterion(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    last_epoch = load_weights_if_needed(cfg, model, optimizer, scheduler, device)
    print_brief_config(
        cfg,
        model,
        train_loader,
        last_epoch,
        eval_mode,
        eval_loaders,
        len(in_paths),
        effective_config_path,
    )
    wandb_module = init_wandb(cfg, model, effective_config_path, len(in_paths), len(train_loader), last_epoch)
    model = maybe_wrap_data_parallel(model, cfg["runtime"], device)
    if isinstance(model, torch.nn.DataParallel):
        print(f"Train mode: DataParallel on cuda devices {list(model.device_ids)}")
    elif device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"Train mode: single-GPU on cuda:{device_index} (visible GPUs: {torch.cuda.device_count()})")
    else:
        print("Train mode: CPU")

    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    global_step = max(0, last_epoch) * len(train_loader)
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        epoch_num = epoch + last_epoch

        sample_per_epoch = int(train_cfg.get("sample_per_epoch", 0))
        if sample_per_epoch > 0:
            sample_size = min(sample_per_epoch, len(in_paths))
            sample_paths = random.sample(in_paths, sample_size)
            loader = build_train_dataloader(cfg, sample_paths)
        else:
            loader = train_loader

        train_loss, global_step, perf_stats = train_epoch(
            cfg,
            model,
            loader,
            criterion,
            optimizer,
            device,
            epoch_num=epoch_num,
            global_step=global_step,
            wandb_module=wandb_module,
        )
        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        print(f"Epoch #{epoch_num} train_average_loss: {train_loss:.6f}, lr: {current_lr:.8f}")
        print(
            f"Epoch #{epoch_num} perf: "
            f"data={perf_stats['avg_data_time']:.3f}s ({perf_stats['data_ratio'] * 100:.1f}%), "
            f"h2d={perf_stats['avg_transfer_time']:.3f}s, "
            f"forward={perf_stats['avg_forward_time']:.3f}s, "
            f"backward={perf_stats['avg_backward_time']:.3f}s, "
            f"step={perf_stats['avg_step_time']:.3f}s, "
            f"iter={perf_stats['avg_iter_time']:.3f}s, "
            f"sps={perf_stats['avg_samples_per_sec']:.1f}"
        )

        should_eval = bool(eval_cfg["enabled"]) and epoch_num % int(eval_cfg["every_n_epochs"]) == 0
        eval_psnrs: list[float] = []
        jpeg_psnrs: list[float] = []

        if should_eval and eval_loaders:
            model.eval()
            with torch.no_grad():
                if eval_mode == "double":
                    eval_psnrs, jpeg_psnrs = evaluate_double_color(cfg, model, eval_loaders, model_win_size, device)
                    print(f"Epoch #{epoch_num} Double JPEG PSNR:")
                    for (pair, _), m_psnr, j_psnr in zip(eval_loaders, eval_psnrs, jpeg_psnrs):
                        print(
                            f"qfs_shift:{pair} model_psnr:{m_psnr:.4f} "
                            f"jpeg_psnr:{j_psnr:.4f} increase:{(m_psnr - j_psnr):.4f}"
                        )
                elif cfg["model"]["mode"] == "color":
                    eval_psnrs, jpeg_psnrs = evaluate_color(model, eval_loaders, model_win_size, device)
                    print(f"Epoch #{epoch_num} Single JPEG (color) PSNR:")
                    current_set = None
                    for ((eval_set, qf), _), m_psnr, j_psnr in zip(eval_loaders, eval_psnrs, jpeg_psnrs):
                        if eval_set != current_set:
                            current_set = eval_set
                            print(f"  dataset: {eval_set}")
                        print(
                            f"    qf:{qf} model_psnr:{m_psnr:.4f} "
                            f"jpeg_psnr:{j_psnr:.4f} increase:{(m_psnr - j_psnr):.4f}"
                        )
                elif cfg["model"]["mode"] == "gray":
                    eval_psnrs, jpeg_psnrs = evaluate_single_gray(model, eval_loaders, model_win_size, device)
                    print(f"Epoch #{epoch_num} Single JPEG (gray) PSNR:")
                    current_set = None
                    for ((eval_set, qf), _), m_psnr, j_psnr in zip(eval_loaders, eval_psnrs, jpeg_psnrs):
                        if eval_set != current_set:
                            current_set = eval_set
                            print(f"  dataset: {eval_set}")
                        print(
                            f"    qf:{qf} model_psnr:{m_psnr:.4f} "
                            f"jpeg_psnr:{j_psnr:.4f} increase:{(m_psnr - j_psnr):.4f}"
                        )
                else:
                    raise ValueError(f"Unsupported eval mode: {eval_mode}")

            ckpt_name = format_ckpt_name(cfg["model"]["mode"], epoch_num, train_loss, eval_psnrs)
            save_checkpoint(
                os.path.join(cfg["paths"]["save_ckpt_path"], ckpt_name),
                epoch_num,
                model,
                optimizer,
                scheduler,
            )
            apply_checkpoint_retention(
                cfg,
                os.path.join(cfg["paths"]["save_ckpt_path"], ckpt_name),
                epoch_num,
                train_loss,
                eval_psnrs,
            )
        elif not eval_cfg["enabled"] and bool(train_cfg.get("save_every_epoch_when_no_eval", True)):
            ckpt_name = format_ckpt_name(cfg["model"]["mode"], epoch_num, train_loss, [])
            save_checkpoint(
                os.path.join(cfg["paths"]["save_ckpt_path"], ckpt_name),
                epoch_num,
                model,
                optimizer,
                scheduler,
            )
            apply_checkpoint_retention(
                cfg,
                os.path.join(cfg["paths"]["save_ckpt_path"], ckpt_name),
                epoch_num,
                train_loss,
                [],
            )

        if wandb_module is not None:
            epoch_metrics = {
                "train/epoch": int(epoch_num),
                "train/loss_epoch": float(train_loss),
                "train/lr": float(current_lr),
                "perf_epoch/data_time": float(perf_stats["avg_data_time"]),
                "perf_epoch/data_ratio": float(perf_stats["data_ratio"]),
                "perf_epoch/transfer_time": float(perf_stats["avg_transfer_time"]),
                "perf_epoch/forward_time": float(perf_stats["avg_forward_time"]),
                "perf_epoch/backward_time": float(perf_stats["avg_backward_time"]),
                "perf_epoch/step_time": float(perf_stats["avg_step_time"]),
                "perf_epoch/iter_time": float(perf_stats["avg_iter_time"]),
                "perf_epoch/samples_per_sec": float(perf_stats["avg_samples_per_sec"]),
            }
            epoch_metrics.update(build_eval_log_metrics(cfg, eval_mode, eval_loaders, eval_psnrs, jpeg_psnrs))
            wandb_module.log(epoch_metrics, step=global_step)

    gc.collect()
    if wandb_module is not None:
        wandb_module.finish()


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DCTransformer training entry")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.color.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    cfg = load_yaml_config(args.config, DEFAULT_CONFIG)
    cfg = normalize_runtime_after_gpu_binding(cfg)
    run_training(cfg)


if __name__ == "__main__":
    main()
