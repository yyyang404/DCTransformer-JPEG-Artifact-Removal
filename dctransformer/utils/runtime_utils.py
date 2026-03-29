from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import yaml


def deep_update(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_config(path: str, default_cfg: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = deepcopy(default_cfg)
    with open(config_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError(f"YAML root must be mapping, got: {type(user_cfg)}")
    return deep_update(cfg, user_cfg)


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_device(runtime_cfg: Mapping[str, Any]) -> torch.device:
    name = runtime_cfg.get("device", "auto")
    if name == "auto":
        if torch.cuda.is_available():
            device_ids = [int(x) for x in runtime_cfg.get("device_ids", [])]
            if device_ids:
                return torch.device(f"cuda:{device_ids[0]}")
            return torch.device("cuda:0")
        return torch.device("cpu")

    name = str(name)
    if name == "cuda":
        device_ids = [int(x) for x in runtime_cfg.get("device_ids", [])]
        if device_ids:
            return torch.device(f"cuda:{device_ids[0]}")
        return torch.device("cuda:0")
    return torch.device(name)


def maybe_wrap_data_parallel(
    model: nn.Module,
    runtime_cfg: Mapping[str, Any],
    device: torch.device,
) -> nn.Module:
    if device.type != "cuda":
        return model
    if not bool(runtime_cfg.get("use_data_parallel", True)):
        return model
    device_ids = runtime_cfg.get("device_ids", [])
    if not device_ids or len(device_ids) <= 1:
        return model
    return torch.nn.DataParallel(model, device_ids=list(device_ids))


def _looks_like_state_dict(d: Mapping[str, Any]) -> bool:
    if not d:
        return False
    if not all(isinstance(k, str) for k in d.keys()):
        return False
    if any(k in d for k in ("epoch", "optimizer_dict", "scheduler_dict")):
        return False
    return all(torch.is_tensor(v) for v in d.values())


def _unwrap_model_state_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping) and "model_dict" in obj:
        return dict(obj["model_dict"])
    if isinstance(obj, Mapping) and "state_dict" in obj:
        return dict(obj["state_dict"])
    if isinstance(obj, Mapping) and _looks_like_state_dict(obj):
        return dict(obj)
    raise ValueError(
        "Unsupported checkpoint format. Expected dict with 'model_dict'/'state_dict' "
        "or a plain state_dict dict."
    )


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def extract_model_state_dict(
    state_dict_or_ckpt: Any,
    *,
    strip_module_prefix: bool = True,
) -> dict[str, Any]:
    state_dict = _unwrap_model_state_dict(state_dict_or_ckpt)
    if strip_module_prefix and state_dict and all(str(k).startswith("module.") for k in state_dict.keys()):
        return _strip_module_prefix(state_dict)
    return dict(state_dict)


def load_state_dict_flexible(model: nn.Module, state_dict_or_ckpt: Any) -> None:
    state_dict = extract_model_state_dict(state_dict_or_ckpt, strip_module_prefix=True)
    try:
        model.load_state_dict(dict(state_dict))
    except RuntimeError:
        if isinstance(state_dict, Mapping) and all(str(k).startswith("module.") for k in state_dict.keys()):
            model.load_state_dict(_strip_module_prefix(state_dict))
        else:
            raise
