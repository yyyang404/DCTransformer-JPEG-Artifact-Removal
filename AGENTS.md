# Repository Guidelines

## Project Structure & Module Organization
Core training and evaluation entrypoints live at the repository root: `train.py` and `eval.py`. Model code is under `models/` (`dctransformer.py`, `base_modules.py`). Shared data, metrics, and runtime helpers are flat modules such as `dataset.py`, `runtime_utils.py`, `image_ops.py`, `pixel_metrics.py`, and `frequency_metrics.py`. Runtime presets are in `configs/` with names like `train.color.yaml` and `eval.quickstart.gray.yaml`. Vendored dependencies live in `third_party/torchjpeg/`; treat that directory as external code unless you are intentionally updating the vendor copy.

## Build, Test, and Development Commands
Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run training or evaluation with a config:

```bash
python train.py --config configs/train.color.yaml
python eval.py --config configs/eval.quickstart.color.yaml
```

`eval.py` accepts either training checkpoints (`*.ckpt`) or plain model weights (`*.pt`) as long as `paths.ckpt_path` points to a compatible file.

Use quickstart eval configs for smoke tests after model or metric changes. Before committing, run a syntax check such as `python -m py_compile train.py eval.py runtime_utils.py models/*.py`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and explicit type hints where practical. Keep modules focused and reuse `runtime_utils.py` for config loading, device parsing, and checkpoint compatibility instead of duplicating helpers. Match existing config naming patterns: `train.<mode>.yaml`, `eval.<mode>.yaml`, `eval.quickstart.<mode>.yaml`.

## Testing Guidelines
There is no checked-in `pytest` or coverage gate yet. Validate changes with targeted smoke tests:
- `python eval.py --config configs/eval.quickstart.color.yaml`
- `python eval.py --config configs/eval.quickstart.gray.yaml`
- a short training run by lowering `train.max_train_iters` or `train.limit_train_images` in a copied config

If you add automated tests, place them under a new `tests/` directory and keep fixtures small.

## Commit & Pull Request Guidelines
Recent history favors short imperative subjects such as `Refactor runtime utils...`, `Add quickstart pt eval workflow...`, and `Vendor torchjpeg...`. Follow that pattern and avoid vague messages like `sync`. Pull requests should state the problem, list changed configs or dataset assumptions, include validation commands, and attach before/after PSNR, SSIM, or sample outputs when restoration behavior changes.

## Configuration & Data Tips
Many YAML files contain machine-specific dataset and checkpoint paths. Update `paths.*`, training glob patterns, and eval dataset locations before running locally. Do not hardcode new absolute paths in source files; keep them in config.
