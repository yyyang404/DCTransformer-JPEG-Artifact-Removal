# DCTransformer for JPEG Artifact Removal

Official implementation of [DCTransformer](https://arxiv.org/abs/2409.14364) for JPEG artifact removal.

## Codebase Overview

- `train.py`: unified training entry, including YAML config loading, checkpoint rotation, best-checkpoint aliasing, optional W&B logging, and periodic validation.
- `eval.py`: unified evaluation entry for both single-JPEG and double-JPEG settings.
- `dctransformer/data/`: dataset definitions and JPEG/DCT image processing helpers.
- `dctransformer/models/`: the DCTransformer model, base modules, and training losses.
- `dctransformer/utils/`: runtime/config helpers, metrics, dataset download script, and model-weight export script.
- `configs/`: checked-in release presets. All tracked paths are repository-relative by default.

## Environment Setup

```bash
conda create -y -n dct python=3.10 pip
conda activate dct
pip install --upgrade pip
pip install -r requirements.txt
```

## Path Policy

The checked-in YAML configs no longer contain machine-specific absolute paths. By default they assume you run commands from the repository root and keep local artifacts under:

```text
./dataset/
./local_save/
./test_results/
```

If your datasets or checkpoints live elsewhere, update only the relevant YAML path fields such as `data.train_patterns`, `paths.ckpt_path`, `paths.resume_ckpt_path`, or `eval.double.jpeg_root`.

## Training Data Setup

Download training data with:

```bash
bash dctransformer/utils/download_datasets.sh ./dataset
```

If you downloaded archives under `./dataset`, extract them into the folder names used by the training configs:

```bash
unzip -q dataset/DIV2K/DIV2K_train.zip -d dataset/DIV2K
unzip -q dataset/Flickr2K/Flickr2K.zip -d dataset/Flickr2K
```

Extract the LSDIR HQ training shards into a recursive folder layout that `glob(..., recursive=True)` can read directly:

```bash
mkdir -p dataset/LSDIR/LSDIR_hd
for f in dataset/LSDIR/shard-*.tar.gz; do
  tar -xzf "$f" -C dataset/LSDIR/LSDIR_hd
done
```

Expected training layout:

```text
dataset/DIV2K/DIV2K_train_HR/*.png
dataset/Flickr2K/Flickr2K_HR/*.png
dataset/LSDIR/LSDIR_hd/**/*.png
```

Double-JPEG configs expect pre-generated inputs under a repository-relative root such as:

```text
dataset/double_jpeg/shift00_jpg/10_90/*.jpg
dataset/double_jpeg/shift44_jpg/95_75/*.jpg
```

## Train

Stage-1 training:

```bash
python train.py --config configs/train.color.yaml
python train.py --config configs/train.gray.yaml
python train.py --config configs/train.double_jpeg.yaml
```

Stage-2 fine-tuning presets resume from the corresponding stage-1 `best.ckpt` by default:

```bash
python train.py --config configs/train.color_s2.yaml
python train.py --config configs/train.gray_s2.yaml
```

Training configs write checkpoints and runtime metadata to `./local_save/...`. `wandb.dir` defaults to `./local_save/wandb`.

## Eval

```bash
python eval.py --config configs/eval.color.yaml
python eval.py --config configs/eval.gray.yaml
python eval.py --config configs/eval.double_jpeg.yaml
```

The full eval presets expect:

- `configs/eval.color.yaml`: `./local_save/color/best.ckpt`
- `configs/eval.gray.yaml`: `./local_save/gray_s2/best.ckpt`
- `configs/eval.double_jpeg.yaml`: `./local_save/color_doublejpeg/best.ckpt`

If you only want a smoke test, copy one of these configs and reduce `eval.sets`, `eval.qfs`, or both.

## Export Publishable Weights

To convert a training checkpoint into a plain `state_dict` file:

```bash
python dctransformer/utils/export_model_weights.py \
  --input ./local_save/color/best.ckpt \
  --output ./weights/dctransformer_color.pt
```

`eval.py` accepts either a training checkpoint (`*.ckpt`) or a plain weights file (`*.pt`) as long as the file contains a supported checkpoint/state-dict format.
