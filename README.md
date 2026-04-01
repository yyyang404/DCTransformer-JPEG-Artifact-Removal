# DCT JPEG Artifact Removal

Official **training and evaluation** implementation of [JPEG Quantized Coefficient Recovery via DCT Domain Spatial-Frequential Transformer (DCT)
](https://arxiv.org/abs/2409.14364) for JPEG artifact removal.


## Codebase Overview

- `train.py`: unified training entry, including YAML config loading, checkpoint rotation, best-checkpoint aliasing, optional W&B logging, and periodic validation.
- `eval.py`: unified evaluation entry for both single-JPEG and double-JPEG settings.
- `dctransformer/models/`: the DCTransformer model, base modules, and training losses.
- `dctransformer/data/`: dataset definitions and JPEG/DCT image processing helpers.
- `configs/`: checked-in release presets. All tracked paths are repository-relative by default.

##  Setup


**Environment:**

```bash
conda create -y -n dct python=3.10 pip
conda activate dct
pip install --upgrade pip
pip install -r requirements.txt
```

**Model Weights:** [[Huggingface] DCT-JPEG-Artifact-Removal](https://huggingface.co/yyyang/DCT-JPEG-Artifact-Removal)

**Data/Model Paths:**

The YAML configs assume you run commands from the repository root and keep local artifacts under `./dataset/`

If your datasets or checkpoints live elsewhere, update only the relevant YAML path fields such as `data.train_patterns`, `paths.ckpt_path`, `paths.resume_ckpt_path`.

## Eval

```bash
python eval.py --config configs/eval.color.yaml
python eval.py --config configs/eval.gray.yaml
```

The full eval presets expect:

- `configs/eval.color.yaml`: `./local_save/color/best.ckpt`
- `configs/eval.gray.yaml`: `./local_save/gray/best.ckpt`

## Train

###  Data Setup
Download training data with:

```bash
bash dctransformer/utils/download_datasets.sh ./dataset
```

Expected training layout:

```text
dataset/DIV2K/DIV2K_train_HR/*.png
dataset/Flickr2K/Flickr2K_HR/*.png
dataset/LSDIR/LSDIR_hd/**/*.png
```

### Training 
```bash
python train.py --config configs/train.color.yaml
python train.py --config configs/train.gray.yaml
```

Training configs write checkpoints and runtime metadata to `./local_save/...`. And `wandb.dir` defaults to `./local_save/wandb`.

## BibTeX
If you find our work helpful, please kindly consider using the following reference.

```
@article{ouyang2024jpeg,
  title={JPEG quantized coefficient recovery via DCT domain spatial-frequential transformer},
  author={Ouyang, Mingyu and Chen, Zhenzhong},
  journal={IEEE Transactions on Image Processing},
  volume={33},
  pages={3385--3398},
  year={2024},
  publisher={IEEE}
}
```
