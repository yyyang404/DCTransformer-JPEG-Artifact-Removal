#!/usr/bin/env bash
set -euo pipefail

# usage:
#   bash download_datasets.sh [output_root]
# example:
#   bash download_datasets.sh ./dataset

OUT_ROOT="${1:-./dataset}"
mkdir -p "${OUT_ROOT}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: Hugging Face CLI 'hf' is not installed."
  echo "Install it with: pip install huggingface_hub"
  exit 1
fi

# optional: login first if needed
# hf auth login

ds_repos=(
  "yangtao9009/Flickr2K"
  "yangtao9009/DIV2K"
)

md_repos=(
  "ofsoundof/LSDIR"
)

failed=()

for repo in "${ds_repos[@]}"; do
  name="${repo##*/}"
  target="${OUT_ROOT}/${name}"
  mkdir -p "${target}"
  echo "Downloading ${repo} -> ${target}"

  if ! hf download "${repo}" \
    --repo-type dataset \
    --local-dir "${target}" \
    --max-workers 8; then
    echo "Failed to download ${repo}"
    echo "If this repo is private, gated, or renamed, run: hf auth login"
    failed+=("${repo}")
  fi
done

for repo in "${md_repos[@]}"; do
  name="${repo##*/}"
  target="${OUT_ROOT}/${name}"
  mkdir -p "${target}"
  echo "Downloading ${repo} -> ${target}"

  if ! hf download "${repo}" \
    --local-dir "${target}" \
    --max-workers 8; then
    echo "Failed to download ${repo}"
    echo "If this repo is private, gated, or renamed, run: hf auth login"
    failed+=("${repo}")
  fi
done

if ((${#failed[@]} > 0)); then
  echo
  echo "Completed with failures:"
  printf '  - %s\n' "${failed[@]}"
  exit 1
fi

echo "All done."
