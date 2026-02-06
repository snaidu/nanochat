#!/bin/bash

# JAX/Flax training on Google Cloud TPU
# This script sets up the environment and runs training on TPU.

# Run as:
# bash runs/jax/runtpu.sh

set -e  # Exit on error

echo "=== Nanochat JAX TPU Training Setup ==="

# Environment setup
export NANOCHAT_BASE_DIR=/mnt/disks/training-data/nanochat

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install dependencies with TPU support
echo "Installing dependencies..."
uv sync --extra tpu
source .venv/bin/activate
pip install tpu-info

echo "=== Starting Training ==="

# Run training with multi-device support
# Adjust parameters based on your TPU configuration:
# - TPU v2-8: 8 cores, 64GB total HBM
# - TPU v3-8: 8 cores, 128GB total HBM
# - TPU v4-8: 8 cores, 256GB total HBM
python scripts/jax/base_train.py \
    --run=dummy \
    --depth=8 \
    --aspect-ratio=64 \
    --head-dim=32 \
    --num-iteration=10000 \
    --batch-size 16 \
    --learning-rate=3e-4 \
    --weight-decay=0.1 \
    --warmup-steps=100 \
    --output-dir=./jax_checkpoints/d8_ar64_hd32 \
    --multi-device

echo "=== Training Complete ==="
