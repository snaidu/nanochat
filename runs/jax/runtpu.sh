#!/bin/bash

# JAX/Flax training on Google Cloud TPU
# This script sets up the environment and runs training on TPU.

# Run as:
# bash runs/jax/runtpu.sh

set -e  # Exit on error

echo "=== Nanochat JAX TPU Training Setup ==="

# Environment setup
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install dependencies with TPU support
echo "Installing dependencies..."
uv python install 3.13
uv sync --extra tpu

# Verify TPU access
echo "Verifying TPU access..."
uv run python -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
print(f'Number of devices: {len(devices)}')
if devices:
    print(f'Device type: {devices[0].platform}')
    if devices[0].platform != 'tpu':
        print('WARNING: TPU not detected!')
        exit(1)
"

# Download dataset (10 shards for quick start, increase for full training)
echo "Downloading dataset..."
uv run python -m nanochat.dataset -n 10

# Train tokenizer
echo "Training tokenizer..."
uv run python -m scripts.tok_train

# Verify setup with a quick model test
echo "Testing model..."
uv run python -c "
import jax
import jax.numpy as jnp
from flax import nnx
from nanochat.jax.gpt import GPT, GPTJaxConfig

config = GPTJaxConfig(
    sequence_len=128, vocab_size=50304, n_layer=2, n_head=4,
    n_kv_head=4, n_embd=128, dtype=jnp.bfloat16,
)
model = GPT(config, rngs=nnx.Rngs(0))
x = jax.random.randint(jax.random.key(0), (2, 64), 0, 1000)
y = jax.random.randint(jax.random.key(1), (2, 64), 0, 1000)
loss = model(x, y)
print(f'Model test passed! Loss: {loss}')
"

echo "=== Starting Training ==="

# Run training with multi-device support
# Adjust parameters based on your TPU configuration:
# - TPU v2-8: 8 cores, 64GB total HBM
# - TPU v3-8: 8 cores, 128GB total HBM
# - TPU v4-8: 8 cores, 256GB total HBM
uv run python -m scripts.jax.base_train \
    --depth=12 \
    --max-seq-len=1024 \
    --device-batch-size=32 \
    --num-iterations=5000 \
    --eval-every=100 \
    --eval-steps=20 \
    --warmup-steps=200 \
    --learning-rate=3e-4 \
    --weight-decay=0.1 \
    --multi-device

echo "=== Training Complete ==="
