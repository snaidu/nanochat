"""
Train JAX/Flax model. From root directory of the project, run as:

python -m scripts.jax.base_train

For multi-device training (uses all available devices automatically):

python -m scripts.jax.base_train --multi-device

Example for quick testing on CPU:
python -m scripts.jax.base_train --depth=4 --max-seq-len=512 --device-batch-size=2 --num-iterations=100 --eval-every=50

NOTE: Requires dataset and tokenizer to be prepared first. Run:
  python -m nanochat.dataset -n 10  # download at least a few shards
  python -m scripts.tok_train       # train the tokenizer
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from nanochat.jax.gpt import GPT, GPTJaxConfig
from nanochat.common import print_banner, get_base_dir

print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain JAX base model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Model architecture
parser.add_argument("--depth", type=int, default=12, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=64, help="target head dimension for attention")
parser.add_argument("--max-seq-len", type=int, default=1024, help="max context length")
# Training
parser.add_argument("--num-iterations", type=int, default=1000, help="number of optimization steps")
parser.add_argument("--device-batch-size", type=int, default=8, help="per-device batch size")
parser.add_argument("--learning-rate", type=float, default=3e-4, help="peak learning rate")
parser.add_argument("--warmup-steps", type=int, default=100, help="number of warmup steps")
parser.add_argument("--weight-decay", type=float, default=0.1, help="weight decay")
# Evaluation
parser.add_argument("--eval-every", type=int, default=100, help="evaluate val loss every N steps (-1 = disable)")
parser.add_argument("--eval-steps", type=int, default=10, help="number of batches to evaluate")
# Multi-device
parser.add_argument("--multi-device", action="store_true", help="enable multi-device data parallelism")
# Misc
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="compute dtype")
# Output
parser.add_argument("--output-dir", type=str, default=None, help="directory to save model and logs (default: jax_checkpoints/<run>)")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# JAX setup

print(f"JAX devices: {jax.devices()}")
num_devices = len(jax.devices())
print(f"Number of devices: {num_devices}")

if args.multi_device and num_devices > 1:
    print(f"Multi-device training enabled with {num_devices} devices")
    # Batch size must be divisible by number of devices for pmap
    assert args.device_batch_size % num_devices == 0, \
        f"device_batch_size ({args.device_batch_size}) must be divisible by num_devices ({num_devices})"
    per_device_batch = args.device_batch_size // num_devices
else:
    args.multi_device = False
    per_device_batch = args.device_batch_size

dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

# -----------------------------------------------------------------------------
# Tokenizer and vocab

# Check if data is prepared
base_dir = get_base_dir()
tokenizer_path = os.path.join(base_dir, "tokenizer", "tokenizer.pkl")
if not os.path.exists(tokenizer_path):
    print("ERROR: Tokenizer not found. Please prepare the data and tokenizer first:")
    print("  python -m nanochat.dataset -n 10  # download at least a few shards")
    print("  python -m scripts.tok_train       # train the tokenizer")
    sys.exit(1)

from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Model configuration

num_layers = args.depth
base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
num_kv_heads = num_heads
head_dim = model_dim // num_heads

print(f"Model config:")
print(f"  num_layers: {num_layers}")
print(f"  model_dim: {model_dim}")
print(f"  num_heads: {num_heads}")
print(f"  head_dim: {head_dim}")

config = GPTJaxConfig(
    sequence_len=args.max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    dtype=dtype,
)

# -----------------------------------------------------------------------------
# Initialize model

rngs = nnx.Rngs(args.seed)
model = GPT(config, rngs=rngs)

# Count parameters
def count_params(model):
    params = nnx.state(model, nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))

num_params = count_params(model)
print(f"Number of parameters: {num_params:,}")

# -----------------------------------------------------------------------------
# Output directory setup

if args.output_dir is None:
    output_dir = os.path.join(base_dir, "jax_checkpoints", args.run)
else:
    output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# -----------------------------------------------------------------------------
# Optimizer with learning rate schedule

def create_lr_schedule(warmup_steps, total_steps, peak_lr):
    """Linear warmup then linear decay to 0."""
    def schedule(step):
        # Warmup
        warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
        # Linear decay after warmup
        decay_steps = total_steps - warmup_steps
        decay_factor = jnp.maximum(1.0 - (step - warmup_steps) / decay_steps, 0.0)
        # Combine: warmup then decay
        return peak_lr * jnp.where(step < warmup_steps, warmup_factor, decay_factor)
    return schedule

lr_schedule = create_lr_schedule(args.warmup_steps, args.num_iterations, args.learning_rate)

# AdamW optimizer
tx = optax.adamw(
    learning_rate=lr_schedule,
    b1=0.9,
    b2=0.95,
    weight_decay=args.weight_decay,
)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

# -----------------------------------------------------------------------------
# Training step

def loss_fn(model: GPT, x: jax.Array, y: jax.Array):
    """Compute cross-entropy loss."""
    return model(x, y)

@nnx.jit
def train_step(model: GPT, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array):
    """Single training step."""
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, x, y)
    optimizer.update(model, grads)  # Flax 0.11+ requires (model, grads)
    return loss

# Multi-device training with data parallelism
if args.multi_device and num_devices > 1:
    from jax.sharding import Mesh, PartitionSpec, NamedSharding

    # Create a mesh over all devices
    mesh = Mesh(jax.devices(), axis_names=('data',))
    data_sharding = NamedSharding(mesh, PartitionSpec('data'))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    def shard_batch(x):
        """Shard batch across devices on the first axis."""
        return jax.device_put(x, data_sharding)

    @nnx.jit
    def train_step_sharded(model: GPT, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array):
        """Training step with gradient averaging across devices."""
        def loss_fn_sharded(model, x, y):
            loss = model(x, y)
            # Average loss across devices
            return jax.lax.pmean(loss, axis_name='data')

        grad_fn = nnx.value_and_grad(loss_fn_sharded)
        with mesh:
            loss, grads = grad_fn(model, x, y)
            # Average gradients across devices
            grads = jax.lax.pmean(grads, axis_name='data')
        optimizer.update(model, grads)  # Flax 0.11+ requires (model, grads)
        return loss

    print(f"Using mesh: {mesh} for data parallel training")

# -----------------------------------------------------------------------------
# Evaluation step

@nnx.jit
def eval_step(model: GPT, x: jax.Array, y: jax.Array):
    """Compute validation loss."""
    return model(x, y)

# -----------------------------------------------------------------------------
# Data loading

def torch_to_jax(tensor):
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(tensor.cpu().numpy())

# Create data loaders (they return PyTorch tensors, we convert to JAX)
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len,
    split="train", device="cpu"  # Load to CPU, then convert to JAX
)

def get_val_loader():
    return tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len,
        split="val", device="cpu"
    )

def get_batch(loader):
    """Get next batch from loader, convert to JAX arrays."""
    x, y, state = next(loader)
    x, y = torch_to_jax(x), torch_to_jax(y)
    return x, y, state

# -----------------------------------------------------------------------------
# Training loop

print(f"\nStarting training for {args.num_iterations} iterations...")
print(f"Batch size: {args.device_batch_size} x {args.max_seq_len} = {args.device_batch_size * args.max_seq_len:,} tokens/step")

model.train()
metrics = nnx.metrics.Average('loss')
best_val_loss = float('inf')
total_tokens = 0
start_time = time.time()

# History tracking for logging
train_loss_history = []  # (step, loss)
val_loss_history = []    # (step, loss)
step_times = []          # time per step (excluding first few warmup steps)

for step in range(args.num_iterations + 1):
    # Evaluation
    if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.num_iterations):
        model.eval()
        val_loader = get_val_loader()
        val_losses = []
        for _ in range(args.eval_steps):
            x, y, _ = get_batch(val_loader)
            val_loss = eval_step(model, x, y)
            val_losses.append(float(val_loss))
        avg_val_loss = sum(val_losses) / len(val_losses)
        val_loss_history.append({"step": int(step), "loss": float(avg_val_loss)})
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        print(f"Step {step:5d} | Val loss: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")
        model.train()

    # Stop after final evaluation
    if step == args.num_iterations:
        break

    # Training step
    t0 = time.time()
    x, y, dataloader_state = get_batch(train_loader)

    if args.multi_device and num_devices > 1:
        # Shard batch across devices
        x, y = shard_batch(x), shard_batch(y)
        loss = train_step_sharded(model, optimizer, x, y)
    else:
        loss = train_step(model, optimizer, x, y)
    loss = float(loss)

    t1 = time.time()
    dt = t1 - t0

    # Track step time (skip first 10 steps for JIT warmup)
    if step >= 10:
        step_times.append(dt)

    # Track training loss
    train_loss_history.append({"step": int(step), "loss": float(loss)})

    # Update metrics
    total_tokens += args.device_batch_size * args.max_seq_len
    tokens_per_sec = args.device_batch_size * args.max_seq_len / dt

    # Logging
    if step % 10 == 0:
        elapsed = time.time() - start_time
        lr = float(lr_schedule(step))
        print(f"Step {step:5d} | Loss: {loss:.4f} | LR: {lr:.2e} | "
              f"Tokens/s: {tokens_per_sec:,.0f} | "
              f"Time: {elapsed:.1f}s | Epoch: {dataloader_state['epoch']}")

# Final stats
elapsed = time.time() - start_time
avg_step_time = sum(step_times) / len(step_times) if step_times else 0

print(f"\nTraining complete!")
print(f"Total time: {elapsed/60:.2f} minutes")
print(f"Average step time: {avg_step_time*1000:.2f} ms")
print(f"Total tokens: {total_tokens:,}")
print(f"Best validation loss: {best_val_loss:.4f}")

# -----------------------------------------------------------------------------
# Save model and logs

# Save model state
model_path = os.path.join(output_dir, "model.ckpt")
print(f"Saving model to {model_path}...")
state = nnx.state(model)
with open(model_path, "wb") as f:
    f.write(nnx.serialization.to_bytes(state))
print(f"Model saved.")

# Save training log as JSON
log_data = {
    "config": {
        "run": args.run,
        "depth": args.depth,
        "aspect_ratio": args.aspect_ratio,
        "head_dim": args.head_dim,
        "max_seq_len": args.max_seq_len,
        "num_iterations": args.num_iterations,
        "device_batch_size": args.device_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "dtype": args.dtype,
        "multi_device": args.multi_device,
    },
    "model": {
        "num_layers": num_layers,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "num_params": num_params,
    },
    "training": {
        "num_devices": num_devices,
        "tokens_per_step": args.device_batch_size * args.max_seq_len,
        "total_tokens": total_tokens,
        "total_steps": args.num_iterations,
        "final_epoch": dataloader_state["epoch"],
    },
    "timing": {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "total_time_seconds": elapsed,
        "total_time_minutes": elapsed / 60,
        "avg_step_time_ms": avg_step_time * 1000,
        "tokens_per_second": total_tokens / elapsed if elapsed > 0 else 0,
    },
    "results": {
        "best_val_loss": float(best_val_loss) if best_val_loss != float('inf') else None,
        "final_train_loss": float(train_loss_history[-1]["loss"]) if train_loss_history else None,
        "final_val_loss": float(val_loss_history[-1]["loss"]) if val_loss_history else None,
    },
    "history": {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
    },
}

log_path = os.path.join(output_dir, "training_log.json")
with open(log_path, "w") as f:
    json.dump(log_data, f, indent=2)
print(f"Training log saved to {log_path}")
