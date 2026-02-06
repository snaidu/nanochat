"""
Train JAX/Flax model. From root directory of the project, run as:

python -m scripts.jax.base_train

Uses all available devices automatically via mesh-based sharding.
For single device, the mesh has shape (1,) so sharding is a no-op.

Example for quick testing on CPU:
python -m scripts.jax.base_train --depth=4 --max-seq-len=512 --batch-size=2 --num-iterations=100 --eval-every=50

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
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from nanochat.jax.gpt import GPT, GPTJaxConfig
from nanochat.jax.gpt_poskernel import GPTPosKernel, GPTPosKernelConfig
from nanochat.common import print_banner, get_base_dir


# -----------------------------------------------------------------------------
# Helpers

def count_params(model):
    params = nnx.state(model, nnx.Param)
    return sum(p.size for p in jax.tree.leaves(params))


def torch_to_jax(tensor):
    """Convert PyTorch tensor to JAX array."""
    return jnp.array(tensor.cpu().numpy())


def create_lr_schedule(warmup_steps, total_steps, peak_lr):
    """Linear warmup then linear decay to 0."""
    def schedule(step):
        warmup_factor = jnp.minimum(step / warmup_steps, 1.0)
        decay_steps = total_steps - warmup_steps
        decay_factor = jnp.maximum(1.0 - (step - warmup_steps) / decay_steps, 0.0)
        return peak_lr * jnp.where(step < warmup_steps, warmup_factor, decay_factor)
    return schedule


# -----------------------------------------------------------------------------
# Setup functions

def parse_args():
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
    parser.add_argument("--batch-size", type=int, default=8, help="batch size (split across devices automatically)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="number of warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="weight decay")
    # Evaluation
    parser.add_argument("--eval-every", type=int, default=100, help="evaluate val loss every N steps (-1 = disable)")
    parser.add_argument("--eval-steps", type=int, default=10, help="number of batches to evaluate")
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="compute dtype")
    # Model selection
    parser.add_argument("--model", type=str, default="gpt", choices=["gpt", "gpt_poskernel"],
                        help="model architecture to train")
    # Output
    parser.add_argument("--output-dir", type=str, default=None, help="directory to save model and logs (default: jax_checkpoints/<run>)")
    return parser.parse_args()


def create_mesh(args):
    """Create a Mesh over all devices. Single device gets mesh shape (1,)."""
    devices = jax.devices()
    num_devices = len(devices)
    print(f"JAX devices: {devices}")
    print(f"Number of devices: {num_devices}")

    if num_devices > 1:
        assert args.batch_size % num_devices == 0, \
            f"batch_size ({args.batch_size}) must be divisible by num_devices ({num_devices})"
        print(f"Multi-device training with {num_devices} devices")

    return Mesh(devices, axis_names=('data',))


def create_model(args, vocab_size):
    """Create GPT model and config. Returns (model, config, model_info)."""
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    num_layers = args.depth
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    num_kv_heads = num_heads
    head_dim = model_dim // num_heads

    print(f"Model: {args.model}")
    print(f"Model config:")
    print(f"  num_layers: {num_layers}")
    print(f"  model_dim: {model_dim}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {head_dim}")

    rngs = nnx.Rngs(args.seed)

    if args.model == "gpt_poskernel":
        config = GPTPosKernelConfig(
            sequence_len=args.max_seq_len,
            vocab_size=vocab_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_kv_head=num_kv_heads,
            n_embd=model_dim,
            dtype=dtype,
        )
        model = GPTPosKernel(config, rngs=rngs)
    else:
        config = GPTJaxConfig(
            sequence_len=args.max_seq_len,
            vocab_size=vocab_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_kv_head=num_kv_heads,
            n_embd=model_dim,
            dtype=dtype,
        )
        model = GPT(config, rngs=rngs)

    num_params = count_params(model)
    print(f"Number of parameters: {num_params:,}")

    model_info = {
        "model": args.model,
        "num_layers": num_layers,
        "model_dim": model_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "num_params": num_params,
    }

    return model, config, model_info


def create_optimizer(model, args):
    """Create LR schedule and AdamW optimizer. Returns (optimizer, schedule_fn)."""
    lr_schedule = create_lr_schedule(args.warmup_steps, args.num_iterations, args.learning_rate)
    tx = optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=0.95,
        weight_decay=args.weight_decay,
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return optimizer, lr_schedule


def evaluate(model, get_val_loader_fn, eval_steps, get_batch_fn):
    """Run eval loop, return average val loss."""
    model.eval()
    val_loader = get_val_loader_fn()
    val_losses = []
    for _ in range(eval_steps):
        x, y, _ = get_batch_fn(val_loader)
        val_loss = model(x, y)
        val_losses.append(float(val_loss))
    model.train()
    return sum(val_losses) / len(val_losses)


def save_checkpoint(model, output_dir, log_data):
    """Save model state and JSON training log."""
    model_path = os.path.join(output_dir, "model.ckpt")
    print(f"Saving model to {model_path}...")
    state = nnx.state(model)
    with open(model_path, "wb") as f:
        f.write(nnx.serialization.to_bytes(state))
    print(f"Model saved.")

    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved to {log_path}")


# -----------------------------------------------------------------------------
# Training

def train_model(args, mesh):
    """All setup + training inside the mesh/sharding context."""
    from nanochat.tokenizer import get_tokenizer
    from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

    # Check if data is prepared
    base_dir = get_base_dir()
    tokenizer_path = os.path.join(base_dir, "tokenizer", "tokenizer.pkl")
    if not os.path.exists(tokenizer_path):
        print("ERROR: Tokenizer not found. Please prepare the data and tokenizer first:")
        print("  python -m nanochat.dataset -n 10  # download at least a few shards")
        print("  python -m scripts.tok_train       # train the tokenizer")
        sys.exit(1)

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    # Model
    model, config, model_info = create_model(args, vocab_size)

    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(base_dir, "jax_checkpoints", args.run)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Optimizer
    optimizer, lr_schedule = create_optimizer(model, args)

    # Sharding for data batches
    data_sharding = NamedSharding(mesh, P('data'))

    # Training step — single implementation, mesh handles sharding
    @nnx.jit
    def train_step(model: GPT, optimizer: nnx.Optimizer, x: jax.Array, y: jax.Array):
        def loss_fn(model, x, y):
            return model(x, y)
        grad_fn = nnx.value_and_grad(loss_fn)
        loss, grads = grad_fn(model, x, y)
        optimizer.update(model, grads)
        return loss

    # Data loading
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.batch_size, args.max_seq_len,
        split="train", device="cpu",
    )

    def get_val_loader():
        return tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer, args.batch_size, args.max_seq_len,
            split="val", device="cpu",
        )

    def get_batch(loader):
        x, y, state = next(loader)
        x, y = torch_to_jax(x), torch_to_jax(y)
        return x, y, state

    # Training loop
    num_devices = len(jax.devices())
    print(f"\nStarting training for {args.num_iterations} iterations...")
    print(f"Batch size: {args.batch_size} x {args.max_seq_len} = {args.batch_size * args.max_seq_len:,} tokens/step")

    model.train()
    best_val_loss = float('inf')
    total_tokens = 0
    start_time = time.time()

    train_loss_history = []
    val_loss_history = []
    step_times = []

    for step in range(args.num_iterations + 1):
        # Evaluation
        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.num_iterations):
            avg_val_loss = evaluate(model, get_val_loader, args.eval_steps, get_batch)
            val_loss_history.append({"step": int(step), "loss": float(avg_val_loss)})
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            print(f"Step {step:5d} | Val loss: {avg_val_loss:.4f} | Best: {best_val_loss:.4f}")

        if step == args.num_iterations:
            break

        # Training step
        t0 = time.time()
        x, y, dataloader_state = get_batch(train_loader)
        x, y = jax.device_put((x, y), data_sharding)
        loss = train_step(model, optimizer, x, y)
        loss = float(loss)
        t1 = time.time()
        dt = t1 - t0

        if step >= 10:
            step_times.append(dt)

        train_loss_history.append({"step": int(step), "loss": float(loss)})

        total_tokens += args.batch_size * args.max_seq_len
        tokens_per_sec = args.batch_size * args.max_seq_len / dt

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

    # Save
    log_data = {
        "config": {
            "model": args.model,
            "run": args.run,
            "depth": args.depth,
            "aspect_ratio": args.aspect_ratio,
            "head_dim": args.head_dim,
            "max_seq_len": args.max_seq_len,
            "num_iterations": args.num_iterations,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "dtype": args.dtype,
        },
        "model": model_info,
        "training": {
            "num_devices": num_devices,
            "tokens_per_step": args.batch_size * args.max_seq_len,
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

    save_checkpoint(model, output_dir, log_data)


# -----------------------------------------------------------------------------
# Entry point

def main():
    print_banner()
    args = parse_args()
    mesh = create_mesh(args)

    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
        train_model(args, mesh)


if __name__ == '__main__':
    main()
