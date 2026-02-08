"""
GPT variant with learnable positional kernel attention.

Instead of RoPE (which mixes positional information into Q and K independently),
this model learns a kernel P(delta) that directly modulates the QK^T dot product:

    S[i,j] = sum_k Q[i,k] * P[delta'(i-j), k] * K[j,k]

where delta' is a warped relative position:

    delta' = s_max * tanh(delta / sigma)

This warping gives fine-grained resolution for nearby tokens and compresses
distant tokens into a bounded range, enabling length generalization.

P is initialized with a RoPE-like cos/sin structure (cos on the first half of
head dimensions, sin on the second half) so the model starts from a well-known
baseline and can learn to deviate.

Based on: https://github.com/snaidu/nanochat (nanochat/jax/gpt.py)
"""

from dataclasses import dataclass
import logging

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx
import optax

from nanochat.jax.gpt import GPTJaxConfig
from nanochat.jax.poskernel_attention import poskernel_flash_attention

Array = jax.Array


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTPosKernelConfig(GPTJaxConfig):
    """Configuration for the positional kernel GPT variant.

    Inherits all standard GPT fields (vocab_size, sequence_len, n_layer,
    n_head, n_kv_head, n_embd, dtype, head_dim) from GPTJaxConfig.
    """
    # Positional kernel parameters
    s_max: int = 128          # half-extent of the P lookup table
    base_freq: float = 10000.0  # RoPE-style base frequency for init
    init_sigma: float = 256.0   # initial value of per-head warp scale


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def norm(x: Array, axis: int = -1, eps: float = 1e-6) -> Array:
    """nGPT-style norm: scale to unit norm times sqrt(dim)."""
    return x * (lax.sqrt(float(x.shape[axis])) * lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps))


# ---------------------------------------------------------------------------
# Positional kernel initialization
# ---------------------------------------------------------------------------

def init_positional_kernel(
    n_heads: int,
    head_dim: int,
    s_max: int,
    base_freq: float = 10000.0,
) -> jnp.ndarray:
    """Initialize P table with RoPE-like cos/sin structure.

    First half of head dimensions get cos(omega_k * delta),
    second half get sin(omega_k * delta). This mirrors the RoPE
    pairing and ensures the model starts with both even and odd
    symmetry components — necessary to distinguish positive from
    negative relative positions at each frequency.

    Returns: P of shape [n_heads, 2*s_max+1, head_dim]
    """
    table_size = 2 * s_max + 1
    deltas = jnp.arange(table_size) - s_max  # [-s_max, ..., s_max]

    d_half = head_dim // 2
    freqs = 1.0 / (base_freq ** (jnp.arange(d_half, dtype=jnp.float32) / d_half))

    phases = deltas[:, None] * freqs[None, :]  # [table_size, d_half]

    P_single = jnp.concatenate([jnp.cos(phases), jnp.sin(phases)], axis=-1)  # [table_size, head_dim]

    # All heads start identical; training differentiates them
    return jnp.tile(P_single[None, :, :], (n_heads, 1, 1))


# ---------------------------------------------------------------------------
# Positional kernel attention
# ---------------------------------------------------------------------------

class PositionalKernelParam(nnx.Param):
    """Marker type for the P table and sigma parameters."""
    pass


class PositionalKernelAttention(nnx.Module):
    """Causal self-attention with learnable positional kernel.

    Instead of applying rotary embeddings to Q and K, this module learns
    a kernel P[delta', d] that modulates each head-dimension's contribution
    to the QK^T dot product based on warped relative position delta'.
    """

    def __init__(self, config: GPTPosKernelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.s_max = config.s_max

        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        # Standard Q, K, V projections
        self.c_q = nnx.Linear(self.n_embd, self.n_head * self.head_dim, use_bias=False,
                              param_dtype=config.dtype, rngs=rngs)
        self.c_k = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, use_bias=False,
                              param_dtype=config.dtype, rngs=rngs)
        self.c_v = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, use_bias=False,
                              param_dtype=config.dtype, rngs=rngs)
        self.c_proj = nnx.Linear(self.n_embd, self.n_embd, use_bias=False,
                                 param_dtype=config.dtype, rngs=rngs)

        # Positional kernel: P table and per-head warp scale sigma
        P_init = init_positional_kernel(
            self.n_head, self.head_dim, config.s_max, config.base_freq,
        )
        self.P = PositionalKernelParam(P_init.astype(config.dtype))

        # Per-head sigma controlling tanh warp steepness
        # Initialize so nearby positions are ~linear (sigma >> 1)
        self.sigma = PositionalKernelParam(
            jnp.full((self.n_head,), config.init_sigma, dtype=config.dtype)
        )

    def _compute_warped_kernel(self, T: int) -> Array:
        """Compute interpolated P values for all relative positions in [0, T-1].

        Returns: P_interp of shape [n_head, T, T, head_dim]

        NOTE: This materializes the full T x T kernel — fine for prototyping
        but should be replaced with a blocked Pallas kernel for production.
        """
        # Raw relative positions: delta[i,j] = i - j
        positions = jnp.arange(T)
        delta = positions[:, None] - positions[None, :]  # [T, T]

        # Warp per head: delta_warped[h, i, j] = s_max * tanh(delta / sigma_h)
        sigma = jnp.abs(self.sigma) + 1e-6  # ensure positive
        delta_warped = self.s_max * jnp.tanh(
            delta[None, :, :] / sigma[:, None, None]
        )  # [H, T, T]

        # Shift to table coordinates: [0, 2*s_max]
        delta_table = delta_warped + self.s_max  # [H, T, T]

        # Soft lookup via linear interpolation
        idx_lo = jnp.floor(delta_table).astype(jnp.int32)
        idx_hi = idx_lo + 1
        P = self.P[...]
        frac = delta_table - idx_lo.astype(P.dtype)

        table_max = 2 * self.s_max
        idx_lo = jnp.clip(idx_lo, 0, table_max)
        idx_hi = jnp.clip(idx_hi, 0, table_max)

        # Gather: P[h, idx, d] for all (h, i, j)
        head_idx = jnp.arange(self.n_head)[:, None, None]  # [H, 1, 1]
        P_lo = P[head_idx, idx_lo, :]  # [H, T, T, D]
        P_hi = P[head_idx, idx_hi, :]  # [H, T, T, D]

        return (1 - frac[..., None]) * P_lo + frac[..., None] * P_hi  # [H, T, T, D]

    def __call__(self, x: Array, **_kwargs):
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        q = self.c_q(x).reshape(B, T, H, D)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, D)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, D)

        # nGPT-style normalization on Q and K (no rotary!)
        q, k = norm(q), norm(k)

        # Expand KV heads if grouped query attention
        if self.n_kv_head < self.n_head:
            repeats = self.n_head // self.n_kv_head
            k = jnp.repeat(k, repeats, axis=2)
            v = jnp.repeat(v, repeats, axis=2)

        # Rearrange to [B, H, T, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        P_table = self.P[...]   # [H, table_size, D]
        sigma = self.sigma[...] # [H]
        sm_scale = 1.0 / jnp.sqrt(float(D))

        if jax.devices()[0].platform == 'tpu':
            # Fused tiled kernel: avoids materializing [H, T, T, D] P_interp
            y = poskernel_flash_attention(
                q, k, v, P_table, sigma,
                self.s_max, True, sm_scale, None,
            )
        else:
            # Naive fallback for CPU/GPU (materializes full P_interp)
            P_interp = self._compute_warped_kernel(T)

            S = jnp.einsum('bhid,hijd,bhjd->bhij', q, P_interp, k)
            S = S * sm_scale

            causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
            S = jnp.where(causal_mask[None, None, :, :], S, jnp.finfo(S.dtype).min)

            A = jax.nn.softmax(S, axis=-1)
            y = A @ v  # [B, H, T, D]

        y = y.transpose(0, 2, 1, 3).reshape(B, T, -1)  # [B, T, C]
        return self.c_proj(y)


# ---------------------------------------------------------------------------
# MLP (unchanged from base model)
# ---------------------------------------------------------------------------

class MLP(nnx.Module):
    def __init__(self, config: GPTPosKernelConfig, *, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, use_bias=False,
                               param_dtype=config.dtype, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, use_bias=False,
                                 param_dtype=config.dtype, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = self.c_fc(x)
        x = jnp.square(nnx.relu(x))
        return self.c_proj(x)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(nnx.Module):
    def __init__(self, config: GPTPosKernelConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.attn = PositionalKernelAttention(config, layer_idx, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, x: Array, **kwargs) -> Array:
        x = x + self.attn(norm(x), **kwargs)
        x = x + self.mlp(norm(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GPTPosKernel(nnx.Module):
    """GPT with learnable positional kernel attention.

    Drop-in replacement for the RoPE-based GPT. The key difference:
    no rotary embeddings — positional information is encoded entirely
    through the learned kernel P that modulates QK^T.
    """

    def __init__(self, config: GPTPosKernelConfig, pad_vocab_size_to: int = 64, *, rngs: nnx.Rngs):
        self.config = config
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            logging.info(f'Padded vocabulary to {padded_vocab_size}')

        self.wte = nnx.Embed(padded_vocab_size, config.n_embd, param_dtype=config.dtype, rngs=rngs)
        self.h = nnx.List(Block(config, idx, rngs=rngs) for idx in range(config.n_layer))
        self.lm_head = nnx.Linear(config.n_embd, padded_vocab_size, use_bias=False,
                                  param_dtype=config.dtype, rngs=rngs)
        self.resid_lambdas = nnx.Param(jnp.ones(config.n_layer, dtype=config.dtype))
        self.x0_lambdas = nnx.Param(jnp.zeros(config.n_layer, dtype=config.dtype))

    def __call__(
        self,
        idx: Array,
        targets: Array | None = None,
        loss_reduction: str = 'mean',
    ):
        B, T = idx.shape

        x = self.wte(idx)
        x = norm(x)
        x0 = x

        for i, block in enumerate(self.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x)

        x = norm(x)
        softcap = 15.0
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = jnp.astype(logits, jnp.float32)
        logits = softcap * lax.tanh(logits / softcap)

        if targets is not None:
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                where=targets.reshape(-1) > -1,
            )
            if loss_reduction == 'mean':
                loss = loss.mean()
            elif loss_reduction == 'sum':
                loss = loss.sum()
            else:
                raise ValueError(f'{loss_reduction=} is not supported.')
            return loss
        else:
            return logits

    def param_count(self) -> dict[str, int]:
        """Count parameters by category."""
        counts: dict[str, int] = {'total': 0, 'positional_kernel': 0, 'other': 0}
        graph_def, state = nnx.split(self)
        flat = state.flat_state()
        for path_tuple, val in flat:
            n = val[...].size
            counts['total'] += n
            path = '.'.join(str(p) for p in path_tuple)
            if 'P' in path or 'sigma' in path:
                counts['positional_kernel'] += n
            else:
                counts['other'] += n
        return counts


# ---------------------------------------------------------------------------
# Smoke test / demo
# ---------------------------------------------------------------------------

def _smoke_test():
    """Quick forward + backward pass to verify everything works."""
    config = GPTPosKernelConfig(
        vocab_size=256,
        sequence_len=128,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        s_max=64,
        init_sigma=64.0,
    )
    model = GPTPosKernel(config, rngs=nnx.Rngs(0))
    counts = model.param_count()
    print(f"Parameter counts: {counts}")

    # Dummy data
    key = jax.random.PRNGKey(42)
    idx = jax.random.randint(key, (2, 64), 0, config.vocab_size)
    targets = jax.random.randint(key, (2, 64), 0, config.vocab_size)

    # Forward
    loss = model(idx, targets=targets)
    print(f"Forward pass loss: {loss:.4f}")

    # Backward
    @nnx.jit
    def train_step(model, idx, targets):
        def loss_fn(model):
            return model(idx, targets=targets)
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        return loss, grads

    loss, grads = train_step(model, idx, targets)
    print(f"JIT'd forward+backward loss: {loss:.4f}")

    # Check that P and sigma got gradients
    grad_def, grad_state = nnx.split(grads)
    flat_grads = grad_state.flat_state()
    for path_tuple, val in flat_grads:
        path = '.'.join(str(p) for p in path_tuple)
        if 'P' in path or 'sigma' in path:
            g = val[...]
            print(f"  grad {path}: shape={g.shape}, norm={jnp.linalg.norm(g):.6f}")

    # Test length generalization: forward on longer sequence than training config
    longer_idx = jax.random.randint(key, (1, 256), 0, config.vocab_size)
    logits = model(longer_idx)
    print(f"Length generalization: input T=256 (config T={config.sequence_len}), output shape={logits.shape}")

    print("\nSmoke test passed!")


if __name__ == '__main__':
    _smoke_test()