from dataclasses import dataclass, field
import logging
import sys

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx
import optax
from nanochat.gpt import GPTConfig

Array = jax.Array

@dataclass
class GPTJaxConfig(GPTConfig):
    dtype: jnp.dtype = field(default=jnp.float32)
    base: float = 10000.0

    @property
    def head_dim(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        return self.n_embd // self.n_head

def apply_rotary_emb(x: Array, cos: Array, sin: Array):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 =    cos * x1 + sin * x2
    y2 = (-sin) * x1 + cos * x2
    return jnp.concat((y1, y2), axis=3)

def norm(x: Array, axis: int = -1, eps=1e-6) -> Array:
    return x * (lax.sqrt(float(x.shape[axis])) * lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps))
    
class RotaryEmbedding(nnx.Variable):
    pass
    
def precompute_cos_sin(config: GPTJaxConfig, len_multiplier: int = 10) -> RotaryEmbedding:
    head_dim = config.head_dim
    channel_range = jnp.arange(0, head_dim, 2, dtype=config.dtype)
    inv_freq = config.base ** ((-channel_range) / head_dim)
    t = jnp.arange(config.sequence_len * len_multiplier, dtype=config.dtype)
    freqs = jnp.outer(t, inv_freq)
    cos, sin = jnp.cos(freqs), jnp.sin(freqs)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return RotaryEmbedding((cos, sin))

class KVCache(nnx.Module):
    def __init__(self, config: GPTJaxConfig):
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        cache_shape = (1, config.sequence_len, self.n_head, self.head_dim)
        self.cached_key = nnx.Cache(jnp.zeros(cache_shape, dtype=config.dtype))
        self.cached_value = nnx.Cache(jnp.zeros(cache_shape, dtype=config.dtype))
        # This could actually be passed but will need to revise caching.
        self.cache_index = nnx.Cache(jnp.array(0, dtype=jnp.int32))

class CausalSelfAttention(nnx.Module):
    def __init__(self, config: GPTJaxConfig, layer_idx, *, rngs: nnx.Rngs):
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nnx.Linear(self.n_embd, self.n_head * self.head_dim, use_bias=False, param_dtype=config.dtype, rngs=rngs)
        self.c_k = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, use_bias=False, param_dtype=config.dtype, rngs=rngs)
        self.c_v = nnx.Linear(self.n_embd, self.n_kv_head * self.head_dim, use_bias=False, param_dtype=config.dtype, rngs=rngs)
        self.c_proj = nnx.Linear(self.n_embd, self.n_embd, use_bias=False, param_dtype=config.dtype, rngs=rngs)

    # Skipping value embedding for now
        # self.ve_gate_channels = 32
        # self.ve_gate = nn.Linear(self.self.ve_gate_channels, self.n_kv_head, bias=False) if has_vs(layer_idx, config.n_layer) else None

    def __call__(self, x: Array, cos_sin: (Array, Array), window_size: int | None = None, kv_cache: KVCache | None = None):
        # For now disable window size.
        assert window_size is None
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        if kv_cache is None:
            y: Array = nnx.dot_product_attention(q, k, v, deterministic=True, is_causal=True)
        else:
            raise NotImplemented

        y = self.c_proj(y.reshape(B, T, -1))
        return y

class MLP(nnx.Module):
    def __init__(self, config: GPTJaxConfig, *, rngs: nnx.Rngs):
        self.c_fc = nnx.Linear(config.n_embd, 4 * config.n_embd, use_bias=False, param_dtype=config.dtype, rngs=rngs)
        self.c_proj = nnx.Linear(4 * config.n_embd, config.n_embd, use_bias=False, param_dtype=config.dtype, rngs=rngs)

    def __call__(self, x: Array):
        x = self.c_fc(x)
        x = jnp.square(nnx.relu(x))
        x = self.c_proj(x)
        return x

class Block(nnx.Module):
    def __init__(self, config: GPTJaxConfig, layer_idx: int, *, rngs: nnx.Rngs):
        self.attn = CausalSelfAttention(config, layer_idx, rngs=rngs)
        self.mlp = MLP(config, rngs=rngs)
        self.n_embd = config.n_embd

    def __call__(self, x, cos_sin: (Array, Array), window_size: int | None = None, kv_cache: KVCache | None = None):
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x

class GPT(nnx.Module):
    def __init__(self, config: GPTJaxConfig, pad_vocab_size_to=64, *, rngs: nnx.Rngs):
        self.config = config
        self.window_sizes = [None] * config.n_layer  # TODO: window size support
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            logging.info(f'Padded vocabulary to {padded_vocab_size}')
        self.wte = nnx.Embed(padded_vocab_size, config.n_embd, param_dtype=config.dtype, rngs=rngs)
        self.h = nnx.List(Block(config, idx, rngs=rngs) for idx in range(config.n_layer))
        self.lm_head = nnx.Linear(config.n_embd, padded_vocab_size, use_bias=False, param_dtype=config.dtype, rngs=rngs)
        self.resid_lambdas = nnx.Param(jnp.ones(config.n_layer, dtype=config.dtype))
        self.x0_lambdas = nnx.Param(jnp.zeros(config.n_layer, dtype=config.dtype))
        # Leaving out bigram and value embeddings.
        self.cos_sin = precompute_cos_sin(config)

    def __call__(self, idx: Array, targets: Array | None = None, kv_cache: list[KVCache] | None = None, loss_reduction='mean'):
        B, T = idx.shape
        assert kv_cache is None
        T0 = 0
        cos_sin = self.cos_sin[0][:, T0:T0+T], self.cos_sin[1][:, T0:T0+T]
        x = self.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin)
        x = norm(x)
        softcap = 15.0
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = jnp.astype(logits, jnp.float32)
        logits = softcap * lax.tanh(logits / softcap)
        if targets is not None:
            targets = targets.reshape(-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                where=targets > -1)
            if loss_reduction == 'mean':
                loss = loss.mean()
            elif loss_reduction == 'sum':
                loss = loss.sum()
            else:
                raise NotImplemented(f'{loss_reduction=} is not supported.')
            return loss
        else:
            return logits
