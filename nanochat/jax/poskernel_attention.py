"""Positional-kernel attention with custom_vjp.

Forward: Chunked flash-attention-style online softmax. P_interp is computed
on-the-fly per tile from the small P_table + sigma — no [H, T, T, D]
materialization.

Backward: Two-pass chunked approach (dQ pass + dKV pass) so only small tile
accumulators live in the inner loop carry. Recomputes P_interp per tile.

Residuals saved: (q, k, v, P_table, sigma, o, l, m) — no T×T tensors.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array

# ---------------------------------------------------------------------------
# Tile-level P_interp (vectorized over all heads, no vmap)
# ---------------------------------------------------------------------------

def _tile_P_interp(q_pos, k_pos, P_table, sigma_abs, s_max):
    """Compute P_interp for all heads for one (q_block, k_block) tile.

    All operations are vectorized over H — no vmap or dynamic head indexing.

    Args:
        q_pos: [block_q] int positions
        k_pos: [block_k] int positions
        P_table: [H, table_size, D]
        sigma_abs: [H] (pre-computed |sigma| + eps)
        s_max: int

    Returns:
        P_interp: [H, block_q, block_k, D]
        idx_lo, idx_hi: [H, block_q, block_k] int32
        frac: [H, block_q, block_k] float32
    """
    H = P_table.shape[0]
    delta = (q_pos[:, None] - k_pos[None, :]).astype(jnp.float32)  # [bq, bk]
    delta_warped = s_max * jnp.tanh(
        delta[None, :, :] / sigma_abs[:, None, None]
    )  # [H, bq, bk]
    delta_table = delta_warped + s_max

    idx_lo = jnp.floor(delta_table).astype(jnp.int32)
    idx_hi = idx_lo + 1
    frac = delta_table - idx_lo.astype(jnp.float32)

    table_max = 2 * s_max
    idx_lo = jnp.clip(idx_lo, 0, table_max)
    idx_hi = jnp.clip(idx_hi, 0, table_max)

    head_idx = jnp.arange(H)[:, None, None]  # [H, 1, 1]
    P_lo = P_table[head_idx, idx_lo, :]  # [H, bq, bk, D]
    P_hi = P_table[head_idx, idx_hi, :]  # [H, bq, bk, D]
    P_interp = (1.0 - frac[..., None]) * P_lo + frac[..., None] * P_hi

    return P_interp, idx_lo, idx_hi, frac


# ---------------------------------------------------------------------------
# Forward: chunked online softmax (no T×T materialization)
# ---------------------------------------------------------------------------

def _forward_chunked(q, k, v, P_table, sigma, s_max, causal,
                     block_q, block_k):
    """Flash-attention-style forward. Peak memory is O(block_q × block_k × D)."""
    B, H, T, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)
    n_q_blocks = T // block_q
    n_k_blocks = T // block_k

    o = jnp.zeros((B, H, T, D), dtype=jnp.float32)
    m = jnp.full((B, H, T), -1e30, dtype=jnp.float32)
    l = jnp.zeros((B, H, T), dtype=jnp.float32)

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    sigma_abs = jnp.abs(sigma) + 1e-6

    def outer_body(qi, carry):
        o, m, l = carry
        q_start = qi * block_q
        q_pos = jnp.arange(block_q) + q_start
        q_block = lax.dynamic_slice(q_f32, (0, 0, q_start, 0), (B, H, block_q, D))

        o_qi = jnp.zeros((B, H, block_q, D), dtype=jnp.float32)
        m_qi = jnp.full((B, H, block_q), -1e30, dtype=jnp.float32)
        l_qi = jnp.zeros((B, H, block_q), dtype=jnp.float32)

        def inner_body(kvi, inner_carry):
            o_qi, m_qi, l_qi = inner_carry
            k_start = kvi * block_k
            k_pos = jnp.arange(block_k) + k_start
            k_block = lax.dynamic_slice(k_f32, (0, 0, k_start, 0), (B, H, block_k, D))
            v_block = lax.dynamic_slice(v_f32, (0, 0, k_start, 0), (B, H, block_k, D))

            P_interp, _, _, _ = _tile_P_interp(q_pos, k_pos, P_table, sigma_abs, s_max)

            # 3-input einsum: avoids materializing [B, H, bq, bk, D]
            S = jnp.einsum('bhid,hijd,bhjd->bhij', q_block, P_interp, k_block)
            S = S * sm_scale

            if causal:
                mask = q_pos[:, None] >= k_pos[None, :]
                S = jnp.where(mask[None, None, :, :], S, -1e30)

            # Online softmax update
            S_max = S.max(axis=-1)
            m_next = jnp.maximum(m_qi, S_max)
            alpha = jnp.exp(m_qi - m_next)
            p = jnp.exp(S - m_next[..., None])
            l_next = alpha * l_qi + p.sum(axis=-1)
            o_next = alpha[..., None] * o_qi + jnp.einsum('bhij,bhjd->bhid', p, v_block)

            return o_next, m_next, l_next

        n_kv_iters = n_k_blocks
        if causal:
            n_kv_iters = jnp.minimum(n_k_blocks, qi + 1)

        o_qi, m_qi, l_qi = lax.fori_loop(0, n_kv_iters, inner_body,
                                          (o_qi, m_qi, l_qi))
        o_qi = o_qi / l_qi[..., None]

        o = lax.dynamic_update_slice(o, o_qi, (0, 0, q_start, 0))
        m = lax.dynamic_update_slice(m, m_qi, (0, 0, q_start))
        l = lax.dynamic_update_slice(l, l_qi, (0, 0, q_start))

        return o, m, l

    o, m, l = lax.fori_loop(0, n_q_blocks, outer_body, (o, m, l))

    return o, l, m


# ---------------------------------------------------------------------------
# Backward: two-pass chunked (dQ pass + dKV pass)
# ---------------------------------------------------------------------------

def _backward_chunked(q, k, v, P_table, sigma, o, l, m, do,
                      s_max, causal, block_q, block_k):
    """Two-pass backward. Only small tile accumulators in inner loop carries."""
    B, H, T, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)
    table_size = 2 * s_max + 1
    n_q_blocks = T // block_q
    n_k_blocks = T // block_k

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    do_f32 = do.astype(jnp.float32)
    sigma_abs = jnp.abs(sigma) + 1e-6

    di = (o.astype(jnp.float32) * do_f32).sum(axis=-1)  # [B, H, T]

    def recompute_A(q_block, k_block, P_interp, q_pos, k_pos, m_block, l_block):
        """Recompute attention weights for a tile from saved m, l."""
        S = jnp.einsum('bhid,hijd,bhjd->bhij', q_block, P_interp, k_block) * sm_scale
        if causal:
            mask = q_pos[:, None] >= k_pos[None, :]
            S = jnp.where(mask[None, None, :, :], S, -1e30)
        p = jnp.exp(S - m_block[..., None])
        return p / l_block[..., None]

    # ---- Pass 1: dQ (Q-major outer, KV inner) ----
    # Inner carry: dq_block [B,H,block_q,D] — small
    dq = jnp.zeros((B, H, T, D), dtype=jnp.float32)

    def dq_outer(qi, dq):
        q_start = qi * block_q
        q_pos = jnp.arange(block_q) + q_start
        q_block = lax.dynamic_slice(q_f32, (0, 0, q_start, 0), (B, H, block_q, D))
        do_block = lax.dynamic_slice(do_f32, (0, 0, q_start, 0), (B, H, block_q, D))
        m_block = lax.dynamic_slice(m, (0, 0, q_start), (B, H, block_q))
        l_block = lax.dynamic_slice(l, (0, 0, q_start), (B, H, block_q))
        di_block = lax.dynamic_slice(di, (0, 0, q_start), (B, H, block_q))

        dq_block = jnp.zeros((B, H, block_q, D), dtype=jnp.float32)

        def dq_inner(kvi, dq_block):
            k_start = kvi * block_k
            k_pos = jnp.arange(block_k) + k_start
            k_block = lax.dynamic_slice(k_f32, (0, 0, k_start, 0), (B, H, block_k, D))
            v_block = lax.dynamic_slice(v_f32, (0, 0, k_start, 0), (B, H, block_k, D))

            P_interp, _, _, _ = _tile_P_interp(q_pos, k_pos, P_table, sigma_abs, s_max)
            A = recompute_A(q_block, k_block, P_interp, q_pos, k_pos, m_block, l_block)

            dov = jnp.einsum('bhid,bhjd->bhij', do_block, v_block)
            dS = A * (dov - di_block[..., None]) * sm_scale

            # dQ += sum_j dS[i,j] * P[i,j,d] * K[j,d] — 3-input einsum
            dq_tile = jnp.einsum('bhij,hijd,bhjd->bhid', dS, P_interp, k_block)
            return dq_block + dq_tile

        n_kv = n_k_blocks
        if causal:
            n_kv = jnp.minimum(n_k_blocks, qi + 1)

        dq_block = lax.fori_loop(0, n_kv, dq_inner, dq_block)
        dq = lax.dynamic_update_slice(dq, dq_block, (0, 0, q_start, 0))
        return dq

    dq = lax.fori_loop(0, n_q_blocks, dq_outer, dq)

    # ---- Pass 2: dK, dV, dP_table, dsigma (KV-major outer, Q inner) ----
    # Inner carry: (dk_block, dv_block, dP_table, dsigma) — dk/dv are tile-sized
    dk = jnp.zeros((B, H, T, D), dtype=jnp.float32)
    dv = jnp.zeros((B, H, T, D), dtype=jnp.float32)
    dP_table = jnp.zeros((H, table_size, D), dtype=jnp.float32)
    dsigma = jnp.zeros((H,), dtype=jnp.float32)

    def dkv_outer(kvi, carry):
        dk, dv, dP_table, dsigma = carry
        k_start = kvi * block_k
        k_pos = jnp.arange(block_k) + k_start
        k_block = lax.dynamic_slice(k_f32, (0, 0, k_start, 0), (B, H, block_k, D))
        v_block = lax.dynamic_slice(v_f32, (0, 0, k_start, 0), (B, H, block_k, D))

        dk_block = jnp.zeros((B, H, block_k, D), dtype=jnp.float32)
        dv_block = jnp.zeros((B, H, block_k, D), dtype=jnp.float32)

        def dkv_inner(qi, inner_carry):
            dk_block, dv_block, dP_table, dsigma = inner_carry
            q_start = qi * block_q
            q_pos = jnp.arange(block_q) + q_start
            q_block = lax.dynamic_slice(q_f32, (0, 0, q_start, 0), (B, H, block_q, D))
            do_block = lax.dynamic_slice(do_f32, (0, 0, q_start, 0), (B, H, block_q, D))
            m_block = lax.dynamic_slice(m, (0, 0, q_start), (B, H, block_q))
            l_block = lax.dynamic_slice(l, (0, 0, q_start), (B, H, block_q))
            di_block = lax.dynamic_slice(di, (0, 0, q_start), (B, H, block_q))

            P_interp, idx_lo, idx_hi, frac = _tile_P_interp(
                q_pos, k_pos, P_table, sigma_abs, s_max
            )
            A = recompute_A(q_block, k_block, P_interp, q_pos, k_pos, m_block, l_block)

            dov = jnp.einsum('bhid,bhjd->bhij', do_block, v_block)
            dS = A * (dov - di_block[..., None]) * sm_scale

            # dV += A^T @ dO
            dv_tile = jnp.einsum('bhij,bhid->bhjd', A, do_block)
            dv_block = dv_block + dv_tile

            # dK += sum_i dS[i,j] * P[i,j,d] * Q[i,d] — 3-input einsum
            dk_tile = jnp.einsum('bhij,hijd,bhid->bhjd', dS, P_interp, q_block)
            dk_block = dk_block + dk_tile

            # dP_table: scatter-add (vectorized over heads, no vmap)
            dP_interp = jnp.einsum('bhij,bhid,bhjd->hijd', dS, q_block, k_block)
            dP_lo = (1.0 - frac[..., None]) * dP_interp  # [H, bq, bk, D]
            dP_hi = frac[..., None] * dP_interp

            head_idx_bc = jnp.broadcast_to(
                jnp.arange(H)[:, None, None], idx_lo.shape
            ).reshape(-1)
            dP_tile = jnp.zeros((H, table_size, D), dtype=jnp.float32)
            dP_tile = dP_tile.at[head_idx_bc, idx_lo.reshape(-1)].add(
                dP_lo.reshape(-1, D)
            )
            dP_tile = dP_tile.at[head_idx_bc, idx_hi.reshape(-1)].add(
                dP_hi.reshape(-1, D)
            )
            dP_table = dP_table + dP_tile

            # dsigma: chain rule through tanh warp (vectorized, no vmap)
            delta = (q_pos[:, None] - k_pos[None, :]).astype(jnp.float32)  # [bq, bk]
            t = jnp.tanh(delta[None, :, :] / sigma_abs[:, None, None])  # [H, bq, bk]
            sech2 = 1.0 - t * t
            ddelta_dsigma = s_max * sech2 * (
                -delta[None, :, :] / (sigma_abs[:, None, None] ** 2)
            )  # [H, bq, bk]

            head_idx = jnp.arange(H)[:, None, None]
            P_lo = P_table[head_idx, idx_lo, :]  # [H, bq, bk, D]
            P_hi = P_table[head_idx, idx_hi, :]  # [H, bq, bk, D]
            dP_dsigma = ddelta_dsigma[..., None] * (P_hi - P_lo)  # [H, bq, bk, D]

            ds_per_head = (dP_interp * dP_dsigma).sum(axis=(1, 2, 3))  # [H]
            dsigma = dsigma + ds_per_head * jnp.sign(sigma)

            return dk_block, dv_block, dP_table, dsigma

        # For causal: only Q blocks at or after this KV block contribute
        q_start_idx = kvi if causal else 0
        dkv_carry = lax.fori_loop(
            q_start_idx, n_q_blocks, dkv_inner,
            (dk_block, dv_block, dP_table, dsigma)
        )
        dk_block, dv_block, dP_table, dsigma = dkv_carry

        dk = lax.dynamic_update_slice(dk, dk_block, (0, 0, k_start, 0))
        dv = lax.dynamic_update_slice(dv, dv_block, (0, 0, k_start, 0))
        return dk, dv, dP_table, dsigma

    dk, dv, dP_table, dsigma = lax.fori_loop(
        0, n_k_blocks, dkv_outer, (dk, dv, dP_table, dsigma)
    )

    return dq, dk, dv, dP_table, dsigma


# ---------------------------------------------------------------------------
# Public API with custom_vjp
# ---------------------------------------------------------------------------

def _default_block_sizes(T):
    """Choose block sizes based on sequence length."""
    if T <= 64:
        return (T, T)
    elif T <= 256:
        return (64, 64)
    else:
        return (128, 128)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def poskernel_flash_attention(q, k, v, P_table, sigma,
                               s_max, causal, block_sizes):
    """Positional-kernel attention with custom_vjp for memory efficiency.

    Forward: chunked online softmax — no T×T materialization.
    Backward: two-pass chunked recomputation — no T×T residuals.

    Args:
        q: [B, H, T, D]
        k: [B, H, T, D]  (already GQA-expanded if needed)
        v: [B, H, T, D]  (already GQA-expanded if needed)
        P_table: [H, 2*s_max+1, D]
        sigma: [H]
        s_max: int
        causal: bool
        block_sizes: tuple (block_q, block_k) or None for auto

    Returns:
        o: [B, H, T, D]
    """
    block_q, block_k = block_sizes if block_sizes else _default_block_sizes(q.shape[2])
    o, _, _ = _forward_chunked(q, k, v, P_table, sigma, s_max, causal,
                                block_q, block_k)
    return o.astype(q.dtype)


def _poskernel_fwd(q, k, v, P_table, sigma, s_max, causal, block_sizes):
    """Forward pass: returns output + residuals (no T×T tensors)."""
    block_q, block_k = block_sizes if block_sizes else _default_block_sizes(q.shape[2])
    o, l, m_val = _forward_chunked(q, k, v, P_table, sigma, s_max, causal,
                                    block_q, block_k)
    o_out = o.astype(q.dtype)
    residuals = (q, k, v, P_table, sigma, o, l, m_val)
    return o_out, residuals


def _poskernel_bwd(s_max, causal, block_sizes, residuals, do):
    """Backward pass: two-pass chunked recomputation."""
    q, k, v, P_table, sigma, o, l, m_val = residuals
    block_q, block_k = block_sizes if block_sizes else _default_block_sizes(q.shape[2])

    dq, dk, dv, dP_table, dsigma = _backward_chunked(
        q, k, v, P_table, sigma, o, l, m_val, do,
        s_max, causal, block_q, block_k
    )

    return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), \
           dP_table.astype(P_table.dtype), dsigma.astype(sigma.dtype)


poskernel_flash_attention.defvjp(_poskernel_fwd, _poskernel_bwd)


# ---------------------------------------------------------------------------
# Naive reference (for testing only)
# ---------------------------------------------------------------------------

def _naive_poskernel_attention(q, k, v, P_table, sigma, s_max, causal):
    """Full-materialization reference. For testing only."""
    B, H, T, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)

    positions = jnp.arange(T)
    delta = (positions[:, None] - positions[None, :]).astype(jnp.float32)
    sigma_abs = jnp.abs(sigma) + 1e-6
    delta_warped = s_max * jnp.tanh(delta[None, :, :] / sigma_abs[:, None, None])
    delta_table = delta_warped + s_max

    idx_lo = jnp.floor(delta_table).astype(jnp.int32)
    idx_hi = idx_lo + 1
    frac = delta_table - idx_lo.astype(jnp.float32)
    table_max = 2 * s_max
    idx_lo = jnp.clip(idx_lo, 0, table_max)
    idx_hi = jnp.clip(idx_hi, 0, table_max)

    head_idx = jnp.arange(H)[:, None, None]
    P_lo = P_table[head_idx, idx_lo, :]
    P_hi = P_table[head_idx, idx_hi, :]
    P_interp = (1 - frac[..., None]) * P_lo + frac[..., None] * P_hi

    S = jnp.einsum('bhid,hijd,bhjd->bhij', q, P_interp, k) * sm_scale

    if causal:
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        S = jnp.where(causal_mask[None, None], S, jnp.finfo(S.dtype).min)

    A = jax.nn.softmax(S, axis=-1)
    return A @ v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_numerical_equivalence():
    """Test that chunked forward matches naive implementation."""
    print("Testing numerical equivalence (forward)...")

    B, H, T, D = 2, 4, 64, 32
    s_max = 16

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 5)
    q = jax.random.normal(keys[0], (B, H, T, D), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=jnp.float32)
    P_table = jax.random.normal(keys[3], (H, 2 * s_max + 1, D), dtype=jnp.float32)
    sigma = jax.random.uniform(keys[4], (H,), minval=10.0, maxval=100.0)

    o_naive = _naive_poskernel_attention(q, k, v, P_table, sigma, s_max, True)

    o_full_block = poskernel_flash_attention(q, k, v, P_table, sigma,
                                             s_max, True, (T, T))
    o_tiled = poskernel_flash_attention(q, k, v, P_table, sigma,
                                        s_max, True, (16, 16))

    diff_full = jnp.abs(o_naive - o_full_block).max()
    diff_tiled = jnp.abs(o_naive - o_tiled).max()
    print(f"  Max diff (block=T): {diff_full:.2e}")
    print(f"  Max diff (block=16): {diff_tiled:.2e}")
    assert diff_full < 1e-4, f"Full-block forward too far from naive: {diff_full}"
    assert diff_tiled < 1e-4, f"Tiled forward too far from naive: {diff_tiled}"
    print("  PASSED")


def _test_gradients():
    """Test that custom_vjp gradients match autograd through naive impl."""
    print("Testing gradient correctness...")

    B, H, T, D = 1, 2, 32, 16
    s_max = 8

    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 5)
    q = jax.random.normal(keys[0], (B, H, T, D), dtype=jnp.float32)
    k = jax.random.normal(keys[1], (B, H, T, D), dtype=jnp.float32)
    v = jax.random.normal(keys[2], (B, H, T, D), dtype=jnp.float32)
    P_table = jax.random.normal(keys[3], (H, 2 * s_max + 1, D), dtype=jnp.float32)
    sigma = jax.random.uniform(keys[4], (H,), minval=10.0, maxval=100.0)

    def loss_custom(q, k, v, P_table, sigma):
        return poskernel_flash_attention(q, k, v, P_table, sigma,
                                         s_max, True, (T, T)).sum()

    def loss_naive(q, k, v, P_table, sigma):
        return _naive_poskernel_attention(q, k, v, P_table, sigma, s_max, True).sum()

    grads_custom = jax.grad(loss_custom, argnums=(0, 1, 2, 3, 4))(
        q, k, v, P_table, sigma
    )
    grads_naive = jax.grad(loss_naive, argnums=(0, 1, 2, 3, 4))(
        q, k, v, P_table, sigma
    )

    names = ['dq', 'dk', 'dv', 'dP_table', 'dsigma']
    all_passed = True
    for name, gc, gn in zip(names, grads_custom, grads_naive):
        diff = jnp.abs(gc - gn).max()
        rel_diff = diff / (jnp.abs(gn).max() + 1e-8)
        status = "PASS" if rel_diff < 1e-2 else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {name}: max_abs_diff={diff:.2e}, max_rel_diff={rel_diff:.2e} [{status}]")

    assert all_passed, "Gradient check failed!"
    print("  PASSED")


if __name__ == '__main__':
    _test_numerical_equivalence()
    _test_gradients()
