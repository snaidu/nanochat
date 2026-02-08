"""Positional-kernel attention with custom_vjp.

Forward: Naive materialized computation — XLA fuses the einsum and softmax
efficiently. P_interp [H, T, T, D] is a temporary that is freed after use.

Backward: Chunked pure-JAX with lax.fori_loop. Recomputes P_interp per tile
from the small P_table + sigma, avoiding any T×T materialization across the
fwd/bwd boundary.

Residuals saved: (q, k, v, P_table, sigma, o, l, m) — no T×T tensors.
The P_table [H, 2*s_max+1, D] is ~32KB per head.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array

# ---------------------------------------------------------------------------
# Shared: P_interp computation
# ---------------------------------------------------------------------------

def _compute_P_interp(P_table, sigma, s_max, T):
    """Compute full interpolated P for all relative positions.

    Args:
        P_table: [H, 2*s_max+1, D]
        sigma: [H]
        s_max: int
        T: sequence length

    Returns:
        P_interp: [H, T, T, D]
    """
    H = P_table.shape[0]
    positions = jnp.arange(T)
    delta = (positions[:, None] - positions[None, :]).astype(jnp.float32)  # [T, T]

    sigma_abs = jnp.abs(sigma) + 1e-6  # [H]
    delta_warped = s_max * jnp.tanh(
        delta[None, :, :] / sigma_abs[:, None, None]
    )  # [H, T, T]
    delta_table = delta_warped + s_max

    idx_lo = jnp.floor(delta_table).astype(jnp.int32)
    idx_hi = idx_lo + 1
    frac = delta_table - idx_lo.astype(jnp.float32)

    table_max = 2 * s_max
    idx_lo = jnp.clip(idx_lo, 0, table_max)
    idx_hi = jnp.clip(idx_hi, 0, table_max)

    head_idx = jnp.arange(H)[:, None, None]
    P_lo = P_table[head_idx, idx_lo, :]  # [H, T, T, D]
    P_hi = P_table[head_idx, idx_hi, :]  # [H, T, T, D]

    return (1.0 - frac[..., None]) * P_lo + frac[..., None] * P_hi


def _compute_tile_P_interp(q_pos, k_pos, P_table_h, sigma_h, s_max):
    """Compute P_interp for a single head's tile of positions.

    Args:
        q_pos: [block_q] integer positions
        k_pos: [block_k] integer positions
        P_table_h: [table_size, D] for one head
        sigma_h: scalar
        s_max: int

    Returns:
        P_interp: [block_q, block_k, D]
        idx_lo, idx_hi: [block_q, block_k] int32
        frac: [block_q, block_k] float32
    """
    delta = (q_pos[:, None] - k_pos[None, :]).astype(jnp.float32)
    sigma_abs = jnp.abs(sigma_h) + 1e-6
    delta_warped = s_max * jnp.tanh(delta / sigma_abs)
    delta_table = delta_warped + s_max

    idx_lo = jnp.floor(delta_table).astype(jnp.int32)
    idx_hi = idx_lo + 1
    frac = delta_table - idx_lo.astype(jnp.float32)

    table_max = 2 * s_max
    idx_lo = jnp.clip(idx_lo, 0, table_max)
    idx_hi = jnp.clip(idx_hi, 0, table_max)

    P_lo = P_table_h[idx_lo]
    P_hi = P_table_h[idx_hi]
    P_interp = (1.0 - frac[..., None]) * P_lo + frac[..., None] * P_hi

    return P_interp, idx_lo, idx_hi, frac


# ---------------------------------------------------------------------------
# Forward: naive materialized (fast — XLA fuses everything)
# ---------------------------------------------------------------------------

def _forward_naive(q, k, v, P_table, sigma, s_max, causal):
    """Materialized forward pass. P_interp is a temporary freed after use.

    Returns o, l, m for the backward to recompute attention weights per tile.
    """
    B, H, T, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    P_interp = _compute_P_interp(P_table, sigma, s_max, T)  # [H, T, T, D]

    S = jnp.einsum('bhid,hijd,bhjd->bhij', q_f32, P_interp, k_f32) * sm_scale

    if causal:
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        S = jnp.where(causal_mask[None, None], S, -1e30)

    # Compute m and l explicitly (needed by backward to recompute A per tile)
    m = S.max(axis=-1)            # [B, H, T]
    p = jnp.exp(S - m[..., None])  # [B, H, T, T]
    l = p.sum(axis=-1)             # [B, H, T]
    A = p / l[..., None]           # [B, H, T, T]

    o = jnp.einsum('bhij,bhjd->bhid', A, v_f32)

    return o, l, m


# ---------------------------------------------------------------------------
# Backward: chunked pure-JAX (memory-efficient)
# ---------------------------------------------------------------------------

def _backward_chunked(q, k, v, P_table, sigma, o, l, m, do,
                      s_max, causal, block_q, block_k):
    """Chunked backward pass. Recomputes P_interp per tile — no T×T storage."""
    B, H, T, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)
    table_size = 2 * s_max + 1
    n_q_blocks = T // block_q
    n_k_blocks = T // block_k

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    do_f32 = do.astype(jnp.float32)

    dq = jnp.zeros((B, H, T, D), dtype=jnp.float32)
    dk = jnp.zeros((B, H, T, D), dtype=jnp.float32)
    dv = jnp.zeros((B, H, T, D), dtype=jnp.float32)
    dP_table = jnp.zeros_like(P_table, dtype=jnp.float32)
    dsigma = jnp.zeros_like(sigma, dtype=jnp.float32)

    # di[b,h,i] = sum_d o[b,h,i,d] * do[b,h,i,d]
    di = (o.astype(jnp.float32) * do_f32).sum(axis=-1)  # [B, H, T]

    def outer_body(qi, carry):
        dq, dk, dv, dP_table, dsigma = carry
        q_start = qi * block_q
        q_pos = jnp.arange(block_q) + q_start

        q_block = lax.dynamic_slice(q_f32, (0, 0, q_start, 0), (B, H, block_q, D))
        do_block = lax.dynamic_slice(do_f32, (0, 0, q_start, 0), (B, H, block_q, D))
        m_block = lax.dynamic_slice(m, (0, 0, q_start), (B, H, block_q))
        l_block = lax.dynamic_slice(l, (0, 0, q_start), (B, H, block_q))
        di_block = lax.dynamic_slice(di, (0, 0, q_start), (B, H, block_q))

        dq_block = jnp.zeros((B, H, block_q, D), dtype=jnp.float32)

        def inner_body(kvi, inner_carry):
            dq_block, dk, dv, dP_table, dsigma = inner_carry
            k_start = kvi * block_k
            k_pos = jnp.arange(block_k) + k_start

            k_block = lax.dynamic_slice(k_f32, (0, 0, k_start, 0), (B, H, block_k, D))
            v_block = lax.dynamic_slice(v_f32, (0, 0, k_start, 0), (B, H, block_k, D))

            # Recompute P_interp for all heads in this tile
            def compute_head_fwd(h):
                P_interp_tile, idx_lo, idx_hi, frac = _compute_tile_P_interp(
                    q_pos, k_pos, P_table[h], sigma[h], s_max
                )
                return P_interp_tile, idx_lo, idx_hi, frac

            P_interp_all, idx_lo_all, idx_hi_all, frac_all = jax.vmap(
                compute_head_fwd
            )(jnp.arange(H))

            # Recompute S and attention weights from saved m, l
            S = jnp.einsum('bhid,hijd,bhjd->bhij', q_block, P_interp_all, k_block)
            S = S * sm_scale

            if causal:
                mask = q_pos[:, None] >= k_pos[None, :]
                S = jnp.where(mask[None, None, :, :], S, -1e30)

            p = jnp.exp(S - m_block[..., None])
            A = p / l_block[..., None]

            # dS = A * (do @ v^T - di)
            dov = jnp.einsum('bhid,bhjd->bhij', do_block, v_block)
            dS = A * (dov - di_block[..., None]) * sm_scale

            # dV += A^T @ do
            dv_tile = jnp.einsum('bhij,bhid->bhjd', A, do_block)
            dv = lax.dynamic_update_slice(
                dv,
                lax.dynamic_slice(dv, (0, 0, k_start, 0), (B, H, block_k, D)) + dv_tile,
                (0, 0, k_start, 0)
            )

            # dQ += dS @ (P_interp * K)
            PK = P_interp_all[None, :, :, :, :] * k_block[:, :, None, :, :]
            dq_tile = jnp.einsum('bhij,bhijd->bhid', dS, PK)
            dq_block = dq_block + dq_tile

            # dK += dS^T @ (P_interp * Q)
            PQ = P_interp_all[None, :, :, :, :] * q_block[:, :, :, None, :]
            dk_tile = jnp.einsum('bhij,bhijd->bhjd', dS, PQ)
            dk = lax.dynamic_update_slice(
                dk,
                lax.dynamic_slice(dk, (0, 0, k_start, 0), (B, H, block_k, D)) + dk_tile,
                (0, 0, k_start, 0)
            )

            # dP_interp and scatter-add to dP_table
            dP_interp = jnp.einsum('bhij,bhid,bhjd->hijd', dS, q_block, k_block)

            def scatter_head(h):
                dP_h = dP_interp[h]
                ilo = idx_lo_all[h]
                ihi = idx_hi_all[h]
                fr = frac_all[h]

                dP_lo = (1.0 - fr[..., None]) * dP_h
                dP_hi = fr[..., None] * dP_h

                dP_table_h = jnp.zeros((table_size, D), dtype=jnp.float32)
                dP_table_h = dP_table_h.at[ilo.reshape(-1)].add(dP_lo.reshape(-1, D))
                dP_table_h = dP_table_h.at[ihi.reshape(-1)].add(dP_hi.reshape(-1, D))
                return dP_table_h

            dP_table_tile = jax.vmap(scatter_head)(jnp.arange(H))
            dP_table = dP_table + dP_table_tile

            # dsigma via chain rule through tanh warp
            def dsigma_head(h):
                sigma_h = jnp.abs(sigma[h]) + 1e-6
                delta = (q_pos[:, None] - k_pos[None, :]).astype(jnp.float32)
                t = jnp.tanh(delta / sigma_h)
                sech2 = 1.0 - t * t
                ddelta_dsigma = s_max * sech2 * (-delta / (sigma_h * sigma_h))

                P_h = P_table[h]
                P_lo = P_h[idx_lo_all[h]]
                P_hi = P_h[idx_hi_all[h]]
                dP_interp_dsigma = ddelta_dsigma[..., None] * (P_hi - P_lo)

                ds = (dP_interp[h] * dP_interp_dsigma).sum()
                ds = ds * jnp.sign(sigma[h])
                return ds

            dsigma_tile = jax.vmap(dsigma_head)(jnp.arange(H))
            dsigma = dsigma + dsigma_tile

            return dq_block, dk, dv, dP_table, dsigma

        n_kv_iters = n_k_blocks
        if causal:
            n_kv_iters = jnp.minimum(n_k_blocks, qi + 1)

        dq_block, dk, dv, dP_table, dsigma = lax.fori_loop(
            0, n_kv_iters, inner_body,
            (dq_block, dk, dv, dP_table, dsigma)
        )

        dq = lax.dynamic_update_slice(
            dq,
            lax.dynamic_slice(dq, (0, 0, q_start, 0), (B, H, block_q, D)) + dq_block,
            (0, 0, q_start, 0)
        )

        return dq, dk, dv, dP_table, dsigma

    dq, dk, dv, dP_table, dsigma = lax.fori_loop(
        0, n_q_blocks, outer_body, (dq, dk, dv, dP_table, dsigma)
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

    Forward: naive materialized computation (fast XLA fusion). P_interp is
    a temporary that is freed after use — not saved across fwd/bwd boundary.

    Backward: chunked recomputation of P_interp per tile from the small
    P_table + sigma. No T×T tensors stored as residuals.

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
    o, _, _ = _forward_naive(q, k, v, P_table, sigma, s_max, causal)
    return o.astype(q.dtype)


def _poskernel_fwd(q, k, v, P_table, sigma, s_max, causal, block_sizes):
    """Forward pass for custom_vjp: returns output + residuals."""
    o, l, m_val = _forward_naive(q, k, v, P_table, sigma, s_max, causal)
    o_out = o.astype(q.dtype)
    # Save residuals: no T×T tensors, just O(T) per head + references to inputs
    residuals = (q, k, v, P_table, sigma, o, l, m_val)
    return o_out, residuals


def _poskernel_bwd(s_max, causal, block_sizes, residuals, do):
    """Backward pass for custom_vjp: chunked recomputation."""
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
# Naive reference implementation (for testing)
# ---------------------------------------------------------------------------

def _naive_poskernel_attention(q, k, v, P_table, sigma, s_max, causal):
    """Full-materialization reference with standard autograd. For testing only."""
    B, H, T, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)

    P_interp = _compute_P_interp(P_table, sigma, s_max, T)

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
    """Test that custom_vjp forward matches naive implementation."""
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
    o_custom = poskernel_flash_attention(q, k, v, P_table, sigma, s_max, True, None)

    diff = jnp.abs(o_naive - o_custom).max()
    print(f"  Max diff: {diff:.2e}")
    assert diff < 1e-4, f"Forward too far from naive: {diff}"
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
        o = poskernel_flash_attention(q, k, v, P_table, sigma, s_max, True, (T, T))
        return o.sum()

    def loss_naive(q, k, v, P_table, sigma):
        o = _naive_poskernel_attention(q, k, v, P_table, sigma, s_max, True)
        return o.sum()

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
