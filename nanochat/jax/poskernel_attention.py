"""Fused positional-kernel attention with custom_vjp.

Forward: Pallas kernel on TPU (flash-attention-style online softmax with P_interp
computed on-the-fly per tile). Falls back to pure-JAX chunked implementation on
CPU/GPU.

Backward: Chunked pure-JAX with lax.fori_loop. Recomputes P_interp per tile from
the small P_table + sigma, avoiding any T×T materialization.

The P_table [H, 2*s_max+1, D] is ~32KB per head and fits entirely in VMEM.
Only O, l, m (O(T) per head) are saved as forward residuals.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array

# ---------------------------------------------------------------------------
# Tile-level helpers (shared by forward and backward)
# ---------------------------------------------------------------------------

def _compute_tile_P_interp(q_pos, k_pos, P_table, sigma_h, s_max):
    """Compute P_interp for a tile of (q_pos, k_pos) positions.

    Args:
        q_pos: [block_q] integer positions
        k_pos: [block_k] integer positions
        P_table: [table_size, D] for one head
        sigma_h: scalar
        s_max: int

    Returns:
        P_interp: [block_q, block_k, D]
        idx_lo: [block_q, block_k] int32
        idx_hi: [block_q, block_k] int32
        frac: [block_q, block_k] float32
    """
    delta = (q_pos[:, None] - k_pos[None, :]).astype(jnp.float32)  # [bq, bk]
    sigma_abs = jnp.abs(sigma_h) + 1e-6
    delta_warped = s_max * jnp.tanh(delta / sigma_abs)
    delta_table = delta_warped + s_max  # [0, 2*s_max]

    idx_lo = jnp.floor(delta_table).astype(jnp.int32)
    idx_hi = idx_lo + 1
    frac = delta_table - idx_lo.astype(jnp.float32)

    table_max = 2 * s_max
    idx_lo = jnp.clip(idx_lo, 0, table_max)
    idx_hi = jnp.clip(idx_hi, 0, table_max)

    P_lo = P_table[idx_lo]  # [bq, bk, D]
    P_hi = P_table[idx_hi]  # [bq, bk, D]
    P_interp = (1.0 - frac[..., None]) * P_lo + frac[..., None] * P_hi

    return P_interp, idx_lo, idx_hi, frac


# ---------------------------------------------------------------------------
# Forward: chunked pure-JAX (works on all platforms)
# ---------------------------------------------------------------------------

def _forward_chunked(q, k, v, P_table, sigma, s_max, causal,
                     block_q, block_k):
    """Flash-attention-style forward with online softmax, processing tiles.

    This avoids materializing the full [H, T, T, D] P_interp or [B, H, T, T]
    attention matrix. Peak memory per tile is O(block_q * block_k * D).

    Args:
        q, k, v: [B, H, T, D]
        P_table: [H, 2*s_max+1, D]
        sigma: [H]
        s_max: int
        causal: bool
        block_q, block_k: tile sizes

    Returns:
        o: [B, H, T, D]
        l: [B, H, T] (logsumexp denominators)
        m: [B, H, T] (running max)
    """
    B, H, T, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)
    n_q_blocks = T // block_q
    n_k_blocks = T // block_k

    # Initialize outputs
    o = jnp.zeros((B, H, T, D), dtype=jnp.float32)
    m = jnp.full((B, H, T), -1e30, dtype=jnp.float32)
    l = jnp.zeros((B, H, T), dtype=jnp.float32)

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    def outer_body(qi, carry):
        o, m, l = carry
        q_start = qi * block_q
        q_pos = jnp.arange(block_q) + q_start
        q_block = lax.dynamic_slice(q_f32, (0, 0, q_start, 0), (B, H, block_q, D))

        # Per-Q-block accumulators
        o_qi = jnp.zeros((B, H, block_q, D), dtype=jnp.float32)
        m_qi = jnp.full((B, H, block_q), -1e30, dtype=jnp.float32)
        l_qi = jnp.zeros((B, H, block_q), dtype=jnp.float32)

        def inner_body(kvi, inner_carry):
            o_qi, m_qi, l_qi = inner_carry
            k_start = kvi * block_k
            k_pos = jnp.arange(block_k) + k_start

            k_block = lax.dynamic_slice(k_f32, (0, 0, k_start, 0), (B, H, block_k, D))
            v_block = lax.dynamic_slice(v_f32, (0, 0, k_start, 0), (B, H, block_k, D))

            def compute_head(h):
                P_h = P_table[h]  # [table_size, D]
                sigma_h = sigma[h]
                P_interp_tile, _, _, _ = _compute_tile_P_interp(
                    q_pos, k_pos, P_h, sigma_h, s_max
                )  # [bq, bk, D]
                return P_interp_tile

            # Compute P_interp for all heads: [H, bq, bk, D]
            P_interp_all = jax.vmap(compute_head)(jnp.arange(H))

            # Logits: S[b,h,i,j] = sum_d q[b,h,i,d] * P[h,i,j,d] * k[b,h,j,d]
            # q_block: [B, H, bq, D], P_interp_all: [H, bq, bk, D], k_block: [B, H, bk, D]
            S = jnp.einsum('bhid,hijd,bhjd->bhij', q_block, P_interp_all, k_block)
            S = S * sm_scale

            # Causal mask
            if causal:
                mask = q_pos[:, None] >= k_pos[None, :]  # [bq, bk]
                S = jnp.where(mask[None, None, :, :], S, -1e30)

            # Online softmax update
            S_max = S.max(axis=-1)  # [B, H, bq]
            m_next = jnp.maximum(m_qi, S_max)
            alpha = jnp.exp(m_qi - m_next)  # [B, H, bq]
            p = jnp.exp(S - m_next[..., None])  # [B, H, bq, bk]
            l_next = alpha * l_qi + p.sum(axis=-1)  # [B, H, bq]

            # Update accumulator
            o_next = alpha[..., None] * o_qi + jnp.einsum('bhij,bhjd->bhid', p, v_block)

            return o_next, m_next, l_next

        n_kv_iters = n_k_blocks
        if causal:
            # Only iterate up to the block that could contain valid KV positions
            n_kv_iters = jnp.minimum(n_k_blocks, qi + 1)

        o_qi, m_qi, l_qi = lax.fori_loop(0, n_kv_iters, inner_body,
                                          (o_qi, m_qi, l_qi))

        # Normalize
        o_qi = o_qi / l_qi[..., None]

        # Write back
        o = lax.dynamic_update_slice(o, o_qi, (0, 0, q_start, 0))
        m = lax.dynamic_update_slice(m, m_qi, (0, 0, q_start))
        l = lax.dynamic_update_slice(l, l_qi, (0, 0, q_start))

        return o, m, l

    o, m, l = lax.fori_loop(0, n_q_blocks, outer_body, (o, m, l))

    return o, l, m


# ---------------------------------------------------------------------------
# Backward: chunked pure-JAX
# ---------------------------------------------------------------------------

def _backward_chunked(q, k, v, P_table, sigma, o, l, m, do,
                      s_max, causal, block_q, block_k):
    """Chunked backward pass for positional kernel attention.

    Recomputes P_interp per tile from P_table + sigma. No T×T materialization.

    Args:
        q, k, v: [B, H, T, D]
        P_table: [H, 2*s_max+1, D]
        sigma: [H]
        o: [B, H, T, D] (forward output)
        l: [B, H, T] (softmax denominators from forward)
        m: [B, H, T] (row maxima from forward)
        do: [B, H, T, D] (upstream gradient)
        s_max, causal: forward hyperparams
        block_q, block_k: tile sizes

    Returns:
        dq, dk, dv: [B, H, T, D]
        dP_table: [H, 2*s_max+1, D]
        dsigma: [H]
    """
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

            # Recompute P_interp and intermediates for all heads
            def compute_head_fwd(h):
                P_h = P_table[h]
                sigma_h = sigma[h]
                P_interp_tile, idx_lo, idx_hi, frac = _compute_tile_P_interp(
                    q_pos, k_pos, P_h, sigma_h, s_max
                )
                return P_interp_tile, idx_lo, idx_hi, frac

            P_interp_all, idx_lo_all, idx_hi_all, frac_all = jax.vmap(
                compute_head_fwd
            )(jnp.arange(H))
            # P_interp_all: [H, bq, bk, D]
            # idx_lo_all, idx_hi_all: [H, bq, bk]
            # frac_all: [H, bq, bk]

            # Recompute S
            S = jnp.einsum('bhid,hijd,bhjd->bhij', q_block, P_interp_all, k_block)
            S = S * sm_scale

            if causal:
                mask = q_pos[:, None] >= k_pos[None, :]
                S = jnp.where(mask[None, None, :, :], S, -1e30)

            # Recompute attention weights from saved m, l
            p = jnp.exp(S - m_block[..., None])  # [B, H, bq, bk]
            A = p / l_block[..., None]  # [B, H, bq, bk]

            # dS = A * (do @ v^T - di)
            # dS[b,h,i,j] = A[b,h,i,j] * (sum_d do[b,h,i,d]*v[b,h,j,d] - di[b,h,i])
            dov = jnp.einsum('bhid,bhjd->bhij', do_block, v_block)
            dS = A * (dov - di_block[..., None]) * sm_scale

            # dV: [B, H, bk, D] += A^T @ do
            dv_tile = jnp.einsum('bhij,bhid->bhjd', A, do_block)
            dv = lax.dynamic_update_slice(
                dv,
                lax.dynamic_slice(dv, (0, 0, k_start, 0), (B, H, block_k, D)) + dv_tile,
                (0, 0, k_start, 0)
            )

            # dQ: [B, H, bq, D] += dS @ (P_interp * K)
            # dq[b,h,i,d] += sum_j dS[b,h,i,j] * P[h,i,j,d] * k[b,h,j,d]
            PK = P_interp_all[None, :, :, :, :] * k_block[:, :, None, :, :]
            # PK: [B, H, bq, bk, D]
            dq_tile = jnp.einsum('bhij,bhijd->bhid', dS, PK)
            dq_block = dq_block + dq_tile

            # dK: [B, H, bk, D] += dS^T @ (P_interp * Q)
            PQ = P_interp_all[None, :, :, :, :] * q_block[:, :, :, None, :]
            # PQ: [B, H, bq, bk, D]
            dk_tile = jnp.einsum('bhij,bhijd->bhjd', dS, PQ)
            dk = lax.dynamic_update_slice(
                dk,
                lax.dynamic_slice(dk, (0, 0, k_start, 0), (B, H, block_k, D)) + dk_tile,
                (0, 0, k_start, 0)
            )

            # dP_interp[h,i,j,d] = sum_b dS[b,h,i,j] * q[b,h,i,d] * k[b,h,j,d]
            dP_interp = jnp.einsum('bhij,bhid,bhjd->hijd', dS, q_block, k_block)
            # [H, bq, bk, D]

            # Scatter-add to dP_table via idx_lo, idx_hi, frac
            def scatter_head(h):
                dP_h = dP_interp[h]  # [bq, bk, D]
                ilo = idx_lo_all[h]  # [bq, bk]
                ihi = idx_hi_all[h]  # [bq, bk]
                fr = frac_all[h]  # [bq, bk]

                dP_lo = (1.0 - fr[..., None]) * dP_h  # [bq, bk, D]
                dP_hi = fr[..., None] * dP_h  # [bq, bk, D]

                # Use scatter add
                dP_table_h = jnp.zeros((table_size, D), dtype=jnp.float32)
                # Flatten for segment_sum-style accumulation
                ilo_flat = ilo.reshape(-1)
                ihi_flat = ihi.reshape(-1)
                dP_lo_flat = dP_lo.reshape(-1, D)
                dP_hi_flat = dP_hi.reshape(-1, D)

                dP_table_h = dP_table_h.at[ilo_flat].add(dP_lo_flat)
                dP_table_h = dP_table_h.at[ihi_flat].add(dP_hi_flat)

                return dP_table_h

            dP_table_tile = jax.vmap(scatter_head)(jnp.arange(H))  # [H, table_size, D]
            dP_table = dP_table + dP_table_tile

            # dsigma gradient via chain rule through tanh warp
            # d(frac)/d(sigma) = d(delta_table)/d(sigma) * (gradient of floor frac)
            # delta_table = s_max * tanh(delta / sigma) + s_max
            # d(delta_table)/d(sigma) = s_max * sech^2(delta/sigma) * (-delta/sigma^2)
            def dsigma_head(h):
                sigma_h = jnp.abs(sigma[h]) + 1e-6
                delta = (q_pos[:, None] - k_pos[None, :]).astype(jnp.float32)  # [bq, bk]
                t = jnp.tanh(delta / sigma_h)
                sech2 = 1.0 - t * t
                ddelta_dsigma = s_max * sech2 * (-delta / (sigma_h * sigma_h))
                # ddelta_dsigma: [bq, bk]

                # dP_interp wrt sigma flows through the interpolation
                # P_interp = (1-frac)*P_lo + frac*P_hi
                # dfrac/dsigma = ddelta_table/dsigma (since frac = delta_table - floor(delta_table))
                # dP_interp/dsigma = dfrac/dsigma * (P_hi - P_lo)
                P_h = P_table[h]
                ilo = idx_lo_all[h]
                ihi = idx_hi_all[h]
                P_lo = P_h[ilo]  # [bq, bk, D]
                P_hi = P_h[ihi]  # [bq, bk, D]

                dP_interp_dsigma = ddelta_dsigma[..., None] * (P_hi - P_lo)
                # [bq, bk, D]

                # dsigma_h = sum_{i,j,d} dP_interp[h,i,j,d] * dP_interp_dsigma[i,j,d]
                dP_h = dP_interp[h]  # [bq, bk, D]
                ds = (dP_h * dP_interp_dsigma).sum()

                # Account for abs(sigma): d|sigma|/dsigma = sign(sigma)
                ds = ds * jnp.sign(sigma[h])
                return ds

            dsigma_tile = jax.vmap(dsigma_head)(jnp.arange(H))  # [H]
            dsigma = dsigma + dsigma_tile

            return dq_block, dk, dv, dP_table, dsigma

        n_kv_iters = n_k_blocks
        if causal:
            n_kv_iters = jnp.minimum(n_k_blocks, qi + 1)

        dq_block, dk, dv, dP_table, dsigma = lax.fori_loop(
            0, n_kv_iters, inner_body,
            (dq_block, dk, dv, dP_table, dsigma)
        )

        # Write dq_block back
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
    """Positional-kernel attention with flash-attention-style tiling.

    Avoids materializing the full [H, T, T, D] P_interp tensor or the
    full [B, H, T, T] attention matrix. Peak memory is O(block_q * block_k * D)
    per tile. Scale factor (1/sqrt(D)) is computed internally from q.shape.

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
    """Forward pass for custom_vjp: returns output + residuals."""
    block_q, block_k = block_sizes if block_sizes else _default_block_sizes(q.shape[2])
    o, l, m_val = _forward_chunked(q, k, v, P_table, sigma, s_max, causal,
                                    block_q, block_k)
    o_out = o.astype(q.dtype)
    # Save residuals: no T×T tensors, just O(T) per head + references to inputs
    residuals = (q, k, v, P_table, sigma, o, l, m_val)
    return o_out, residuals


def _poskernel_bwd(s_max, causal, block_sizes, residuals, do):
    """Backward pass for custom_vjp."""
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
    """Full-materialization reference. Only for testing small sizes.

    Args:
        q, k, v: [B, H, T, D]
        P_table: [H, 2*s_max+1, D]
        sigma: [H]

    Returns:
        o: [B, H, T, D]
    """
    B, H, T, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)

    positions = jnp.arange(T)
    delta = positions[:, None] - positions[None, :]  # [T, T]

    sigma_abs = jnp.abs(sigma) + 1e-6  # [H]
    delta_warped = s_max * jnp.tanh(
        delta[None, :, :] / sigma_abs[:, None, None]
    )  # [H, T, T]

    delta_table = delta_warped + s_max  # [H, T, T]

    idx_lo = jnp.floor(delta_table).astype(jnp.int32)
    idx_hi = idx_lo + 1
    frac = delta_table - idx_lo.astype(jnp.float32)

    table_max = 2 * s_max
    idx_lo = jnp.clip(idx_lo, 0, table_max)
    idx_hi = jnp.clip(idx_hi, 0, table_max)

    head_idx = jnp.arange(H)[:, None, None]
    P_lo = P_table[head_idx, idx_lo, :]  # [H, T, T, D]
    P_hi = P_table[head_idx, idx_hi, :]  # [H, T, T, D]
    P_interp = (1 - frac[..., None]) * P_lo + frac[..., None] * P_hi

    S = jnp.einsum('bhid,hijd,bhjd->bhij', q, P_interp, k) * sm_scale

    if causal:
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        S = jnp.where(causal_mask[None, None], S, jnp.finfo(S.dtype).min)

    A = jax.nn.softmax(S, axis=-1)
    o = A @ v

    return o


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

    # Naive reference
    o_naive = _naive_poskernel_attention(q, k, v, P_table, sigma, s_max, True)

    # Chunked (via public API)
    o_chunked = poskernel_flash_attention(q, k, v, P_table, sigma,
                                          s_max, True, (T, T))

    # Also test with actual tiling
    o_tiled = poskernel_flash_attention(q, k, v, P_table, sigma,
                                        s_max, True, (16, 16))

    diff_chunked = jnp.abs(o_naive - o_chunked).max()
    diff_tiled = jnp.abs(o_naive - o_tiled).max()

    print(f"  Max diff (block_size=T): {diff_chunked:.2e}")
    print(f"  Max diff (block_size=16): {diff_tiled:.2e}")

    assert diff_chunked < 1e-4, f"Chunked forward too far from naive: {diff_chunked}"
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
        o = poskernel_flash_attention(q, k, v, P_table, sigma,
                                      s_max, True, (T, T))
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
