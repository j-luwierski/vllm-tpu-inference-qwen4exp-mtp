# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU (torchax) Qwen Sparse Attention (QSA) for Qwen4Exp.

Port of ``vllm/models/qwen4_exp/nvidia/qsa.py`` +
``vllm/models/qwen4_exp/nvidia/indexer_qsa.py`` (with their Triton kernels in
``nvidia/ops/qsa*.py``) to torch/JAX that torchax lowers to XLA.

Semantics follow the vLLM CUDA reference:

1. An indexer side branch projects per-token indexer Q/K
   (``index_qk_proj``), Gemma-normalizes Q (K is normalized only after group
   pooling), applies the main attention's 1-D RoPE, stores the raw
   (un-normed, un-roped) keys in a per-request raw-key ring, mean-pools each
   completed group of ``compress_ratio`` consecutive raw keys, normalizes the
   pooled key, ropes it at the group's first position, and stores it in a
   per-request compressed-key bank.
2. Every query row scores visible compressed groups with
   ``sum_h(relu(q_h . k_g)) / sqrt(index_head_dim)`` (MQA), keeps the top
   ``indexer_budget / compress_ratio`` groups, expands them to token indices
   and appends the visible tail tokens of the open group (the same math as
   ``tests/models/qwen4_exp/test_qsa_reference.py`` in the vLLM repo).
3. The main attention attends only to the selected tokens, gathered from the
   paged main KV cache (GQA + sigmoid output gate).

TPU-specific engineering decisions (deliberate, none silent):

* Side caches are per-request static buffers allocated through the TPU
  runner's Mamba-state mechanism (``MambaSpec``) instead of vLLM's paged
  side caches. The raw-key ring only needs ``compress_ratio + num_spec``
  entries (group pooling never reads further back, and same-step reads use
  the pre-step functional state, so ring aliasing cannot corrupt a group).
  The compressed-key bank holds
  ``min(max_model_len, TPU_QSA_STATE_CAP_TOKENS) / compress_ratio`` groups
  per request: faithful by default, memory heavy at 262k context;
  production long-context serving needs paged side caches (follow-up).
* Selection logits/top-k stream through a ``jax.lax.fori_loop`` merge so the
  trip count stays dynamic in context length; the final attention runs over
  the static selection width with an unrolled chunked online softmax.
* The reference's per-row stable sort of expanded indices is skipped:
  softmax over a set is order-invariant, so numerics are identical.
* The MTP "skip_topk" step-0 selection reuse is not implemented; every draft
  step re-selects (causally valid, slightly more compute).
* 2-D (MRoPE) positions are rejected: the Qwen4Exp text path is 1-D RoPE.
* dcp/pcp mesh axes > 1 and KV-cache quantization are rejected, matching the
  CUDA reference's own guardrails.

Block 0 of every paged cache (main KV and Mamba-style states) is treated as
the reserved null block, consistent with the GDN bridge in
``tpu_inference/layers/vllm/custom_ops/gdn_attention_op.py``; masked writes
target it with zero values.
"""

import math
import os
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from torch import nn
from torchax.interop import jax_view, torch_view
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)

# Compressed groups scored per streaming iteration of the indexer loop.
_GROUP_CHUNK = 256
# Selected tokens attended per unrolled chunk of the sparse attention.
_SEL_CHUNK = 64


def _state_cap_tokens(max_model_len: int) -> int:
    """Per-request compressed-key bank length in tokens."""
    cap = os.getenv("TPU_QSA_STATE_CAP_TOKENS")
    if cap is None:
        return max_model_len
    cap_tokens = int(cap)
    if cap_tokens <= 0:
        raise ValueError("TPU_QSA_STATE_CAP_TOKENS must be positive")
    if cap_tokens < max_model_len:
        logger.warning_once(
            "TPU_QSA_STATE_CAP_TOKENS=%d caps the QSA compressed-key bank "
            "below max_model_len=%d. Group selection beyond the cap "
            "degenerates to recent-context-only indexing; long-context "
            "quality will degrade.", cap_tokens, max_model_len)
    return min(cap_tokens, max_model_len)


def _get_attention_metadata():
    """The TPU runner's AttentionMetadata for the current step, or None."""
    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        md = next(iter(md.values()), None)
    return md


class _QSAKeyStateCache(nn.Module, MambaBase):
    """Per-request key state for one QSA side cache (no checkpoint weights).

    State layout per slot: ``[capacity, 1, head_size]``. The size-1 axis
    absorbs the runner's hardcoded Mamba sharding spec (state index 0 shards
    ``[blocks, None, head]``) so the state stays TP-replicated, matching the
    replicated indexer.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        capacity_tokens: int,
        head_size: int,
        dtype: torch.dtype,
        prefix: str,
    ) -> None:
        nn.Module.__init__(self)
        self.capacity_tokens = capacity_tokens
        self.head_size = head_size
        self._dtype = dtype
        self.prefix = prefix
        self.kv_cache = (torch.tensor([]), )
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_state_shape(self) -> Tuple[Tuple[int, int, int]]:
        return ((self.capacity_tokens, 1, self.head_size), )

    def get_state_dtype(self) -> Tuple[torch.dtype]:
        return (self._dtype, )


def _token_to_req(query_start_loc: jax.Array, num_tokens: int) -> jax.Array:
    """Request index per flat token slot, ``[num_tokens]`` int32.

    ``query_start_loc`` is padded with the total token count, so padded slots
    map to the last (empty, padded) request and are masked out by callers.
    """
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    return (query_start_loc[None, :-1].astype(jnp.int32) <=
            tok[:, None]).sum(axis=1, dtype=jnp.int32) - 1


def apply_qsa_rope(rotary_emb: nn.Module, positions: torch.Tensor,
                   tensor: torch.Tensor) -> torch.Tensor:
    """Apply the main attention's 1-D RoPE composition to QSA heads.

    Reference: ``apply_qsa_rope`` in ``nvidia/indexer_qsa.py``. ``tensor`` is
    ``[num_tokens, num_heads, head_dim]`` with ``head_dim`` possibly larger
    than the rope's own head size, so the rotation is applied inline to the
    first ``rotary_dim`` channels.
    """
    rotary_dim = int(rotary_emb.rotary_dim)
    half = rotary_dim // 2
    cache = rotary_emb.cos_sin_cache
    pos = positions.long().clamp(0, cache.shape[0] - 1)
    cos_sin = cache[pos].to(tensor.dtype)
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x = tensor[..., :rotary_dim]
    if getattr(rotary_emb, "is_neox_style", True):
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat(
            (x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack(
            (x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


class QSAIndexer(nn.Module):
    """Qwen4Exp QSA indexer: projections, side caches, group selection."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        rotary_emb: nn.Module,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen4Exp QSA on TPU currently requires BF16 model dtype")
        cache_dtype = getattr(vllm_config.cache_config, "cache_dtype", "auto")
        if cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen4Exp QSA on TPU requires a BF16 main KV cache")
        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        if self.index_kv_heads != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA on TPU requires indexer_kv_heads=1")
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.block_topk = self.token_topk // self.compress_ratio
        self.rotary_emb = rotary_emb
        self.prefix = prefix

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )

        max_model_len = vllm_config.model_config.max_model_len
        cap_tokens = _state_cap_tokens(max_model_len)
        num_spec = vllm_config.num_speculative_tokens
        # Group pooling never reads further back than the open group; the
        # speculative slack keeps rejected-draft writes from aliasing a
        # committed member of the group still being collected.
        self.raw_capacity = self.compress_ratio - 1 + num_spec + 1
        self.compressed_capacity = max(1, cap_tokens // self.compress_ratio)
        self.raw_key_state = _QSAKeyStateCache(
            vllm_config=vllm_config,
            capacity_tokens=self.raw_capacity,
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.raw_key_cache",
        )
        self.compressed_key_state = _QSAKeyStateCache(
            vllm_config=vllm_config,
            capacity_tokens=self.compressed_capacity,
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.compressed_key_cache",
        )

    @property
    def output_width(self) -> int:
        """Selection (index) columns per row."""
        return self.token_topk + self.compress_ratio - 1

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Indexer Q/K projection, ``[T, (nH + 1) * index_head_dim]``."""
        return self.index_qk_proj(hidden_states)[0]


# ----------------------------------------------------------------------
# JAX core: paged main-cache update, group pooling, selection, attention
# ----------------------------------------------------------------------
def _qsa_update_main_cache(
    k: jax.Array,  # [T, KH, D] bf16
    v: jax.Array,  # [T, KH, D] bf16
    positions: jax.Array,  # [T] int32
    token_to_req: jax.Array,  # [T] int32
    valid: jax.Array,  # [T] bool
    seq_lens: jax.Array,  # [R] int32
    block_tables: jax.Array,  # [R, bt_w] int32
    cache: jax.Array,  # [B, S, G, packing, D_pad]
) -> jax.Array:
    """Scatter this step's K/V into the packed paged main KV cache.

    Layout matches ``tpu_inference.runner.kv_cache.create_kv_caches`` /
    ``ragged_paged_attention.v3.kernel.get_kv_cache_shape`` for bf16
    (packing 2). The cache splits the logical ``[2 * KH]`` KV-row axis into
    ``(G, packing)`` in row-major order, so ``concat([k_heads, v_heads])
    .reshape(T, G, packing, D)`` is exactly the packed layout.
    """
    T = k.shape[0]
    B, S, G, packing, D_pad = cache.shape
    num_kv_heads = k.shape[1]
    head_dim = k.shape[-1]
    assert G * packing == 2 * num_kv_heads
    pos = jnp.where(valid, positions, 0)
    req = jnp.clip(token_to_req, 0, seq_lens.shape[0] - 1)
    bt_col = jnp.clip(pos // S, 0, block_tables.shape[1] - 1)
    page = jnp.clip(block_tables[req, bt_col], 0, B - 1)
    slot = page * S + pos % S
    # Masked writes target the reserved null block (block 0, offset 0).
    slot = jnp.where(valid & (seq_lens[req] > 0), slot, 0)
    kv = jnp.concatenate((k, v), axis=1)  # [T, 2KH, D], K heads then V
    kv = kv.reshape(T, G, packing, head_dim)
    kv = jnp.pad(kv, ((0, 0), (0, 0), (0, 0), (0, D_pad - head_dim)))
    kv = jnp.where(valid[:, None, None, None], kv,
                   jnp.zeros_like(kv, dtype=kv.dtype))
    flat = cache.reshape(B * S, G, packing, D_pad)
    return flat.at[slot].set(kv.astype(flat.dtype)).reshape(cache.shape)


def _qsa_update_raw_and_pool(
    raw_cache: jax.Array,  # [num_blocks, raw_cap, 1, D] bf16 (pre-step)
    raw_keys: jax.Array,  # [T, D] bf16, as-projected (no norm, no rope)
    positions: jax.Array,  # [T] int32
    token_to_req: jax.Array,  # [T] int32
    valid: jax.Array,  # [T] bool
    seq_lens: jax.Array,  # [R] int32
    query_start_loc: jax.Array,  # [R+1] int32
    slot_indices: jax.Array,  # [R] int32
    compress_ratio: int,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Store raw keys into the per-request ring, then pool completed groups.

    Same-step reads use the pre-step ring state (purely functional), and the
    ring keeps ``compress_ratio + num_spec`` committed entries, so neither
    speculative rollbacks nor step-boundary groups can alias a member.

    Members committed before this step come from the raw-key ring; members
    scheduled this step come from the activations. Returns the updated ring,
    the pooled keys ``[T, D]`` bf16 (valid only where ``completing``), the
    group-first logical positions ``[T]`` int32, and the ``completing`` mask
    ``[T]``.
    """
    cr = compress_ratio
    raw_cap = raw_cache.shape[1]
    head_dim = raw_keys.shape[-1]
    num_tokens = raw_keys.shape[0]
    req = jnp.clip(token_to_req, 0, seq_lens.shape[0] - 1)
    slot_r = slot_indices[req]
    num_computed = seq_lens - (query_start_loc[1:] - query_start_loc[:-1])
    nc_r = num_computed[req]

    pos = jnp.where(valid, positions, 0)
    write_ok = valid & (seq_lens[req] > 0) & (slot_r >= 0)
    raw_flat = raw_cache.reshape(-1, 1, head_dim)
    raw_slot = jnp.where(write_ok, slot_r * raw_cap + pos % raw_cap, 0)
    write_vals = jnp.where(write_ok[:, None, None], raw_keys[:, None, :],
                           jnp.zeros_like(raw_keys)[:, None, :])
    new_raw_flat = raw_flat.at[raw_slot].set(write_vals)
    new_raw_cache = new_raw_flat.reshape(raw_cache.shape)

    group = pos // cr
    members = group[:, None] * cr + jnp.arange(cr)[None, :]  # [T, cr]
    in_batch = members >= nc_r[:, None]
    batch_idx = jnp.clip(
        query_start_loc[req][:, None] + (members - nc_r[:, None]), 0,
        num_tokens - 1)
    batch_vals = raw_keys[batch_idx]  # [T, cr, D]
    ring_idx = slot_r[:, None] * raw_cap + members % raw_cap
    ring_vals = raw_flat[ring_idx][:, :, 0]
    member_vals = jnp.where(in_batch[..., None], batch_vals, ring_vals)
    pooled = member_vals.astype(jnp.float32).mean(axis=1).astype(
        raw_keys.dtype)  # [T, D]

    completing = (valid & (pos % cr == cr - 1) &
                  (seq_lens[req] > 0) & (slot_r >= 0))
    first_pos = jnp.where(completing, group * cr, 0)
    return new_raw_cache, pooled, first_pos, completing


def _qsa_select_and_store(
    comp_cache: jax.Array,  # [num_blocks, comp_cap, 1, D] bf16 (pre-step)
    pooled_roped: jax.Array,  # [T, 1, D] bf16, normed + roped, masked
    completing: jax.Array,  # [T] bool
    q_sel: jax.Array,  # [T, nH_idx, D] bf16 (normed + roped)
    positions: jax.Array,  # [T] int32
    token_to_req: jax.Array,  # [T] int32
    valid: jax.Array,  # [T] bool
    seq_lens: jax.Array,  # [R] int32
    slot_indices: jax.Array,  # [R] int32
    compress_ratio: int,
    block_topk: int,
    index_head_dim: int,
) -> Tuple[jax.Array, jax.Array]:
    """Store pooled keys and run the streaming top-k group selection.

    Returns the updated compressed-key cache and the packed selection
    ``[T, block_topk * compress_ratio + compress_ratio - 1]`` int32 of
    request-relative token indices, ``-1`` padded.
    """
    cr = compress_ratio
    comp_cap = comp_cache.shape[1]
    head_dim = index_head_dim
    num_tokens = q_sel.shape[0]
    req = jnp.clip(token_to_req, 0, seq_lens.shape[0] - 1)
    slot_r = slot_indices[req]

    pos = jnp.where(valid, positions, 0)
    group = pos // cr

    # --- store this step's compressed keys -----------------------------
    comp_flat = comp_cache.reshape(-1, 1, head_dim)
    store_slot = jnp.where(completing & (slot_r >= 0),
                           slot_r * comp_cap + group % comp_cap, 0)
    vals = jnp.where(completing[:, None, None], pooled_roped,
                     jnp.zeros_like(pooled_roped))
    new_comp_flat = comp_flat.at[store_slot].set(vals)
    new_comp = new_comp_flat.reshape(comp_cache.shape)

    # --- streaming top-k over visible groups ---------------------------
    visible_groups = jnp.minimum((pos + 1) // cr, seq_lens[req] // cr)
    visible_groups = jnp.where(valid, visible_groups, 0)
    max_visible = jnp.max(visible_groups)
    n_iter = (max_visible + _GROUP_CHUNK - 1) // _GROUP_CHUNK

    def body(i, state):
        best_v, best_i = state
        g0 = i * _GROUP_CHUNK
        g = g0 + jnp.arange(_GROUP_CHUNK)
        idx = jnp.clip(
            slot_r[:, None] * comp_cap + g[None, :], 0,
            new_comp_flat.shape[0] - 1)
        kg = new_comp_flat[idx][:, :, 0, :].astype(jnp.float32)
        scores = jnp.einsum("thd,tgd->thg", q_sel.astype(jnp.float32), kg)
        scores = jax.nn.relu(scores).sum(axis=1) / math.sqrt(head_dim)
        scores = jnp.where(g[None, :] < visible_groups[:, None], scores,
                           -jnp.inf)
        kk = min(_GROUP_CHUNK, block_topk)
        chunk_v, chunk_i = jax.lax.top_k(scores, kk)
        merged_v = jnp.concatenate((best_v, chunk_v), axis=-1)
        merged_i = jnp.concatenate((best_i, g0 + chunk_i), axis=-1)
        new_v, top_idx = jax.lax.top_k(merged_v, block_topk)
        new_i = jnp.take_along_axis(merged_i, top_idx, axis=-1)
        return new_v, new_i

    best_v = jnp.full((num_tokens, block_topk), -jnp.inf,
                      dtype=jnp.float32)
    best_i = jnp.zeros((num_tokens, block_topk), dtype=jnp.int32)
    best_v, best_i = jax.lax.fori_loop(0, n_iter, body, (best_v, best_i))
    blocks = jnp.where(best_v > -jnp.inf, best_i, -1)

    # --- expand blocks to token indices + append the open-group tail ----
    visible_tokens = pos + 1
    seq_r = seq_lens[req]
    expanded = blocks[:, :, None] * cr + jnp.arange(cr)[None, None, :]
    exp_valid = ((blocks >= 0)[:, :, None] &
                 (expanded < seq_r[:, None, None]))
    expanded = jnp.where(exp_valid, expanded,
                         -1).reshape(num_tokens, block_topk * cr)
    tail_start = (visible_tokens // cr) * cr
    tail = tail_start[:, None] + jnp.arange(cr - 1)[None, :]
    tail_valid = (jnp.arange(cr - 1)[None, :] <
                  (visible_tokens - tail_start)[:, None]) & \
                 (tail < seq_r[:, None])
    tail = jnp.where(tail_valid, tail, -1)
    selection = jnp.concatenate((expanded, tail), axis=-1)
    return new_comp, selection


def _qsa_sparse_attention(
    selection: jax.Array,  # [T, W_sel] int32 (-1 padded)
    q_main: jax.Array,  # [T, nH, D] bf16 (post-norm, post-rope)
    block_tables: jax.Array,  # [R, bt_w] int32
    token_to_req: jax.Array,  # [T] int32
    cache: jax.Array,  # [B, S, G, packing, D_pad]
    sm_scale: float,
    num_kv_heads: int,
    num_heads: int,
) -> jax.Array:
    """Chunked online-softmax attention over each row's selected tokens.

    The selected index sets are already causally valid (selection is bounded
    by the query position), so no additional causal mask is needed — only
    the ``-1`` padding mask. Unrolled over static ``_SEL_CHUNK`` columns.
    """
    T, n_heads, head_dim = q_main.shape
    W = selection.shape[1]
    pad = (-W) % _SEL_CHUNK
    if pad:
        selection = jnp.concatenate(
            (selection,
             jnp.full((T, pad), -1, jnp.int32)),
            axis=-1)
    B, S, G, packing, D_pad = cache.shape
    assert G * packing == 2 * num_kv_heads
    group_q = num_heads // num_kv_heads
    req = jnp.clip(token_to_req, 0, block_tables.shape[0] - 1)
    cache_flat = cache.reshape(B * S, G, packing, D_pad)

    q_f32 = q_main.astype(jnp.float32).reshape(T, num_kv_heads, group_q,
                                               head_dim)
    m = jnp.full((T, n_heads), -jnp.inf, dtype=jnp.float32)
    l = jnp.zeros((T, n_heads), dtype=jnp.float32)
    acc = jnp.zeros((T, n_heads, head_dim), dtype=jnp.float32)

    for c0 in range(0, selection.shape[1], _SEL_CHUNK):
        idx = selection[:, c0:c0 + _SEL_CHUNK]
        valid = idx >= 0
        safe = jnp.maximum(idx, 0)
        col = jnp.clip(safe // S, 0, block_tables.shape[1] - 1)
        page = jnp.clip(block_tables[req[:, None], col], 0, B - 1)
        slot = page * S + safe % S
        rows = cache_flat[slot]  # [T, C, G, packing, D_pad]
        flat_rows = rows.reshape(rows.shape[0], rows.shape[1],
                                 2 * num_kv_heads, D_pad)
        kc = flat_rows[..., :num_kv_heads, :head_dim].astype(jnp.float32)
        vc = flat_rows[..., num_kv_heads:, :head_dim].astype(jnp.float32)
        # scores[t, c, k, g] = q[t,k,g,:] . k[t,c,k,:]
        scores = jnp.einsum("tkgd,tckd->tckg", q_f32, kc) * sm_scale
        scores = scores.transpose(0, 2, 3, 1).reshape(
            T, n_heads, idx.shape[1])
        scores = jnp.where(valid[:, None, :], scores, -jnp.inf)
        m_new = jnp.maximum(m, scores.max(axis=-1))
        m_safe = jnp.where(jnp.isfinite(m_new), m_new, 0.0)
        alpha = jnp.exp(jnp.where(jnp.isfinite(m), m, -jnp.inf) - m_safe)
        p = jnp.exp(scores - m_safe[..., None])
        l = l * alpha + p.sum(axis=-1)
        p4 = p.reshape(T, num_kv_heads, group_q, idx.shape[1])
        acc = acc * alpha[..., None] + jnp.einsum(
            "tkgc,tckd->tkgd", p4, vc).reshape(T, n_heads, head_dim)
        m = m_new

    l = jnp.where(l > 0, l, 1.0)
    return (acc / l[:, :, None]).astype(q_main.dtype)


class Qwen4ExpQSAAttention(nn.Module, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch (TPU).

    Registered in vLLM's ``static_forward_context`` under ``<prefix>.attn``
    (matching the vLLM ``Attention`` naming convention) so the TPU runner
    allocates this layer's main paged KV cache; the indexer's raw-key and
    compressed-key state caches register as their own Mamba-state entries so
    the runner allocates per-request slots for them as well.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen4Exp QSA on TPU currently requires BF16")
        if getattr(config, "dual_chunk_attention_config", None) is not None:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support dual-chunk RoPE")
        if not getattr(config, "is_causal", True):
            raise NotImplementedError(
                "Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        tp_size = 1  # TP head sharding is applied by the runner below.
        try:
            from vllm.distributed import \
                get_tensor_model_parallel_world_size as _get_tp
            tp_size = max(1, _get_tp())
        except Exception:  # pragma: no cover - single-device init
            tp_size = 1
        if self.total_num_heads % tp_size or self.total_num_kv_heads % tp_size:
            raise ValueError(
                "QSA attention/KV heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if self.num_heads % self.num_kv_heads:
            raise ValueError("QSA query heads must divide into KV heads")
        self.head_dim = int(config.head_dim or
                            self.hidden_size // self.num_heads)
        self.head_size = self.head_dim  # runner-facing attribute name
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says else.
        self.attn_output_gate = True
        self.sliding_window = None
        self.kv_sharing_target_layer_name = None
        self.attn_type = AttentionType.DECODER
        self.layer_name = f"{prefix}.attn"

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )

        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

    def get_attn_backend(self) -> type:
        """TPU attention backend interface stub.

        The TPU runner builds the cache spec from this layer's attributes
        and does not dispatch through an attention backend; the standard
        TPU FLASH_ATTN backend is reported for interface compatibility.
        """
        from tpu_inference.layers.vllm.backends.flash_attn import \
            PallasAttentionBackend
        return PallasAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=vllm_config.model_config.dtype,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if positions.ndim == 2:
            raise NotImplementedError(
                "Qwen4Exp QSA on TPU does not support MRoPE (2-D positions)")

        md = _get_attention_metadata()
        if md is None or getattr(md, "block_tables", None) is None:
            # Profiling / metadata-less call: produce no output and leave
            # every cache untouched (the CUDA reference does the same).
            return torch.zeros_like(hidden_states)

        wrapper_ctx = get_vllm_model_wrapper_context()
        num_tokens = hidden_states.shape[0]
        cfg = self.indexer

        # --- torch: projections, norms, RoPE ---------------------------
        qkv, _ = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        q_gate = q_gate.view(num_tokens, self.num_heads, -1)
        q, gate = q_gate.chunk(2, dim=-1)
        q = q.reshape(num_tokens, -1)
        gate = gate.reshape(num_tokens, -1)
        q = self.q_norm(q.view(-1, self.num_heads,
                               self.head_dim)).view(num_tokens, -1)
        k = self.k_norm(k.view(-1, self.num_kv_heads,
                               self.head_dim)).view(num_tokens, -1)
        q, k = self.rotary_emb(positions, q, k)

        proj = cfg.project(hidden_states)
        q_sel = proj[..., :cfg.index_n_heads * cfg.index_head_dim].view(
            num_tokens, cfg.index_n_heads, cfg.index_head_dim)
        raw_keys = proj[..., cfg.index_n_heads * cfg.index_head_dim:].reshape(
            num_tokens, cfg.index_head_dim)
        q_sel = cfg.q_layernorm(q_sel)
        q_sel = apply_qsa_rope(self.rotary_emb, positions, q_sel)

        # --- jax: cache updates, pooling, selection, attention ---------
        main_idx = wrapper_ctx.layer_name_to_kvcache_index[self.layer_name]
        raw_idx = wrapper_ctx.layer_name_to_kvcache_index[
            cfg.raw_key_state.prefix]
        comp_idx = wrapper_ctx.layer_name_to_kvcache_index[
            cfg.compressed_key_state.prefix]
        main_cache = wrapper_ctx.kv_caches[main_idx]
        raw_cache = wrapper_ctx.kv_caches[raw_idx]
        comp_cache = wrapper_ctx.kv_caches[comp_idx]

        jq = jax_view(q.view(num_tokens, self.num_heads, self.head_dim))
        jk = jax_view(k.view(num_tokens, self.num_kv_heads, self.head_dim))
        jv = jax_view(v.view(num_tokens, self.num_kv_heads, self.head_dim))
        jq_sel = jax_view(q_sel)
        jraw = jax_view(raw_keys)
        jpos = jax_view(positions).astype(jnp.int32)

        qsl = md.query_start_loc.astype(jnp.int32)
        seq_lens = md.seq_lens.astype(jnp.int32)
        slots = md.mamba_state_indices.astype(jnp.int32)
        token_to_req = _token_to_req(qsl, num_tokens)
        valid = jnp.arange(num_tokens, dtype=jnp.int32) < qsl[-1]

        new_main = _qsa_update_main_cache(jk, jv, jpos, token_to_req, valid,
                                          seq_lens,
                                          md.block_tables.astype(jnp.int32),
                                          main_cache)
        new_raw, pooled, first_pos, completing = _qsa_update_raw_and_pool(
            raw_cache, jraw, jpos, token_to_req, valid, seq_lens, qsl, slots,
            cfg.compress_ratio)

        # --- torch: pooled-key norm + RoPE at the group's first position
        pooled_t = torch_view(pooled).view(num_tokens, 1, cfg.index_head_dim)
        pooled_t = cfg.k_layernorm(pooled_t)
        pooled_t = apply_qsa_rope(self.rotary_emb, torch_view(first_pos),
                                  pooled_t)
        completing_t = torch_view(completing).view(num_tokens, 1, 1)
        pooled_t = pooled_t * completing_t.to(pooled_t.dtype)
        jpooled = jax_view(pooled_t)

        new_comp, selection = _qsa_select_and_store(
            comp_cache, jpooled, completing, jq_sel, jpos, token_to_req,
            valid, seq_lens, slots, cfg.compress_ratio, cfg.block_topk,
            cfg.index_head_dim)
        attn_out = _qsa_sparse_attention(
            selection, jq, md.block_tables.astype(jnp.int32), token_to_req,
            main_cache, self.scaling, self.num_kv_heads, self.num_heads)

        wrapper_ctx.kv_caches[main_idx] = new_main
        wrapper_ctx.kv_caches[raw_idx] = new_raw
        wrapper_ctx.kv_caches[comp_idx] = new_comp

        # --- torch: output gate + o_proj -------------------------------
        flat_out = torch_view(attn_out).reshape(num_tokens, -1)
        flat_out = flat_out * torch.sigmoid(gate)
        return self.o_proj(flat_out)[0]
