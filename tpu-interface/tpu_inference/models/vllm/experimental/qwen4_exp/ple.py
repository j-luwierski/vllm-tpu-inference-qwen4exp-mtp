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
"""TPU (torchax) PLE (Position/Lexicon-Embedding) layer for Qwen4Exp.

Port of ``vllm/models/qwen4_exp/nvidia/ple_layer.py`` +
``vllm/models/qwen4_exp/common/ple.py`` (with the Triton kernels in
``nvidia/ops/ple.py``) to torch/JAX that torchax lowers to XLA.

Semantics follow the vLLM CUDA reference:

1. N-gram ids: for every n-gram order ``n = 2..ngram_size`` and head, hash
   the last ``n`` tokens (XOR of ``token * layer_multiplier[j]``) and map to
   a per-head prime-sized vocabulary row. Tokens outside their request or
   before the segment's last EOS hash as ``eos_token_id``. The math is the
   checkpoint-native host reference (``compute_ngram_ids`` CPU path in the
   vLLM checkout) reformulated for the TPU runner's flat token layout.
2. The n-gram rows look up the ``[padded_ngram_vocab, head_dim]`` embedding
   table (optionally FP8 with one global scale), project through
   ``kv_proj`` into per-stream keys ``[T, hc_count*hidden]`` and one shared
   value ``[T, hidden]``, gate with the sigmoid ``ple_gate`` product of two
   grouped Gemma norms, and add a dilated depthwise short convolution of the
   gated output back in.
3. The result is added to the multi-stream hidden state at the layer's
   ``ple_layer_ids`` positions.

TPU-specific engineering decisions (deliberate, none silent):

* The runner does not pass ``ngram_context`` to the model, so the last
  ``ngram_size - 1`` committed tokens per request live in a per-request
  context ring (Mamba-state slot) written every step. Reads use the
  pre-step functional state and the ring keeps
  ``ngram_context_len + num_spec`` entries, so speculative rollbacks cannot
  surface rejected tokens in the context window.
* The dilated conv state is the same kind of per-request ring holding the
  last ``(kernel_size - 1) * dilation`` conv inputs; same functional-read
  discipline applies.
* The segment (EOS) logic avoids ``cummax``: with ``ngram_size`` tiny, the
  EOS-in-window test gathers the at most ``ngram_size - 1`` window values
  directly, which is equivalent for the flat batch layout.
* FP8 embedding rows are dequantized after lookup with the checkpoint's
  single global scale, matching ``Qwen4ExpPLEFp8EmbeddingMethod``.
* KNOWN CONSTRAINT: the n-gram hash needs 64-bit integer wraparound.
  JAX with ``jax_enable_x64=False`` (torchax performance mode) silently
  downcasts int64 to int32, so the layer refuses to build unless x64 is
  enabled (torchax ``enable_accuracy_mode()``) — a 32-bit-limb hash
  reimplementation is the follow-up that removes the requirement.
"""

import math
from typing import Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import torch
from torch import nn
from torchax.interop import jax_view, torch_view
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from tpu_inference.logger import init_logger

logger = init_logger(__name__)


def _get_attention_metadata():
    """The TPU runner's AttentionMetadata for the current step, or None."""
    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        md = next(iter(md.values()), None)
    return md


def _token_to_req(query_start_loc: jax.Array, num_tokens: int) -> jax.Array:
    """Request index per flat token slot, ``[num_tokens]`` int32."""
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    return (query_start_loc[None, :-1].astype(jnp.int32) <=
            tok[:, None]).sum(axis=1, dtype=jnp.int32) - 1


class _PLEState(nn.Module, MambaBase):
    """Per-request PLE state ring (no checkpoint weights).

    State layout per slot: ``[capacity, 1, width]``; the size-1 axis absorbs
    the runner's Mamba sharding spec so the state stays TP-replicated,
    matching the TP-replicated PLE embedding and projections.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        capacity: int,
        width: int,
        dtype: torch.dtype,
        prefix: str,
    ) -> None:
        nn.Module.__init__(self)
        self.capacity = capacity
        self.width = width
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
        return ((self.capacity, 1, self.width), )

    def get_state_dtype(self) -> Tuple[torch.dtype]:
        return (self._dtype, )


class _GroupedNormWeight(nn.Module):
    """Checkpoint-shaped holder ``[C]`` for one grouped (1 + w) affine."""

    def __init__(self, num_elements: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_elements, dtype=dtype))


class Qwen4ExpNGramEmbedding(nn.Module):
    """Hash-based N-gram embedding table with checkpoint shard loading."""

    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PLE_LAYER_PRIME = 10007

    @classmethod
    def _splitmix64(cls, value: int) -> int:
        value = (value + cls._SPLITMIX_GAMMA) & cls._MASK64
        value = ((value ^ (value >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        value = ((value ^ (value >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (value ^ (value >> 31)) & cls._MASK64

    @staticmethod
    def _is_prime_64(value: int) -> bool:
        if value < 2:
            return False
        for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if value % prime == 0:
                return value == prime
        exponent = value - 1
        shifts = 0
        while exponent % 2 == 0:
            exponent //= 2
            shifts += 1
        for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
            if base % value == 0:
                continue
            witness = pow(base, exponent, value)
            if witness in (1, value - 1):
                continue
            for _ in range(shifts - 1):
                witness = pow(witness, 2, value)
                if witness == value - 1:
                    break
            else:
                return False
        return True

    @classmethod
    def _nth_prime_after(cls, start: int, count: int) -> int:
        prime = int(start)
        for _ in range(count):
            candidate = prime + 1
            if candidate <= 2:
                prime = 2
                continue
            if candidate % 2 == 0:
                candidate += 1
            while not cls._is_prime_64(candidate):
                candidate += 2
            prime = candidate
        return prime

    @classmethod
    def _make_layer_multipliers(
        cls,
        *,
        ngram_size: int,
        unigram_vocab_size: int,
        seed: int,
        ple_dense_layer_id: int,
    ) -> list:
        max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        base_seed = seed + cls._PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(ngram_size):
            value = base_seed + cls._SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (cls._splitmix64(value) % half_bound) + 1)
        return multipliers

    @classmethod
    def _make_vocab_layout(
        cls,
        *,
        ngram_vocab_size_base: int,
        ngram_heads: int,
        ple_dense_layer_id: int,
    ) -> Tuple[list, list, int]:
        sizes = []
        offsets = []
        offset = 0
        for local_head in range(ngram_heads):
            global_head = ple_dense_layer_id * ngram_heads + local_head
            size = cls._nth_prime_after(ngram_vocab_size_base - 1,
                                        global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        return sizes, offsets, offset

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        prefix: str,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0")
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        multipliers = self._make_layer_multipliers(
            ngram_size=self.ngram_size,
            unigram_vocab_size=self.unigram_vocab_size,
            seed=int(getattr(config, "seed", 1234)),
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        sizes, offsets, total_vocab_size = self._make_vocab_layout(
            ngram_vocab_size_base=int(config.ngram_vocab_size_base),
            ngram_heads=self.ngram_heads,
            ple_dense_layer_id=ple_dense_layer_id,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total_vocab_size + divisor - 1) //
                             divisor) * divisor
        self.org_vocab_size = padded_vocab_size

        # FP8 PLE checkpoints store the whole table quantized with one
        # global scale (see Qwen4ExpPLEFp8EmbeddingMethod in the reference).
        self.fp8 = False
        weight_dtype = params_dtype
        if quant_config is not None and getattr(quant_config, "get_name",
                                                lambda: "")() == "fp8":
            if getattr(quant_config, "is_checkpoint_fp8_serialized", False):
                self.fp8 = True
                weight_dtype = torch.float8_e4m3fn
        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=weight_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
        )
        self.weight_scale = None
        if self.fp8:
            scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
            scale.weight_loader = self._load_weight_scale
            self.register_parameter("weight_scale", scale)

    def _load_weight_scale(self, param: torch.Tensor,
                           loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight.reshape(param.shape).to(param.dtype))

    def dequantize(self, embeddings: torch.Tensor,
                   output_dtype: torch.dtype) -> torch.Tensor:
        if not self.fp8:
            return embeddings
        if self.weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its scale")
        return embeddings.to(output_dtype) * self.weight_scale.to(
            output_dtype)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> set:
        """Load hash buffers and checkpoint-split embedding rows."""
        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set = set()
        regular_weights: list = []
        shard_prefix = "ngram_embedding.shard_"
        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == \
                    "token_lookup":
                # Non-persistent runtime state rebuilt in __init__.
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if tuple(buffer.shape) != tuple(loaded_weight.shape):
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got "
                        f"{tuple(loaded_weight.shape)}")
                buffer.copy_(loaded_weight.to(buffer.device))
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix):-len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}")
                embedding = self.ngram_embedding
                shard_size = (self.org_vocab_size + self.split_ngram_parts -
                              1) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, self.org_vocab_size - checkpoint_start))
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard "
                        f"{shard_index}: expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}")
                self._load_embedding_shard(embedding.weight, loaded_weight,
                                           checkpoint_start)
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            from vllm.model_executor.models.utils import AutoWeightsLoader
            loaded.update(
                AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded

    def _load_embedding_shard(self, param: torch.Tensor,
                              loaded_weight: torch.Tensor,
                              checkpoint_start: int) -> None:
        """Copy the shard rows overlapping this rank's vocabulary range.

        Port of ``copy_ple_embedding_shard_``: the table stays
        vocab-parallel, so a checkpoint shard can span several TP ranks.
        """
        shard_indices = getattr(self.ngram_embedding, "shard_indices", None)
        if shard_indices is None:
            tp_start, tp_end = 0, param.shape[0]
        else:
            tp_start = shard_indices.org_vocab_start_index
            tp_end = shard_indices.org_vocab_end_index
        overlap_start = max(checkpoint_start, tp_start)
        overlap_end = min(checkpoint_start + loaded_weight.shape[0], tp_end)
        if overlap_start >= overlap_end:
            return
        source_start = overlap_start - checkpoint_start
        dest_start = overlap_start - tp_start
        row_count = overlap_end - overlap_start
        with torch.no_grad():
            param.data[dest_start:dest_start +
                       row_count].copy_(loaded_weight[source_start:
                                                      source_start +
                                                      row_count].to(
                                                          param.data.dtype))


def _grouped_norm(x: torch.Tensor, weight: torch.Tensor,
                  eps: float) -> torch.Tensor:
    """Grouped Gemma RMSNorm over the last dim; ``weight`` is ``[C]``.

    ``x`` is ``[T, HC, H]`` and each H-sized stream normalizes independently
    with the matching slice of the ``[HC*H]`` (1 + w) affine.
    """
    hc, h = x.shape[-2], x.shape[-1]
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    w = weight.float().view(hc, h)
    return (xf * torch.rsqrt(variance + eps) * (1.0 + w)).to(x.dtype)


def _ple_gate(key: torch.Tensor, value: torch.Tensor, hidden: torch.Tensor,
              norm_key_w: torch.Tensor, norm_query_w: torch.Tensor,
              norm_conv_w: torch.Tensor, eps: float,
              hc_count: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch port of the Triton ``ple_gate`` kernel.

    ``key`` ``[T, HC*H]``, ``value`` ``[T, H]`` (one shared value vector,
    broadcast across streams by the kernel), ``hidden`` ``[T, HC*H]``.
    Returns ``(gated, normed)`` both ``[T, HC*H]``.
    """
    h = hidden.shape[-1] // hc_count
    k = key.view(key.shape[0], hc_count, h)
    q = hidden.view(hidden.shape[0], hc_count, h)
    k_n = _grouped_norm(k, norm_key_w, eps)
    q_n = _grouped_norm(q, norm_query_w, eps)
    dot = (k_n.float() * q_n.float()).sum(dim=-1)  # [T, HC]
    d = dot / math.sqrt(h)
    magnitude = torch.sqrt(torch.clamp(d.abs(), min=1e-6))
    g = torch.sigmoid(torch.sign(d) * magnitude)  # [T, HC]
    gated = (g.to(value.dtype).unsqueeze(-1) *
             value.unsqueeze(1)).reshape(key.shape)  # [T, HC*H]
    normed = _grouped_norm(gated.view(gated.shape[0], hc_count, h),
                           norm_conv_w, eps).reshape(key.shape)
    return gated, normed


class Qwen4ExpPLELayer(nn.Module):
    """GPU-resident Qwen4Exp PLE layer, TPU (torchax) edition."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (int(ple_dense_layer_id) if
                                   ple_dense_layer_id is not None else
                                   int(layer_idx))
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size -
                               1) * self.short_conv_dilation
        num_spec = vllm_config.num_speculative_tokens
        self.num_spec_tokens = num_spec

        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            prefix=f"{prefix}.ple_embedding",
            quant_config=quant_config,
            params_dtype=model_config.dtype,
        )
        # The PLE cache is TP-replicated, so this merged projection is too.
        self.kv_proj = MergedColumnParallelLinear(
            int(config.ple_embed_dim),
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            params_dtype=model_config.dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
            disable_tp=True,
        )
        self.norm_key = _GroupedNormWeight(self.hc_hidden_size,
                                           model_config.dtype)
        self.norm_query = _GroupedNormWeight(self.hc_hidden_size,
                                             model_config.dtype)
        self.norm_conv = _GroupedNormWeight(self.hc_hidden_size,
                                            model_config.dtype)
        self.norm_eps = float(config.rms_norm_eps)
        # Kept as an nn.Conv1d purely for checkpoint weight naming/shape
        # ([C, 1, K] depthwise); the forward math is a jax dilated gather.
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)

        # The n-gram hash multiplies 64-bit checkpoints constants and
        # relies on int64 wraparound; with JAX x64 disabled jnp silently
        # downcasts int64 to int32 and the ids would be corrupt. Fail
        # closed instead (remediation: run with torchax accuracy mode /
        # jax_enable_x64, or reimplement the hash in 32-bit limbs).
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "Qwen4Exp PLE n-gram hashing requires 64-bit integer "
                "arithmetic. Enable jax_enable_x64 (e.g. torchax."
                "enable_accuracy_mode()) before loading the model, or "
                "reimplement the hash in 32-bit limbs; with x64 disabled "
                "the ids would be silently wrong.")

        self.ngram_context_len = max(int(config.ngram_size) - 1, 0)
        # Ring capacities include the speculative slack so rejected draft
        # writes cannot alias committed history (functional reads).
        self._conv_ring_cap = self.conv_state_len + num_spec + 1
        self._ctx_ring_cap = self.ngram_context_len + num_spec + 1
        self.conv_state = _PLEState(
            vllm_config=vllm_config,
            capacity=self._conv_ring_cap,
            width=self.hc_hidden_size,
            dtype=model_config.dtype,
            prefix=f"{prefix}.conv_state",
        )
        self.ngram_ring = _PLEState(
            vllm_config=vllm_config,
            capacity=max(1, self._ctx_ring_cap),
            width=1,
            dtype=torch.int32,
            prefix=f"{prefix}.ngram_ring",
        )
        # Wired lazily on first forward: after weight loading these
        # buffers turn into device-resident torchax tensors.
        self._head_sizes = None
        self._head_offsets = None

    # ------------------------------------------------------------------
    def _compute_ngram_ids(self, input_ids: jax.Array, positions: jax.Array,
                           token_to_req: jax.Array, valid: jax.Array,
                           seq_lens: jax.Array,
                           query_start_loc: jax.Array,
                           ctx_ring: jax.Array) -> jax.Array:
        """Flat-layout n-gram ids, ``[T, ngram_heads]`` int64.

        Reformulation of ``Qwen4ExpNGramEmbedding.compute_ngram_ids`` for
        the flat batch: the per-request context (last committed tokens)
        lives in the context ring, and EOS-segment validity is tested by
        gathering the at most ``ngram_size - 1`` window values instead of a
        cumulative max.
        """
        n = self.ngram_size
        heads = self.heads_per_ngram
        ring_cap = ctx_ring.shape[1]
        num_tokens = input_ids.shape[0]
        if self._head_sizes is None:
            sizes_all = jax_view(self.ngram_heads_vocab_sizes).astype(
                jnp.int64)
            offsets_all = jax_view(self.ngram_heads_offsets).astype(jnp.int64)
            self._head_sizes = sizes_all
            self._head_offsets = offsets_all
        req = jnp.clip(token_to_req, 0, seq_lens.shape[0] - 1)
        slot_r = self._ple_slots[req]
        nc_r = (seq_lens -
                (query_start_loc[1:] - query_start_loc[:-1]))[req]
        i_t = jnp.arange(num_tokens, dtype=jnp.int32) - query_start_loc[req]

        ring_flat = ctx_ring.reshape(-1)

        def value_at(k):
            """Token at t-k, ring value before the chunk, or EOS."""
            src_pos = nc_r + i_t - k  # logical position of the source
            in_batch = i_t - k >= 0
            batch_v = input_ids[jnp.clip(
                jnp.arange(num_tokens) - k, 0, num_tokens - 1)]
            ring_v = ring_flat[jnp.clip(slot_r * ring_cap + src_pos % ring_cap,
                                        0, ring_flat.shape[0] - 1)]
            v = jnp.where(in_batch, batch_v, ring_v)
            return jnp.where(src_pos >= 0, v, self.eos_token_id)

        vals = jax.vmap(lambda k: value_at(k).astype(jnp.int64))(
            jnp.arange(n))  # [n, T]
        # valid(k): no EOS in the window [t-k, t-1] = vals[1..k]
        is_eos = vals == self.eos_token_id  # [n, T]
        prefix_eos = jnp.cumsum(is_eos, axis=0)  # inclusive over j=0..k
        # window [t-k, t-1] excludes j=0: eos count in 1..k =
        # prefix_eos[k] - is_eos[0]
        eos_in_window = (prefix_eos - is_eos[0][None, :]) > 0  # [n, T]
        valid_shift = jnp.concatenate([
            jnp.ones((1, num_tokens), dtype=bool),
            ~eos_in_window[1:]
        ], axis=0)
        shifted = jnp.where(valid_shift, vals, self.eos_token_id)  # [n, T]

        # jax_view, not .numpy(): inside the jitted step these buffers are
        # device-resident torchax tensors.
        mult = jax_view(self.layer_multipliers).astype(jnp.int64)  # [n]
        mixed = shifted * mult[:, None]
        id_blocks = []
        for ngram in range(2, n + 1):
            acc = mixed[0]
            for j in range(1, ngram):
                acc = jnp.bitwise_xor(acc, mixed[j])
            start = (ngram - 2) * heads
            sizes = self._head_sizes[start:start + heads]
            offsets = self._head_offsets[start:start + heads]
            ids = jnp.remainder(acc[:, None], sizes[None, :]) + \
                offsets[None, :]
            id_blocks.append(ids)
        return jnp.concatenate(id_blocks, axis=-1).astype(jnp.int64)

    def _short_conv(self, conv_input: jax.Array, gated: jax.Array,
                    positions: jax.Array, token_to_req: jax.Array,
                    valid: jax.Array, seq_lens: jax.Array,
                    query_start_loc: jax.Array,
                    conv_ring: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Dilated depthwise causal conv with the per-request ring state."""
        cap = conv_ring.shape[1]
        num_tokens = conv_input.shape[0]
        channels = conv_input.shape[-1]
        req = jnp.clip(token_to_req, 0, seq_lens.shape[0] - 1)
        slot_r = self._ple_slots[req]
        nc_r = (seq_lens -
                (query_start_loc[1:] - query_start_loc[:-1]))[req]
        pos = jnp.where(valid, positions, 0)

        src_pos = pos[:, None] - jnp.arange(
            self.conv_kernel_size)[None, :] * self.short_conv_dilation
        # A tap is valid only inside the request's committed+chunked
        # history: positions before the request's start (src_pos < 0)
        # contribute zero, matching the reference's EOS-segment clamping.
        in_batch = src_pos >= nc_r[:, None]
        use_ring = (~in_batch) & (src_pos >= 0)
        batch_idx = jnp.clip(
            query_start_loc[req][:, None] + (src_pos - nc_r[:, None]), 0,
            num_tokens - 1)
        batch_vals = conv_input[batch_idx]  # [T, K, C]
        ring_idx = jnp.clip(slot_r[:, None] * cap + src_pos % cap, 0,
                            conv_ring.shape[0] * cap - 1)
        ring_vals = conv_ring.reshape(-1, 1, channels)[ring_idx][:, :, 0]
        vals = jnp.where(in_batch[..., None], batch_vals,
                         jnp.where(use_ring[..., None], ring_vals,
                                   jnp.zeros_like(ring_vals))).astype(
                                       jnp.float32)
        weight = self.conv1d.weight.squeeze(1).detach()
        weight = jax_view(weight).astype(jnp.float32)  # [C, K]
        out = jnp.einsum("tkc,ck->tc", vals, weight)
        delta = gated + jnp.where(valid[:, None], out.astype(gated.dtype),
                                  jnp.zeros_like(gated))

        write_ok = valid & (seq_lens[req] > 0) & (slot_r >= 0)
        ring_flat = conv_ring.reshape(-1, 1, channels)
        write_slot = jnp.where(write_ok, slot_r * cap + pos % cap, 0)
        write_vals = jnp.where(write_ok[:, None, None],
                               conv_input[:, None, :],
                               jnp.zeros_like(conv_input)[:, None, :])
        new_ring = ring_flat.at[write_slot].set(write_vals)
        return delta, new_ring.reshape(conv_ring.shape)

    # ------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor,
                input_ids: torch.Tensor) -> torch.Tensor:
        """Return the PLE delta to add to the multi-stream hidden state."""
        md = _get_attention_metadata()
        if md is None or getattr(md, "block_tables", None) is None:
            return torch.zeros_like(hidden_states)

        wrapper_ctx = get_vllm_model_wrapper_context()
        conv_idx = wrapper_ctx.layer_name_to_kvcache_index[
            self.conv_state.prefix]
        ring_idx = wrapper_ctx.layer_name_to_kvcache_index[
            self.ngram_ring.prefix]
        conv_ring = wrapper_ctx.kv_caches[conv_idx]
        ctx_ring = wrapper_ctx.kv_caches[ring_idx]

        num_tokens = hidden_states.shape[0]
        jids = jax_view(input_ids.reshape(-1)).astype(jnp.int32)
        # Logical positions come from the runner metadata, not a fresh
        # arange: speculative verify steps schedule multiple tokens per
        # request at consecutive positions.
        jpos = md.input_positions.astype(jnp.int32)
        qsl = md.query_start_loc.astype(jnp.int32)
        seq_lens = md.seq_lens.astype(jnp.int32)
        self._ple_slots = md.mamba_state_indices.astype(jnp.int32)
        token_to_req = _token_to_req(qsl, num_tokens)
        valid = jnp.arange(num_tokens, dtype=jnp.int32) < qsl[-1]

        # --- torch: embedding lookup (+dequant), projection, gate --------
        ngram_ids = self._compute_ngram_ids(jids, jpos, token_to_req, valid,
                                            seq_lens, qsl, ctx_ring)
        ids_t = torch_view(ngram_ids)
        # Vocab-parallel lookup through the module so TP-sharded rows map
        # correctly (mirrors the reference's ngram_embedding(ngram_ids)).
        embeddings = self.ngram_embedding(ids_t)
        embeddings = embeddings.reshape(num_tokens, -1)
        embeddings = self.ple_embedding.dequantize(
            embeddings, hidden_states.dtype)
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split([self.hc_hidden_size, self.hidden_size],
                              dim=-1)
        gated, conv_input = _ple_gate(key, value, hidden_states,
                                      self.norm_key.weight,
                                      self.norm_query.weight,
                                      self.norm_conv.weight, self.norm_eps,
                                      self.hc_count)

        # --- jax: dilated conv + ring writes ------------------------------
        jgated = jax_view(gated)
        jconv_in = jax_view(conv_input)
        delta, new_conv_ring = self._short_conv(jconv_in, jgated, jpos,
                                                token_to_req, valid, seq_lens,
                                                qsl, conv_ring)
        ctx_cap = ctx_ring.shape[1]
        write_ok = valid & (seq_lens[token_to_req] > 0) & (
            self._ple_slots[token_to_req] >= 0)
        ctx_flat = ctx_ring.reshape(-1, 1, 1)
        ctx_slot = jnp.where(
            write_ok,
            self._ple_slots[token_to_req] * ctx_cap +
            jpos % ctx_cap, 0)
        write_vals = jnp.where(
            write_ok[:, None], jids.astype(jnp.int32)[:, None],
            jnp.zeros((num_tokens, 1), dtype=jnp.int32))
        new_ctx_ring = ctx_flat.at[ctx_slot].set(
            write_vals[:, :, None]).reshape(ctx_ring.shape)
        wrapper_ctx.kv_caches[conv_idx] = new_conv_ring
        wrapper_ctx.kv_caches[ring_idx] = new_ctx_ring
        return torch_view(delta)
