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
"""Differential unit tests: torchax Qwen4Exp port vs vLLM eager references.

Everything here runs on CPU (jax CPU backend). The ground-truth
implementations come from two places:

* the installed vllm package (the common eager hyper-connection and the
  n-gram CPU fallback path),
* pure-torch reference functions copied verbatim from
  ``vllm/tests/models/qwen4_exp/{test_qsa_reference.py,test_ple.py}``
  (attributed inline) because tpu-inference CI has no vllm source tree.

These tests pin the port's math to the CUDA reference's own reference
math; they do not exercise the TPU runner, the Pallas kernels, or any
device-side plumbing.
"""

import math
import socket
from itertools import accumulate
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torchax
from torchax.interop import torch_view
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config

from tpu_inference.models.vllm.experimental.qwen4_exp.hyper_connection import (
    GatedResidual as TpuGatedResidual)
from tpu_inference.models.vllm.experimental.qwen4_exp.hyper_connection import (
    GroupedGemmaRMSNorm as TpuGroupedGemmaRMSNorm)
from tpu_inference.models.vllm.experimental.qwen4_exp.hyper_connection import (
    HyperConnectionConfig as TpuHCConfig)
from tpu_inference.models.vllm.experimental.qwen4_exp.ple import (
    Qwen4ExpPLELayer, _ple_gate)
from tpu_inference.models.vllm.experimental.qwen4_exp.qsa import (
    Qwen4ExpQSAAttention, _qsa_sparse_attention, _qsa_select_and_store,
    _qsa_update_main_cache, _token_to_req, apply_qsa_rope)

# ---------------------------------------------------------------------------
# Reference helpers copied from the vLLM test suite (attribution inline).
# ---------------------------------------------------------------------------
_NGRAM_MULTIPLIERS = (
    18_014_398_509_481_983,
    17_114_398_509_481_981,
    16_214_398_509_481_979,
    15_314_398_509_481_977,
)
_NGRAM_HEADS_VOCAB_SIZES = (
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
    163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227,
)
_NGRAM_EOS_TOKEN_ID = 251
_NGRAM_HEADS_PER_NGRAM = 8


def _reference_ngram_ids(input_ids, query_start_loc, ngram_context,
                         layer_multipliers, ngram_heads_vocab_sizes,
                         ngram_heads_offsets, eos_token_id, heads_per_ngram):
    """Copied from vllm tests/models/qwen4_exp/test_ple.py."""
    tokens = input_ids.tolist()
    starts = query_start_loc.tolist()
    contexts = ngram_context.tolist()
    multipliers = layer_multipliers.cpu()
    sizes = ngram_heads_vocab_sizes.cpu()
    offsets = ngram_heads_offsets.cpu()
    rows = []
    context_len = len(contexts[0])
    for req, (start, end) in enumerate(zip(starts, starts[1:])):
        history = contexts[req] + tokens[start:end]
        for pos in range(context_len, len(history)):
            shifted = [history[pos]]
            crossed_eos = False
            for shift in range(1, context_len + 1):
                token = eos_token_id if crossed_eos else history[pos - shift]
                shifted.append(token)
                crossed_eos |= token == eos_token_id
            row = []
            mixed = torch.tensor(shifted[0], dtype=torch.int64) * multipliers[0]
            for ngram_order in range(2, context_len + 2):
                shift = ngram_order - 1
                mixed ^= shifted[shift] * multipliers[shift]
                head_start = (ngram_order - 2) * heads_per_ngram
                for head in range(head_start, head_start + heads_per_ngram):
                    row.append(
                        torch.remainder(mixed, sizes[head]) + offsets[head])
            rows.append(torch.stack(row))
    return torch.stack(rows)


def _expand_qsa_indices_reference(block_indices, query_positions,
                                  sequence_lengths, compress_ratio,
                                  token_topk):
    """Copied from vllm tests/models/qwen4_exp/test_qsa_reference.py."""
    rows = block_indices.shape[0]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    offsets = torch.arange(compress_ratio, device=block_indices.device)
    blocks = block_indices.long()
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded,
        torch.full_like(expanded, -1)).reshape(rows, block_topk *
                                               compress_ratio)
    expanded = expanded[:, :token_topk]
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded, torch.full_like(expanded, -1))
    tail_offsets = torch.arange(compress_ratio - 1,
                                device=block_indices.device)
    visible_tokens = query_positions + 1
    tail_start = visible_tokens // compress_ratio * compress_ratio
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_count = (visible_tokens - tail_start).unsqueeze(1)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count) & (
        tail < sequence_lengths.unsqueeze(1))
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))
    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(output_width, device=result.device).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + output_width)
    return result.gather(
        1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


def _qsa_sparse_paged_attention_reference(q, k_cache, v_cache,
                                          logical_indices, block_table,
                                          token_to_req, softmax_scale):
    """Copied from vllm tests/models/qwen4_exp/test_qsa_reference.py."""
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            continue
        request = token_to_req[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = k_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        values = v_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float())
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities,
                                   values.float()).to(q.dtype)
    return output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tpu_env():
    """One-rank gloo world so vLLM parallel layers can be constructed."""
    import torch.distributed as dist
    import vllm.distributed as vd
    if not dist.is_initialized():
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        store = dist.TCPStore("127.0.0.1", port, 1, is_master=True)
        dist.init_process_group(backend="gloo",
                                rank=0,
                                world_size=1,
                                store=store)
        vd.init_distributed_environment(world_size=1,
                                        rank=0,
                                        distributed_init_method="env://",
                                        backend="gloo")
    yield


@pytest.fixture(scope="module")
def vllm_config(tpu_env, tmp_path_factory):
    """Minimal VllmConfig with parallel state initialized."""
    import vllm.distributed as vd
    dummy = tmp_path_factory.mktemp("dummy_model") / "dummy"
    dummy.mkdir()
    # A MoE model config: initialize_model_parallel creates the expert
    # parallel group only for MoE configs, and the qwen4_exp test modules
    # share one pytest process.
    (dummy / "config.json").write_text(
        """{
  "architectures": ["Qwen3MoeForCausalLM"],
  "model_type": "qwen3_moe",
  "hidden_size": 8,
  "num_hidden_layers": 1,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
  "head_dim": 4,
  "moe_intermediate_size": 8,
  "num_experts": 4,
  "num_experts_per_tok": 2,
  "shared_expert_intermediate_size": 8,
  "vocab_size": 32,
  "rms_norm_eps": 1e-6,
  "max_position_embeddings": 64
}""")
    model_config = ModelConfig(model=str(dummy),
                               skip_tokenizer_init=True,
                               dtype="bfloat16",
                               max_model_len=64)
    config = VllmConfig(model_config=model_config)
    # Keep the config context active for the whole test module: vLLM ops
    # (get_rope, linears) read the current config at construction time.
    with set_current_vllm_config(config):
        # Shared with the other qwen4_exp test modules in one pytest run.
        from vllm.distributed import parallel_state as _ps
        if _ps._TP is None:
            vd.initialize_model_parallel(tensor_model_parallel_size=1)
        yield config


def _build_hc_pair(hc_count, hidden_size, lowrank, seed):
    """A TPU GatedResidual and a common vLLM one sharing weights."""
    from vllm.models.qwen4_exp.common.hyperconnection import (
        GatedResidual as CommonGatedResidual, HyperConnectionConfig as
        CommonHCConfig)

    torch.manual_seed(seed)
    tpu_cfg = TpuHCConfig(hc_count=hc_count,
                          hidden_size=hidden_size,
                          params_dtype=torch.float32,
                          hc_lowrank=lowrank,
                          rms_norm_eps=1e-6,
                          hc_per_branch_norm=True)
    tpu_hc = TpuGatedResidual(tpu_cfg, prefix="t")
    common_cfg = CommonHCConfig(hc_count=hc_count,
                                hidden_size=hidden_size,
                                params_dtype=torch.float32,
                                hc_lowrank=lowrank,
                                rms_norm_eps=1e-6,
                                hc_per_branch_norm=True)
    common_hc = CommonGatedResidual(common_cfg)
    with torch.no_grad():
        common_hc.hc_norm.weight.copy_(tpu_hc.hc_norm.weight)
        common_hc.input_mix_weight_down.weight.copy_(
            tpu_hc.input_mix_weight_down.weight)
        common_hc.input_mix_weight_up.weight.copy_(
            tpu_hc.input_mix_weight_up.weight)
        common_hc.block_inject_weight.weight.copy_(
            tpu_hc.block_inject_weight.weight)
    return tpu_hc, common_hc


# ---------------------------------------------------------------------------
# Hyper-connection (Gated Residual)
# ---------------------------------------------------------------------------
def test_grouped_gemma_rmsnorm_matches_vllm(vllm_config):
    torch.manual_seed(0)
    tpu_norm = TpuGroupedGemmaRMSNorm(256,
                                      eps=1e-6,
                                      group_size=64,
                                      dtype=torch.float32)
    with torch.no_grad():
        tpu_norm.weight.normal_(0.0, 0.1)
    from vllm.models.qwen4_exp.common.hyperconnection import \
        GroupedGemmaRMSNorm as CommonNorm
    common_norm = CommonNorm(256, eps=1e-6, group_size=64, dtype=torch.float32)
    common_norm.weight = tpu_norm.weight
    x = torch.randn(3, 256)
    torch.testing.assert_close(tpu_norm(x), common_norm(x))


def test_gated_residual_mix_combine_matches_vllm(vllm_config):
    tpu_hc, common_hc = _build_hc_pair(hc_count=4,
                                       hidden_size=16,
                                       lowrank=8,
                                       seed=1)
    x = torch.randn(3, 64)
    block_output = torch.randn(3, 16)

    _, tpu_block_input, tpu_injection = tpu_hc.mix(x)
    common_block_input, residuals = common_hc.mix(x)
    torch.testing.assert_close(tpu_block_input, common_block_input)

    combined = tpu_hc.combine(x, block_output, tpu_injection)
    expected = common_hc.combine(block_output, residuals)
    torch.testing.assert_close(combined, expected)

    unit_combined = tpu_hc.combine(x, block_output, None)
    torch.testing.assert_close(unit_combined,
                               (x.unflatten(-1, (4, 16)) +
                                block_output.unsqueeze(-2)).flatten(-2))


def test_gated_residual_delayed_chain_matches_vllm(vllm_config):
    """The delayed-combine pipeline equals the eager per-block pipeline.

    The NVIDIA reference computes each block's injection during its own mix
    and fuses the combine into the next module's input norm; this test pins
    that scheme to the common (non-delayed) eager implementation with
    identical weights.
    """
    torch.manual_seed(7)
    hc1_tpu, hc1_common = _build_hc_pair(2, 16, 4, seed=11)
    hc2_tpu, hc2_common = _build_hc_pair(2, 16, 4, seed=12)
    mixer_tpu = TpuGatedResidual(TpuHCConfig(hc_count=2,
                                             hidden_size=16,
                                             params_dtype=torch.float32,
                                             hc_lowrank=4,
                                             rms_norm_eps=1e-6,
                                             hc_per_branch_norm=True),
                                 use_combine=False,
                                 prefix="f")
    mixer_common = _build_common_mixer(mixer_tpu)
    attn1 = torch.nn.Linear(16, 16)
    attn2 = torch.nn.Linear(16, 16)

    x = torch.randn(2, 32)
    # Common (non-delayed) pipeline.
    bi1, res1 = hc1_common.mix(x)
    a1 = attn1(bi1)
    h1 = hc1_common.combine(a1, res1)
    bi2, res2 = hc2_common.mix(h1)
    a2 = attn2(bi2)
    h2 = hc2_common.combine(a2, res2)
    sample_common, _ = mixer_common.mix(h2)
    # TPU (delayed) pipeline.
    _, bi1_tpu, inj1 = hc1_tpu.mix(x)
    a1_tpu = attn1(bi1_tpu)
    h1_tpu, bi2_tpu, inj2 = hc2_tpu.combine_and_mix(x, a1_tpu, inj1)
    a2_tpu = attn2(bi2_tpu)
    h2_tpu, sample_tpu, _ = mixer_tpu.combine_and_mix(h1_tpu, a2_tpu, inj2)

    torch.testing.assert_close(a1_tpu, a1)
    torch.testing.assert_close(h1_tpu, h1)
    torch.testing.assert_close(bi2_tpu, bi2)
    torch.testing.assert_close(a2_tpu, a2)
    torch.testing.assert_close(h2_tpu, h2)
    torch.testing.assert_close(sample_tpu, sample_common)


def _build_common_mixer(tpu_mixer):
    from vllm.models.qwen4_exp.common.hyperconnection import (
        GatedResidual as CommonGatedResidual, HyperConnectionConfig as
        CommonHCConfig)
    common = CommonGatedResidual(
        CommonHCConfig(hc_count=tpu_mixer.hc_count,
                       hidden_size=tpu_mixer.hidden_size,
                       params_dtype=torch.float32,
                       hc_lowrank=tpu_mixer.config.hc_lowrank,
                       rms_norm_eps=tpu_mixer.config.rms_norm_eps,
                       hc_per_branch_norm=True))
    with torch.no_grad():
        common.hc_norm.weight.copy_(tpu_mixer.hc_norm.weight)
        common.input_mix_weight_down.weight.copy_(
            tpu_mixer.input_mix_weight_down.weight)
        common.input_mix_weight_up.weight.copy_(
            tpu_mixer.input_mix_weight_up.weight)
    return common


# ---------------------------------------------------------------------------
# PLE gate
# ---------------------------------------------------------------------------
def test_ple_gate_matches_eager_reference():
    """Eager reference extracted from vllm test_ple.py
    test_fused_gate_correctness."""
    torch.manual_seed(3)
    hc, h = 2, 32
    kv = torch.randn(5, hc * h + h, dtype=torch.float32)
    key, value = kv[:, :hc * h], kv[:, hc * h:]
    hidden = torch.randn(5, hc * h, dtype=torch.float32)
    norm_key = torch.empty(hc * h).normal_(-0.1, 0.1)
    norm_query = torch.empty(hc * h).normal_(-0.1, 0.1)
    norm_conv = torch.empty(hc * h).normal_(-0.1, 0.1)

    def grouped_norm(inputs, weight):
        grouped = inputs.float().unflatten(-1, (hc, h))
        var = grouped.square().mean(dim=-1, keepdim=True)
        normalized = grouped * torch.rsqrt(var + 1e-6)
        return (normalized.flatten(-2) *
                (1.0 + weight.float())).to(inputs.dtype)

    gated, normed = _ple_gate(key, value, hidden, norm_key, norm_query,
                              norm_conv, 1e-6, hc)

    key_n = grouped_norm(key, norm_key).reshape(5, hc, h)
    query_n = grouped_norm(hidden, norm_query).reshape(5, hc, h)
    dot = (key_n * query_n).sum(dim=-1, keepdim=True)
    dot = (dot / math.sqrt(h))
    gate = torch.sigmoid(dot.sign() * dot.abs().clamp_min(1e-6).sqrt())
    expected_gated = (gate * value.unsqueeze(-2)).flatten(-2)
    expected_normed = grouped_norm(expected_gated, norm_conv)
    torch.testing.assert_close(gated, expected_gated)
    torch.testing.assert_close(normed, expected_normed)


# ---------------------------------------------------------------------------
# PLE n-gram ids (jax flat-layout port vs the segment reference)
# ---------------------------------------------------------------------------
def _make_ple_layer_stub(ngram_size):
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    layer.ngram_size = ngram_size
    layer.heads_per_ngram = _NGRAM_HEADS_PER_NGRAM
    layer.eos_token_id = _NGRAM_EOS_TOKEN_ID
    layer._head_sizes = None
    layer._head_offsets = None
    layer._ple_slots = None
    return layer


def _run_port_ngram_ids(ngram_size, scenario):
    """scenario: (query_lens, eos_offsets, contexts, first_token_id)."""
    query_lens, eos_offsets, contexts, first_token_id = scenario
    context_len = ngram_size - 1
    heads = _NGRAM_HEADS_PER_NGRAM
    num_heads = context_len * heads
    num_reqs = len(query_lens)
    num_tokens = sum(query_lens)

    multipliers = np.asarray(_NGRAM_MULTIPLIERS[:ngram_size], dtype=np.int64)
    sizes = np.asarray(_NGRAM_HEADS_VOCAB_SIZES[:num_heads], dtype=np.int64)
    offsets = np.zeros_like(sizes)
    offsets[1:] = np.cumsum(sizes)[:-1]

    input_ids = np.arange(first_token_id,
                          first_token_id + num_tokens,
                          dtype=np.int32)
    for off in eos_offsets:
        input_ids[off] = _NGRAM_EOS_TOKEN_ID

    # Sequence state: request r has committed context_len + nc_extra tokens
    # before its chunk; the ring holds the last context_len of them.
    ring_cap = context_len + 2
    ring = np.full((num_reqs * ring_cap, ), _NGRAM_EOS_TOKEN_ID,
                   dtype=np.int32)
    starts = [0]
    for length in query_lens:
        starts.append(starts[-1] + length)
    committed_ends = []
    logical_positions = []
    for req, length in enumerate(query_lens):
        committed = [100 + 7 * req + i for i in range(context_len)]
        for i, token in enumerate(committed):
            if req == 0 and i == 0:
                token = _NGRAM_EOS_TOKEN_ID
                committed[i] = token
            ring[req * ring_cap + i % ring_cap] = token
        committed_ends.append(context_len)
        for i in range(length):
            logical_positions.append(context_len + i)
    positions = jnp.asarray(logical_positions, dtype=jnp.int32)

    seq_lens = jnp.asarray([committed_ends[r] + query_lens[r]
                            for r in range(num_reqs)], dtype=jnp.int32)
    qsl = jnp.asarray(starts, dtype=jnp.int32)
    slots = jnp.asarray(np.arange(num_reqs, dtype=np.int32), dtype=jnp.int32)
    token_to_req = _token_to_req(qsl, num_tokens)
    valid = jnp.arange(num_tokens, dtype=jnp.int32) < qsl[-1]
    ctx_ring = jnp.asarray(ring.reshape(num_reqs, ring_cap, 1, 1),
                           dtype=jnp.int32)

    # x64 must be on for the hash (see ple.py's construction guard); this
    # test validates the math itself, deployment gating is separate.
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        layer = _make_ple_layer_stub(ngram_size)
        layer.layer_multipliers = torch_view(
            jnp.asarray(multipliers, dtype=jnp.int64))
        layer._head_sizes = jnp.asarray(sizes, dtype=jnp.int64)
        layer._head_offsets = jnp.asarray(offsets, dtype=jnp.int64)
        layer._ple_slots = slots
        ids = layer._compute_ngram_ids(
            jnp.asarray(input_ids, dtype=jnp.int32), positions, token_to_req,
            valid, seq_lens, qsl, ctx_ring)
    finally:
        jax.config.update("jax_enable_x64", previous)

    expected_context = []
    for req in range(num_reqs):
        committed = [100 + 7 * req + i for i in range(context_len)]
        if req == 0:
            committed[0] = _NGRAM_EOS_TOKEN_ID
        expected_context.append(committed)
    reference = _reference_ngram_ids(
        torch.tensor(input_ids, dtype=torch.int64),
        torch.tensor(starts, dtype=torch.int64),
        torch.tensor(expected_context, dtype=torch.int64),
        layer_multipliers=torch.tensor(multipliers, dtype=torch.int64),
        ngram_heads_vocab_sizes=torch.tensor(sizes, dtype=torch.int64),
        ngram_heads_offsets=torch.tensor(offsets, dtype=torch.int64),
        eos_token_id=_NGRAM_EOS_TOKEN_ID,
        heads_per_ngram=heads)
    return np.asarray(ids), reference.numpy(), sizes, offsets


@pytest.mark.parametrize("ngram_size", [2, 3, 4])
def test_ngram_ids_match_reference_single_request(ngram_size):
    scenario = ([5], [], [[10 + i for i in range(ngram_size - 1)]], 30)
    actual, expected, sizes, offsets = _run_port_ngram_ids(
        ngram_size, scenario)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    assert ((actual >= offsets) & (actual < offsets + sizes)).all()


def test_ngram_ids_match_reference_multi_request_with_eos():
    scenario = ([4, 3], [2], [[11, 12], [13, 14]], 20)
    actual, expected, sizes, offsets = _run_port_ngram_ids(3, scenario)
    np.testing.assert_array_equal(actual, expected)
    assert ((actual >= offsets) & (actual < offsets + sizes)).all()


def test_ngram_ids_match_reference_first_chunk_has_eos_context():
    """A request at its first chunk hashes the EOS-filled context."""
    scenario = ([6], [], [[11, 12]], 20)
    actual, expected, sizes, offsets = _run_port_ngram_ids(3, scenario)
    np.testing.assert_array_equal(actual, expected)
    assert ((actual >= offsets) & (actual < offsets + sizes)).all()


# ---------------------------------------------------------------------------
# QSA: rope, token-to-request mapping, cache, selection, attention
# ---------------------------------------------------------------------------
def test_apply_qsa_rope_matches_vllm_rotary_embedding():
    from vllm.model_executor.layers.rotary_embedding import get_rope
    rope = get_rope(head_size=32,
                    max_position=256,
                    rope_parameters={
                        "rope_theta": 10000.0,
                        "partial_rotary_factor": 0.5,
                    })
    torch.manual_seed(5)
    tokens, heads, head_dim = 6, 3, 32
    q = torch.randn(tokens, heads, head_dim, dtype=torch.float32)
    positions = torch.tensor([0, 1, 2, 17, 100, 255])

    reference, _ = rope(
        positions, q.reshape(tokens, -1),
        torch.zeros(tokens, heads * head_dim))
    actual = apply_qsa_rope(rope, positions, q)
    torch.testing.assert_close(actual, reference.reshape(tokens, heads,
                                                         head_dim))


def test_token_to_req_matches_searchsorted():
    qsl = jnp.asarray([0, 3, 3, 7, 7, 9], dtype=jnp.int32)
    num_tokens = 9
    actual = np.asarray(_token_to_req(qsl, num_tokens))
    expected = np.searchsorted(np.asarray(qsl), np.arange(num_tokens),
                               side="right") - 1
    np.testing.assert_array_equal(actual, expected)
    # Padded slots (t >= total) land on the last padded request row.
    padded = np.asarray(_token_to_req(qsl, 12))
    assert (padded[9:] == len(qsl) - 2).all()


def _build_qsa_world(seed, num_tokens=14, num_blocks=8, page_size=4):
    torch.manual_seed(seed)
    head_dim, num_heads, num_kv_heads = 16, 2, 1
    packing = 2
    cache = jnp.asarray(
        torch.zeros(num_blocks, page_size, 1, packing, head_dim)
        .numpy(), dtype=jnp.bfloat16)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.float32)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=torch.float32)
    k_j = jnp.asarray(k.numpy(), dtype=jnp.bfloat16)
    v_j = jnp.asarray(v.numpy(), dtype=jnp.bfloat16)
    positions = jnp.arange(num_tokens, dtype=jnp.int32)
    token_to_req = jnp.zeros(num_tokens, dtype=jnp.int32)
    valid = jnp.ones(num_tokens, dtype=bool)
    seq_lens = jnp.asarray([num_tokens], dtype=jnp.int32)
    block_tables = jnp.asarray(
        np.arange(num_blocks).reshape(1, num_blocks), dtype=jnp.int32)
    return (cache, k, v, k_j, v_j, positions, token_to_req, valid, seq_lens,
            block_tables, head_dim, num_heads, num_kv_heads)


def test_qsa_main_cache_write_read_roundtrip():
    (cache, k, v, k_j, v_j, positions, token_to_req, valid, seq_lens,
     block_tables, head_dim, num_heads, num_kv_heads) = _build_qsa_world(8)

    new_cache = _qsa_update_main_cache(k_j, v_j, positions, token_to_req,
                                       valid, seq_lens, block_tables, cache)

    rows = new_cache.reshape(-1, 1, 2, head_dim)
    read_k = np.asarray(rows[:k.shape[0], 0, 0, :].astype(jnp.float32))
    read_v = np.asarray(rows[:v.shape[0], 0, 1, :].astype(jnp.float32))
    # bf16 storage: compare to the bf16-rounded originals.
    np.testing.assert_array_equal(
        read_k, np.asarray(k_j.astype(jnp.float32)).reshape(-1, head_dim))
    np.testing.assert_array_equal(
        read_v, np.asarray(v_j.astype(jnp.float32)).reshape(-1, head_dim))


def test_qsa_selection_and_expansion_match_reference():
    compress_ratio, token_topk = 2, 8
    block_topk = token_topk // compress_ratio
    head_dim, index_heads = 16, 2
    comp_cap = 16
    torch.manual_seed(9)
    # Keys strictly aligned with the query direction: every dot is
    # positive and the scores increase strictly with the group index, so
    # top-k is unique (no tie-breaking involved). Block 0 is the reserved
    # null block; the request lives in block 1.
    q_dir = torch.nn.functional.normalize(torch.randn(head_dim), dim=0)
    keys = torch.stack([
        q_dir * scale + 0.01 * torch.randn(head_dim)
        for scale in torch.linspace(0.2, 3.0, comp_cap)
    ])
    bank = torch.zeros(2, comp_cap, 1, head_dim)
    bank[1] = keys.unsqueeze(1)
    q_sel = (q_dir.reshape(1, 1, head_dim) + 0.01 * torch.randn(
        3, index_heads, head_dim)).to(torch.bfloat16).float()
    positions = jnp.asarray([5, 9, 13], dtype=jnp.int32)
    seq_lens = jnp.asarray([14], dtype=jnp.int32)
    slots = jnp.asarray([1], dtype=jnp.int32)
    qsl = jnp.asarray([0, 3], dtype=jnp.int32)
    token_to_req = _token_to_req(qsl, 3)
    valid = jnp.ones(3, dtype=bool)

    _, selection = _qsa_select_and_store(
        jnp.asarray(bank.numpy(), dtype=jnp.bfloat16),
        jnp.zeros((3, 1, head_dim), dtype=jnp.bfloat16),
        jnp.zeros(3, dtype=bool),
        jnp.asarray(q_sel.numpy(), dtype=jnp.bfloat16), positions,
        token_to_req, valid, seq_lens, slots, compress_ratio, block_topk,
        head_dim)
    selection = np.asarray(selection)

    # Reference: torch scores over the same bank, torch.topk per row,
    # then the vLLM reference expansion.
    scores = torch.relu(
        torch.einsum("thd,gd->thg", q_sel,
                     bank[1, :, 0, :])).sum(dim=1) / math.sqrt(head_dim)
    reference = torch.full((3, block_topk), -1, dtype=torch.int32)
    for row, position in enumerate([5, 9, 13]):
        visible = min((position + 1) // compress_ratio,
                      int(seq_lens[0]) // compress_ratio)
        masked = scores[row].clone()
        masked[visible:] = -float("inf")
        reference[row] = torch.topk(masked, block_topk).indices.to(
            torch.int32)
    expected = _expand_qsa_indices_reference(
        reference, torch.tensor([5, 9, 13]), torch.tensor([14]),
        compress_ratio, token_topk)

    # top-k tie-breaking is implementation-defined (torch.topk, jax
    # top_k and the CUDA Triton kernel each break exact ties their own
    # way), so pin the mathematically meaningful invariants instead of
    # exact index sets: same selection count, causally-valid groups,
    # and the worst selected score equals the block_topk-th largest
    # visible score on both sides.
    for row, position in enumerate([5, 9, 13]):
        mine = sorted(int(i) for i in selection[row] if i >= 0)
        ref = sorted(int(i) for i in expected[row] if i >= 0)
        assert len(mine) == len(ref), (row, mine, ref)
        visible = min((position + 1) // compress_ratio,
                      int(seq_lens[0]) // compress_ratio)
        my_groups = {idx // compress_ratio for idx in mine}
        assert all(group < visible for group in my_groups), (row, my_groups)
        if visible >= block_topk:
            threshold = float(
                torch.kthvalue(-scores[row, :visible],
                               block_topk).values.abs())
        else:
            # Fewer visible groups than k: everything visible is selected,
            # so the worst selected score is the smallest visible one.
            threshold = float(scores[row, :visible].min())
        mine_min = float(scores[row, torch.tensor(mine)].min())
        ref_min = float(scores[row, torch.tensor(ref)].min())
        assert mine_min >= threshold - 1e-3, (row, mine_min, threshold)
        assert ref_min >= threshold - 1e-3, (row, ref_min, threshold)
        # With strictly increasing aligned scores the selection is unique
        # and must match the reference exactly.
        assert mine == ref, (row, mine, ref)


def test_qsa_sparse_attention_matches_reference():
    (cache, k, v, k_j, v_j, positions, token_to_req, valid, seq_lens,
     block_tables, head_dim, num_heads, num_kv_heads) = _build_qsa_world(10)
    cache = _qsa_update_main_cache(k_j, v_j, positions, token_to_req, valid,
                                   seq_lens, block_tables, cache)

    torch.manual_seed(11)
    scale = head_dim**-0.5
    q_main = torch.randn(3, num_heads, head_dim)
    q_j = jnp.asarray(q_main.numpy(), dtype=jnp.bfloat16)
    torch.manual_seed(12)
    q_dir = torch.nn.functional.normalize(torch.randn(head_dim), dim=0)
    keys = torch.stack([
        q_dir * scale + 0.01 * torch.randn(head_dim)
        for scale in torch.linspace(0.2, 3.0, 16)
    ])
    bank = torch.zeros(2, 16, 1, head_dim)
    bank[1] = keys.unsqueeze(1)
    q_sel = (q_dir.reshape(1, 1, head_dim) + 0.01 * torch.randn(
        3, 2, head_dim)).to(torch.bfloat16).float()
    sel_positions = jnp.asarray([5, 9, 13], dtype=jnp.int32)
    qsl = jnp.asarray([0, 3], dtype=jnp.int32)
    t2r = _token_to_req(qsl, 3)
    ones = jnp.ones(3, dtype=bool)
    slots = jnp.asarray([1], dtype=jnp.int32)  # 0 = null block
    _, selection = _qsa_select_and_store(
        jnp.asarray(bank.numpy(), dtype=jnp.bfloat16),
        jnp.zeros((3, 1, head_dim), dtype=jnp.bfloat16),
        jnp.zeros(3, dtype=bool), jnp.asarray(
            q_sel.numpy(), dtype=jnp.bfloat16), sel_positions, t2r, ones,
        seq_lens, slots, 2, 4, head_dim)

    out = _qsa_sparse_attention(selection, q_j, block_tables, t2r, cache,
                                scale, num_kv_heads, num_heads)
    out_np = np.asarray(out).astype(np.float32)

    # Reference attention over the same selection, reading the same cache.
    cache_t = torch.from_numpy(
        np.asarray(cache.astype(jnp.float32)))
    q_flat = torch.from_numpy(np.asarray(q_j.astype(jnp.float32)))
    k_cache = cache_t[:, :, 0:1, 0, :head_dim]
    v_cache = cache_t[:, :, 0:1, 1, :head_dim]
    reference = _qsa_sparse_paged_attention_reference(
        q_flat, k_cache, v_cache,
        torch.from_numpy(np.asarray(selection)), torch.from_numpy(
            np.asarray(block_tables)), torch.zeros(3, dtype=torch.long),
        scale)
    torch.testing.assert_close(
        torch.from_numpy(out_np), reference, atol=5e-2, rtol=5e-2)


# ---------------------------------------------------------------------------
# PLE dilated short convolution
# ---------------------------------------------------------------------------
def test_short_conv_matches_naive_reference_and_updates_ring():
    torch.manual_seed(13)
    channels, kernel_size, dilation = 6, 3, 2
    cap = kernel_size * dilation
    num_tokens, num_reqs = 6, 2
    # The port runs on bf16 activations; quantize the reference inputs the
    # same way so the comparison only measures accumulation order.
    conv_input = torch.randn(num_tokens, channels).to(torch.bfloat16).float()
    gated = torch.randn(num_tokens, channels).to(torch.bfloat16).float()
    weight = torch.randn(channels, 1, kernel_size).to(torch.bfloat16).float()
    ring = torch.randn(num_reqs * cap, 1, channels)  # pre-step ring state
    ring_j = jnp.asarray(ring.numpy(), dtype=jnp.bfloat16).reshape(
        num_reqs, cap, 1, channels)
    conv_input_j = jnp.asarray(conv_input.numpy(), dtype=jnp.bfloat16)
    gated_j = jnp.asarray(gated.numpy(), dtype=jnp.bfloat16)

    # Request 0: chunk at logical positions 2..5 (nc=2). Request 1: nc=0.
    qsl = jnp.asarray([0, 4, 6], dtype=jnp.int32)
    seq_lens = jnp.asarray([6, 2], dtype=jnp.int32)
    positions = jnp.asarray([2, 3, 4, 5, 0, 1], dtype=jnp.int32)
    token_to_req = jnp.asarray([0, 0, 0, 0, 1, 1], dtype=jnp.int32)
    valid = jnp.ones(num_tokens, dtype=bool)
    slots = jnp.asarray([0, 1], dtype=jnp.int32)

    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    layer.conv_kernel_size = kernel_size
    layer.short_conv_dilation = dilation
    layer._ple_slots = slots
    layer.conv1d = SimpleNamespace(weight=torch_view(
        jnp.asarray(weight.numpy(), dtype=jnp.bfloat16).reshape(
            channels, 1, kernel_size)))

    # The runner always executes the step function inside the torchax
    # environment; mirror that here for the weight-view conversion.
    with torchax.default_env():
        delta, new_ring = layer._short_conv(conv_input_j, gated_j, positions,
                                            token_to_req, valid, seq_lens,
                                            qsl, ring_j)

    # Naive reference: out[t] = sum_k w[c,k] * x[pos - k*dilation], with
    # in-batch sources for positions >= nc and ring sources before that.
    ring_np = ring.reshape(num_reqs, cap, channels).numpy()
    delta_np = np.asarray(delta.astype(jnp.float32))
    conv_input_np = conv_input.numpy()
    for t in range(num_tokens):
        req = int(token_to_req[t])
        nc = int(seq_lens[req]) - (
            int(qsl[req + 1]) - int(qsl[req]))
        pos = int(positions[t])
        for c in range(channels):
            acc = 0.0
            for k_idx in range(kernel_size):
                src = pos - k_idx * dilation
                if src >= nc:
                    value = float(conv_input_np[int(qsl[req]) + src - nc, c])
                elif src >= 0:
                    value = float(ring_np[req, src % cap, c])
                else:
                    value = 0.0
                acc += float(weight[c, 0, k_idx]) * value
            expected = float(gated[t, c]) + acc
            # bf16 output rounding: relative tolerance on the total.
            assert abs(float(delta_np[t, c]) - expected) <= 0.02 * abs(
                expected) + 0.05

    # The ring must now hold the last `cap` conv inputs per request.
    new_ring_np = np.asarray(new_ring.astype(jnp.float32)).reshape(
        num_reqs, cap, channels)
    for req in range(num_reqs):
        nc = int(seq_lens[req]) - (int(qsl[req + 1]) - int(qsl[req]))
        end = int(seq_lens[req])
        # Modular ring: slot s holds the newest token with pos % cap == s.
        # Slots of positions < 0 were never written and keep the pre-step
        # values, exactly like the port's functional read model.
        for slot in range(cap):
            position = None
            for p in range(end - cap, end):
                if p % cap == slot:
                    position = p
                    break
            if position is not None and position >= nc:
                expected = conv_input_np[int(qsl[req]) + position - nc, :]
            else:
                expected = ring_np[req, slot, :]
            np.testing.assert_allclose(new_ring_np[req, slot, :],
                                       expected.astype(np.float32),
                                       atol=0.05)
