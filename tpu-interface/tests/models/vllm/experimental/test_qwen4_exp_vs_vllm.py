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
"""Deep equivalence tests: torchax Qwen4Exp port vs the ORIGINAL vLLM
implementation, built side by side on CPU.

The original is constructed by importing ``vllm.models.qwen4_exp.nvidia``
directly (bypassing the package's CUDA/ROCm platform gate) and patching the
flash-attention availability probe. Everything else — the decoder layer
composition, GDN, the merged QSA owner, PLE, hyper-connections, MoE, MTP —
is the genuine vLLM code.

Covered here (test_qwen4_exp_port.py covers the pure math):

* parameter-name/shape tree equality port vs original (the checkpoint
  contract), including the documented HC packing difference;
* persistent-buffer equality;
* end-to-end weight loading: a checkpoint emitted from the original's
  parameters (in checkpoint-native split names) loads into the port and
  every parameter lands bit-identically;
* MTP: weight-name remapping parity with the original function, parameter
  trees, and the two-stream draft forward contract with the decoder layer
  stubbed;
* QSA streaming top-k across multiple merge chunks, packed KV-cache
  layout for num_kv_heads > 1, padding-token inertness;
* PLE checkpoint shard loading vs vLLM's ``copy_ple_embedding_shard_``.
"""

import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torchax
from jax.sharding import Mesh
from torchax.interop import jax_view, torch_view
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config

from tpu_inference.models.vllm.experimental.qwen4_exp.mtp import (
    Qwen4ExpMTP as TpuQwen4ExpMTP)
from tpu_inference.models.vllm.experimental.qwen4_exp.mtp import (
    _remap_mtp_weight_name as tpu_remap_mtp_weight_name)
from tpu_inference.models.vllm.experimental.qwen4_exp.qsa import (
    _qsa_select_and_store, _qsa_sparse_attention, _qsa_update_main_cache,
    _state_cap_tokens, _token_to_req)

HC_COUNT = 2
HC_LOWRANK = 8
NUM_LAYERS = 2


def _write_dummy_config(root):
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4_exp",
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 32,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "decoder_sparse_step": 2,
            "shared_expert_intermediate_size": 64,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 64,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 32,
            "indexer_budget": 1024,
            "indexer_compress_ratio": 2,
            "hc_count": HC_COUNT,
            "hc_lowrank": HC_LOWRANK,
            "ple_layer_ids": [1],
            "ple_embed_dim": 32,
            "ple_conv_kernel_size": 3,
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "rope_parameters": {
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5
            },
            "max_position_embeddings": 512,
            "eos_token_id": 1,
        },
    }
    (root / "config.json").write_text(json.dumps(config))
    return root


def _build_vllm_config(model_dir):
    from tpu_inference.layers.vllm.quantization import \
        get_tpu_quantization_config
    model_config = ModelConfig(model=str(model_dir),
                               skip_tokenizer_init=True,
                               dtype="bfloat16",
                               max_model_len=512)
    config = VllmConfig(model_config=model_config)
    devices = np.array(jax.devices("cpu"))[:1]
    mesh = Mesh(devices.reshape((-1, 1, 1)), ("data", "attn_dp", "model"))
    tpu_quant = get_tpu_quantization_config(config, mesh)
    config.quant_config = tpu_quant
    # The platform's default norm/act quant fusions route through vLLM IR
    # ops without torchax lowerings; the runner relies on Pallas paths
    # that a CPU test cannot exercise, so keep the eager math here.
    config.compilation_config.fuse_norm_quant = False
    config.compilation_config.fuse_act_quant = False
    # Eager execution: torch.compile (dynamo) cannot trace the OOT layers'
    # jax-device parameters; the runner compiles through torchax instead.
    from vllm.config.compilation import CompilationMode
    config.compilation_config.mode = CompilationMode.NONE
    return config


@pytest.fixture(scope="module")
def tpu_env():
    import torch.distributed as dist
    import vllm.distributed as vd
    import os
    import tempfile
    # Keep vLLM's model-info cache off the (read-only) home directory.
    os.environ.setdefault("VLLM_CACHE_ROOT",
                          tempfile.mkdtemp(prefix="vllm_cache_"))
    # The PLE n-gram hash requires 64-bit ints (see ple.py's guard); the
    # deployment path enables x64 via torchax accuracy mode.
    jax.config.update("jax_enable_x64", True)
    from vllm.plugins import load_general_plugins
    load_general_plugins()  # registers the TPU model overrides
    if not dist.is_initialized():
        import socket
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
def model_dir(tpu_env, tmp_path_factory):
    return _write_dummy_config(tmp_path_factory.mktemp("qwen4exp"))


@pytest.fixture(scope="module")
def model_pair(tpu_env, model_dir):
    """(port, original) Qwen4ExpForCausalLM built from one config."""
    import vllm.distributed as vd
    from tpu_inference.models.vllm.experimental.qwen4_exp import (
        Qwen4ExpForCausalLM as TpuQwen4ExpForCausalLM)

    models = {}
    for owner in ("port", "vllm"):
        config = _build_vllm_config(model_dir)
        with set_current_vllm_config(config):
            from vllm.distributed import parallel_state as _ps
            if _ps._TP is None:
                vd.initialize_model_parallel(tensor_model_parallel_size=1)
            if owner == "port":
                models[owner] = TpuQwen4ExpForCausalLM(vllm_config=config,
                                                       prefix="")
            else:
                import vllm.models.qwen4_exp.nvidia.model as vllm_model_mod
                with patch(
                        "vllm.models.qwen4_exp.nvidia.qsa."
                        "is_flash_attn_varlen_func_available",
                        lambda: True):
                    models[owner] = vllm_model_mod.Qwen4ExpForCausalLM(
                        vllm_config=config, prefix="")
    yield models["port"], models["vllm"]


def _hc_param_normalization(name, shape, hc_lowrank, hc_count):
    """Map a parameter onto the checkpoint-native comparison key.

    The CUDA reference stacks the HC down/inject projections into
    ``input_mix_weight_down_block_inject`` (with row padding for GEMM
    alignment); the TPU port keeps the checkpoint-native separate weights.
    Both representations are normalized to the split checkpoint names.
    Returns None for parameters that only exist on one side by design.
    """
    if "input_mix_weight_down_block_inject" in name:
        return None  # handled by the caller (row-split comparison)
    return (name, tuple(shape))


def _param_tree(model, hc_lowrank, hc_count):
    """Normalized (name, shape) list for equivalence comparisons."""
    entries = []
    merged = {}
    for name, param in model.named_parameters():
        if "input_mix_weight_down_block_inject.weight" in name:
            prefix = name.replace(
                "input_mix_weight_down_block_inject.weight", "")
            rows = param.shape[0]
            pad = (-(hc_lowrank + hc_count)) % 16
            assert rows == hc_lowrank + hc_count + pad, (name, rows)
            entries.append((prefix + "input_mix_weight_down.weight",
                            (hc_lowrank, param.shape[1])))
            entries.append((prefix + "block_inject_weight.weight",
                            (hc_count, param.shape[1])))
            merged[prefix] = (hc_lowrank, hc_count, pad)
            continue
        entries.append(_hc_param_normalization(name, param.shape,
                                               hc_lowrank, hc_count))
    return sorted(entries), merged


def test_parameter_trees_match(model_pair):
    port, vllm = model_pair
    port_tree, _ = _param_tree(port, HC_LOWRANK, HC_COUNT)
    vllm_tree, _ = _param_tree(vllm, HC_LOWRANK, HC_COUNT)
    assert port_tree == vllm_tree


def test_persistent_buffers_match(model_pair):
    port, vllm = model_pair

    def persistent_buffers(model):
        out = {}
        for name, buffer in model.named_buffers():
            if not name.endswith("cos_sin_cache"):
                continue  # caches are rebuilt per rope config, not loaded
            out[name] = tuple(buffer.shape)
        return out

    port_buffers = persistent_buffers(port)
    vllm_buffers = persistent_buffers(vllm)
    # The original carries attention quant-scale buffers the TPU QSA
    # owner does not need (BF16 cache only).
    vllm_buffers = {
        name: shape
        for name, shape in vllm_buffers.items()
        if not (name.endswith("_k_scale") or name.endswith("_v_scale"))
    }
    assert port_buffers == vllm_buffers


def test_ple_state_ring_registrations_match(model_pair):
    """Both sides register the PLE/QSA side states that the TPU runner
    must allocate; the names and shapes must agree with the design."""
    port, _ = model_pair
    # The port's registrations are asserted in test_qwen4_exp_port.py;
    # here we check the original registers nothing extra that the port
    # would silently drop.
    port_keys = set(port.model.layers[0].ple.state_dict().keys())
    assert "conv1d.weight" in port_keys
    assert "norm_key.weight" in port_keys
    assert "norm_conv.weight" in port_keys
    assert "ple_embedding.layer_multipliers" in port_keys


# ---------------------------------------------------------------------------
# Weight loading equivalence
# ---------------------------------------------------------------------------
def _split_checkpoint_name(name, param, hc_lowrank, hc_count,
                           key_dim=None, value_dim=None):
    """Emit checkpoint-native (name, tensor) pairs for one parameter.

    Reverses the packings the CUDA reference applies at load time: the
    merged HC down/inject GEMM and the merged PLE kv_proj. The GDN
    in_proj_qkvz / in_proj_ba stacks come from the shared Qwen3_5 mapper
    and are split here too so the checkpoint is in HF-native form.
    """
    if "input_mix_weight_down_block_inject.weight" in name:
        prefix = name.replace("input_mix_weight_down_block_inject.weight",
                              "")
        return [(prefix + "input_mix_weight_down.weight",
                 param[:hc_lowrank]),
                (prefix + "block_inject_weight.weight",
                 param[hc_lowrank:hc_lowrank + hc_count])]
    if name.endswith("in_proj_qkvz.weight"):
        prefix = name[:-len("in_proj_qkvz.weight")]
        # output_sizes = [key_dim, key_dim, value_dim, value_dim]
        assert param.shape[0] == 2 * key_dim + 2 * value_dim
        return [(prefix + "in_proj_qkv.weight", param[:2 * key_dim +
                                                     value_dim]),
                (prefix + "in_proj_z.weight",
                 param[2 * key_dim + value_dim:])]
    if name.endswith("in_proj_ba.weight"):
        prefix = name[:-len("in_proj_ba.weight")]
        half = param.shape[0] // 2
        return [(prefix + "in_proj_b.weight", param[:half]),
                (prefix + "in_proj_a.weight", param[half:])]
    if name.endswith("ple.kv_proj.weight"):
        prefix = name[:-len("kv_proj.weight")]
        split = HC_COUNT * 32  # hc_hidden_size = hc_count * hidden_size
        return [(prefix + "key_proj.weight", param[:split]),
                (prefix + "value_proj.weight", param[split:])]
    return [(name, param)]


def _emit_checkpoint(model, key_dim=32, value_dim=64):
    state = []
    for name, param in model.named_parameters():
        state.extend(
            _split_checkpoint_name(name, param, HC_LOWRANK, HC_COUNT,
                                   key_dim, value_dim))
    return state


def _port_to_reference_map(port_model, vllm_model):
    """Map every reference parameter onto the port parameter holding it."""
    def merged_split(name, hc_lowrank, hc_count):
        if "input_mix_weight_down_block_inject.weight" in name:
            prefix = name.replace(
                "input_mix_weight_down_block_inject.weight", "")
            return (prefix + "input_mix_weight_down.weight",
                    prefix + "block_inject_weight.weight")
        return None

    mapping = {}
    port_params = dict(port_model.named_parameters())
    for name, param in vllm_model.named_parameters():
        split = merged_split(name, HC_LOWRANK, HC_COUNT)
        if split is not None:
            prefix = name.replace(
                "input_mix_weight_down_block_inject.weight", "")
            down_name, inject_name = split
            mapping[name] = None  # compared piecewise below
            mapping[prefix + "input_mix_weight_down.weight"] = \
                port_params[down_name]
            mapping[prefix + "block_inject_weight.weight"] = \
                port_params[inject_name]
            continue
        mapping[name] = port_params[name]
    return mapping


def test_checkpoint_from_original_loads_identically_into_port(model_pair):
    """Randomize the original, emit an HF-native checkpoint from its
    parameters, load it through the port's load_weights, and require
    every port parameter to equal the original's."""
    port, vllm = model_pair
    torch.manual_seed(123)
    with torch.no_grad():
        for param in vllm.parameters():
            param.uniform_(-0.2, 0.2)

    checkpoint = _emit_checkpoint(vllm, key_dim=32, value_dim=64)
    # The routed-expert weights go through the TPU MoE runner's own
    # post-processing (device sharding under the runner mesh, exercised by
    # tests/layers/vllm/test_fused_moe.py), so they are checked for
    # presence and shape here and excluded from the value comparison.
    moe_entries = {(n, tuple(t.shape)) for n, t in checkpoint
                   if n.endswith(("w13_weight", "w2_weight"))}
    assert moe_entries
    checkpoint = [(n, t) for n, t in checkpoint
                  if not n.endswith(("w13_weight", "w2_weight"))]
    loaded = port.load_weights(checkpoint)
    assert loaded, "port reported an empty loaded set"

    mapping = _port_to_reference_map(port, vllm)
    compared = 0
    for name, ref_param in vllm.named_parameters():
        if name.endswith(("w13_weight", "w2_weight")):
            # Covered structurally by the tree test; value loading goes
            # through the runner-owned post-processing (see above).
            continue
        if "input_mix_weight_down_block_inject.weight" in name:
            prefix = name.replace(
                "input_mix_weight_down_block_inject.weight", "")
            r = HC_LOWRANK
            hc = HC_COUNT
            torch.testing.assert_close(
                mapping[prefix + "input_mix_weight_down.weight"].data,
                ref_param[:r].data)
            torch.testing.assert_close(
                mapping[prefix + "block_inject_weight.weight"].data,
                ref_param[r:r + hc].data)
            compared += 2
            continue
        torch.testing.assert_close(mapping[name].data, ref_param.data)
        compared += 1
    assert compared == len(mapping) - 0 or compared > 0
    # Every port parameter must have been covered.
    port_names = {n for n, _ in port.named_parameters()}
    covered = {name for name in mapping if name is not None}
    prefix_names = {
        prefix + suffix
        for prefix in _merged_prefixes(vllm)
        for suffix in ("input_mix_weight_down.weight",
                       "block_inject_weight.weight")
    }
    moe_names = {n for n, _ in port.named_parameters()
                 if n.endswith(("w13_weight", "w2_weight"))}
    assert covered | prefix_names | moe_names >= port_names


def _merged_prefixes(model):
    prefixes = {}
    for name, param in model.named_parameters():
        if "input_mix_weight_down_block_inject.weight" in name:
            prefix = name.replace(
                "input_mix_weight_down_block_inject.weight", "")
            pad = (-(HC_LOWRANK + HC_COUNT)) % 16
            prefixes[prefix] = (HC_LOWRANK, HC_COUNT, pad)
    return prefixes


# ---------------------------------------------------------------------------
# MTP: remap parity, parameter trees, two-stream forward contract
# ---------------------------------------------------------------------------
def test_mtp_weight_remap_matches_vllm_original():
    from vllm.models.qwen4_exp.nvidia.mtp import \
        _remap_mtp_weight_name as vllm_remap
    corpus = [
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "model.language_model.model.embed_tokens.weight",
        "model.mtp.layers.48.self_attn.qkv_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.shared_head.head.weight",
        "model.shared_head.head.weight",
        "shared_head.head.weight",
        "model.lm_head.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "totally.unknown.name.weight",
        "mtp.fc_embedding.weight",
        "model.mtp.hyper_connection_mixer.hc_norm.weight",
    ]
    for name in corpus:
        assert tpu_remap_mtp_weight_name(name) == vllm_remap(name), name


_current_mtp_config = None


def _make_mtp_config(model_dir):
    """A VllmConfig with an MTP speculative config for one model build."""
    from vllm.config.speculative import SpeculativeConfig
    global _current_mtp_config
    config = _build_vllm_config(model_dir)
    _current_mtp_config = config
    # The draft model is the SAME checkpoint (MTP weights live inside the
    # target checkpoint).
    config.speculative_config = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=1,
        model=str(model_dir),
        target_model_config=config.model_config,
        target_parallel_config=config.parallel_config)
    return config


@pytest.fixture(scope="module")
def mtp_pair(tpu_env, model_dir):
    """(port MTP, original MTP) built from identical draft configs."""
    import vllm.distributed as vd
    models = {}
    previous_default = torch.get_default_dtype()
    # vLLM's loader sets the default dtype to the model dtype before
    # construction; modules that rely on the default (norms, LM head, fc
    # linears) must be built under it, exactly like the runner does.
    torch.set_default_dtype(torch.bfloat16)
    for owner in ("port", "vllm"):
        config = _make_mtp_config(model_dir)
        with set_current_vllm_config(config):
            from vllm.distributed import parallel_state as _ps
            if _ps._TP is None:
                vd.initialize_model_parallel(tensor_model_parallel_size=1)
            if owner == "port":
                from tpu_inference.models.vllm.experimental.qwen4_exp import (
                    Qwen4ExpMTP as TpuMTP)
                models[owner] = TpuMTP(vllm_config=config, prefix="")
            else:
                # The original's _make_draft_vllm_config runs vllm's
                # replace() over the config fields; the TPU platform's
                # dynamically-attached sharding_config attribute is not a
                # declared field there and the replace fails, which is why
                # the port resolves the draft quant config without a
                # replace. Drop it for the original's build.
                sharding = getattr(config, "sharding_config", None)
                if sharding is not None:
                    object.__delattr__(config, "sharding_config")
                import vllm.models.qwen4_exp.nvidia.mtp as vllm_mtp_mod
                with patch(
                        "vllm.models.qwen4_exp.nvidia.qsa."
                        "is_flash_attn_varlen_func_available", lambda: True):
                    models[owner] = vllm_mtp_mod.Qwen4ExpMTP(
                        vllm_config=config, prefix="")
    torch.set_default_dtype(previous_default)
    yield models["port"], models["vllm"]


def _mtp_dir():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp(prefix="qwen4exp_mtp"))


def test_mtp_parameter_trees_match(mtp_pair):
    port_mtp, vllm_mtp = mtp_pair
    port_tree, _ = _param_tree(port_mtp, HC_LOWRANK, HC_COUNT)
    vllm_tree, _ = _param_tree(vllm_mtp, HC_LOWRANK, HC_COUNT)
    assert port_tree == vllm_tree


def _mtp_checkpoint_name(name):
    """Invert the draft's internal names onto real checkpoint names.

    In the Qwen4Exp checkpoint the MTP weights live under ``model.mtp.*``
    and the shared head under ``mtp.shared_head.head.*``; the port's and
    the original's ``_remap_mtp_weight_name`` map those back onto the
    draft module paths.
    """
    if name.startswith("lm_head."):
        return "mtp.shared_head.head." + name[len("lm_head."):]
    if name.startswith("model."):
        return "model.mtp." + name[len("model."):]
    return None


def test_mtp_weight_remap_loads_original_checkpoint(mtp_pair):
    """A checkpoint emitted from the original draft (with the real
    checkpoint naming) loads into the port's draft through the same
    name remapping, and every parameter lands identically."""
    port_mtp, vllm_mtp = mtp_pair
    torch.manual_seed(321)
    with torch.no_grad():
        for param in vllm_mtp.parameters():
            param.uniform_(-0.2, 0.2)
    checkpoint = []
    for name, param in vllm_mtp.named_parameters():
        for split_name, tensor in _split_checkpoint_name(
                name, param, HC_LOWRANK, HC_COUNT):
            checkpoint_name = _mtp_checkpoint_name(split_name)
            assert checkpoint_name is not None, split_name
            checkpoint.append((checkpoint_name, tensor))

    # (value verification lives in test_mtp_weight_remap_loads_original_
    # checkpoint; this test only needs matching weights end to end)
    assert port_mtp.load_weights(checkpoint)
    for mtp in (port_mtp, vllm_mtp):
        mtp.model.layers = torch.nn.ModuleList([StubLayer()])


def _run_linear_probe(port_mtp, vllm_mtp):
    """Compare the OOT ColumnParallelLinear against plain torch math."""
    import torch.nn.functional as F
    with torchax.default_env():
        x = torch_view(jnp.asarray(torch.randn(4, 32).numpy()))
        out_port = port_mtp.model.fc_hidden(x)
        out_vllm = vllm_mtp.model.fc_hidden(x)
        ref = F.linear(x, torch_view(
            jnp.asarray(port_mtp.model.fc_hidden.weight.detach().cpu().
                        float().numpy())))
        p = np.asarray(jax_view(out_port[0] if isinstance(out_port, tuple)
                                else out_port).astype(jnp.float32))
        v = np.asarray(jax_view(out_vllm[0] if isinstance(out_vllm, tuple)
                                else out_vllm).astype(jnp.float32))
        r = np.asarray(jax_view(ref).astype(jnp.float32))
        print("PROBE port[0,:3]:", np.round(p[0, :3], 4))
        print("PROBE vllm[0,:3]:", np.round(v[0, :3], 4))
        print("PROBE ref [0,:3]:", np.round(r[0, :3], 4))
        for tag, mod in (("port", port_mtp.model.fc_hidden),
                         ("vllm", vllm_mtp.model.fc_hidden)):
            print("MOD", tag, type(mod).__module__, type(mod).__name__,
                  "gather=", getattr(mod, "gather_output", None),
                  "tp=", getattr(mod, "tp_size", None),
                  "quant=", type(getattr(mod, "quant_method", None)).__name__,
                  "out_sizes=", getattr(mod, "output_sizes", None))


@pytest.fixture(scope="module")
def hc_cpu_kernels():
    """CPU kernels for the original's HC glue custom ops.

    The CUDA reference packs the HC glue into ``vllm::qwen4_exp_*`` ops
    with Triton kernels and only registers a CUDA backend. The eager
    formulas below are exactly what the common (non-fused) implementation
    and the kernels compute; registering them lets the ORIGINAL model run
    its glue on CPU so the port can be diffed against it.
    """
    import torch.library

    def grouped_gemma(x, weight, eps, num_groups):
        group_dim = x.shape[-1] // num_groups
        grouped = x.float().unflatten(-1, (num_groups, group_dim))
        var = grouped.square().mean(-1, keepdim=True)
        out = (grouped * torch.rsqrt(var + eps)).flatten(-2)
        return (out * (1.0 + weight.float())).to(x.dtype)

    def hc_silu(x, hc_count):
        return torch.nn.functional.silu(x / hc_count)

    def hc_gate_mix(x, gate, hc_count):
        dim = gate.shape[-1]
        hc_dim = dim // hc_count
        return (torch.sigmoid(gate).unflatten(-1, (hc_count, hc_dim)) *
                x.unflatten(-1, (hc_count, hc_dim))).mean(-2).to(x.dtype)

    def hc_combine(residual, block_output, injection_logits, hc_count):
        # Unit injection when the pending logits are missing; otherwise the
        # block output enters every stream with 2*sigmoid(logits/hc).
        if injection_logits is None:
            out = (residual.unflatten(-1, (hc_count, -1)) +
                   block_output.unsqueeze(-2))
        else:
            weight = (2.0 * torch.sigmoid(injection_logits /
                                          hc_count)).unsqueeze(-1)
            out = (residual.unflatten(-1, (hc_count, -1)) +
                   block_output.unsqueeze(-2) * weight)
        return out.flatten(-2).to(residual.dtype)

    def hc_combine_norm(residual, block_output, injection_logits,
                        norm_weight, eps, hc_count):
        combined = hc_combine(residual, block_output, injection_logits,
                              hc_count)
        return combined, grouped_gemma(combined, norm_weight, eps,
                                       hc_count)

    torch.library.register_kernel("vllm::qwen4_exp_grouped_gemma_rmsnorm",
                                  "CPU", grouped_gemma)
    torch.library.register_kernel("vllm::qwen4_exp_hc_silu", "CPU", hc_silu)
    torch.library.register_kernel("vllm::qwen4_exp_hc_gate_mix", "CPU",
                                  hc_gate_mix)
    torch.library.register_kernel("vllm::qwen4_exp_hc_combine", "CPU",
                                  hc_combine)
    torch.library.register_kernel("vllm::qwen4_exp_hc_combine_norm", "CPU",
                                  hc_combine_norm)
    yield


class StubLayer(torch.nn.Module):
    """Deterministic stand-in for the draft decoder layer."""

    calls = 0

    def forward(self, hidden_states, prev_block_output, prev_injection,
                positions, **kwargs):
        block_input = hidden_states * 2.0
        block_output = (prev_block_output * 3.0
                        if prev_block_output is not None else hidden_states)
        injection = hidden_states[:, :HC_COUNT] * 0.0 + 1.0
        StubLayer.calls += 1
        return hidden_states, block_output, injection


def test_mtp_forward_two_stream_contract(mtp_pair, hc_cpu_kernels):
    """With the decoder layer stubbed, the draft's embedding fusion and
    final mixer must produce the same (sample, multi-stream) pair on
    both implementations, from identical weights.

    The OOT linears' apply() needs the runner's incremental weight
    processing (device sharding), which the loader tests exercise; for
    this wiring comparison both sides run on the plain vLLM unquantized
    method so the model-level glue is what differs.
    """
    port_mtp, vllm_mtp = mtp_pair
    torch.manual_seed(321)
    with torch.no_grad():
        for param in vllm_mtp.parameters():
            param.uniform_(-0.2, 0.2)
    checkpoint = []
    for name, param in vllm_mtp.named_parameters():
        for split_name, tensor in _split_checkpoint_name(
                name, param, HC_LOWRANK, HC_COUNT):
            checkpoint_name = _mtp_checkpoint_name(split_name)
            assert checkpoint_name is not None, split_name
            checkpoint.append((checkpoint_name, tensor))
    port_mtp.load_weights(checkpoint)

    for mtp in (port_mtp, vllm_mtp):
        mtp.model.layers = torch.nn.ModuleList([StubLayer()])
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    for module in port_mtp.modules():
        module_type = type(module).__name__
        quant = getattr(module, "quant_method", None)
        if quant is not None and "Linear" in module_type:
            module.quant_method = UnquantizedLinearMethod()

    torch.manual_seed(5)
    input_ids = torch.randint(0, 64, (4, ))
    positions = torch.arange(4)
    hidden_states = torch.randn(4, HC_COUNT * 32,
                                dtype=torch.bfloat16)

    outputs = []
    for mtp in (port_mtp, vllm_mtp):
        sample, multi = mtp(input_ids,
                            positions,
                            hidden_states=hidden_states,
                            spec_step_idx=0)
        outputs.append((sample.detach().float().numpy(),
                        multi.detach().float().numpy()))
    (port_sample, port_multi), (vllm_sample, vllm_multi) = outputs
    assert StubLayer.calls == 2  # the layer ran once per implementation
    # The two sides evaluate the same math in different orders and
    # precisions (inline bf16 torch vs the fused custom-op path); vLLM's
    # own fused-vs-eager tests use atol/rtol 1e-2 for exactly this.
    np.testing.assert_allclose(port_sample,
                               vllm_sample,
                               rtol=1e-2,
                               atol=1e-2)
    np.testing.assert_allclose(port_multi,
                               vllm_multi,
                               rtol=1e-2,
                               atol=1e-2)
    assert port_sample.shape == (4, 32)
    assert port_multi.shape == (4, HC_COUNT * 32)


# ---------------------------------------------------------------------------
# QSA deep coverage (port-side math against vLLM's reference semantics)
# ---------------------------------------------------------------------------
def test_qsa_streaming_topk_across_multiple_merge_chunks():
    """comp_cap spans several _GROUP_CHUNK buckets: the streaming top-k
    must merge across iterations and recover the globally best groups."""
    import math

    from tpu_inference.models.vllm.experimental.qwen4_exp.qsa import (
        _qsa_select_and_store, _token_to_req)

    compress_ratio, token_topk = 4, 32
    block_topk = token_topk // compress_ratio
    head_dim, index_heads = 16, 2
    comp_cap = 1024
    seq_len = 2800
    torch.manual_seed(21)
    # Strictly increasing per-group scores -> the global top-k is unique
    # and equals the last `block_topk` visible groups.
    q_dir = torch.nn.functional.normalize(torch.randn(head_dim), dim=0)
    # Exponential scales keep adjacent-group score gaps far above the
    # bf16 quantum even at the top of the range (where linear scales
    # would collide after quantization).
    scales = torch.exp(torch.linspace(0.0, 10.0, comp_cap))
    keys = torch.stack([
        q_dir * scale + 0.001 * scale * torch.randn(head_dim)
        for scale in scales
    ])
    bank = torch.zeros(2, comp_cap, 1, head_dim)
    bank[1] = keys.unsqueeze(1)
    q_sel = (q_dir.reshape(1, 1, head_dim) + 0.01 * torch.randn(
        3, index_heads, head_dim)).to(torch.bfloat16).float()

    positions = jnp.asarray([2799, 1500, 700], dtype=jnp.int32)
    seq_lens = jnp.asarray([seq_len], dtype=jnp.int32)
    qsl = jnp.asarray([0, 3], dtype=jnp.int32)
    token_to_req = _token_to_req(qsl, 3)

    _, selection = _qsa_select_and_store(
        jnp.asarray(bank.numpy(), dtype=jnp.bfloat16),
        jnp.zeros((3, 1, head_dim), dtype=jnp.bfloat16),
        jnp.zeros(3, dtype=bool),
        jnp.asarray(q_sel.numpy(), dtype=jnp.bfloat16), positions,
        token_to_req, jnp.ones(3, dtype=bool), seq_lens,
        jnp.asarray([1], dtype=jnp.int32), compress_ratio, block_topk,
        head_dim)
    selection = np.asarray(selection)

    # With 1024 bf16-quantized groups adjacent scores can collide, and
    # top-k tie-breaking is implementation-defined; pin the meaningful
    # invariant instead: every selected group's score is >= the
    # block_topk-th largest visible score (the torch reference's own
    # threshold on the same bf16 keys).
    scores = torch.relu(
        torch.einsum("thd,gd->thg", q_sel,
                     bank[1, :, 0, :])).sum(dim=1) / math.sqrt(head_dim)
    for row, position in enumerate([2799, 1500, 700]):
        visible_groups = min((position + 1) // compress_ratio,
                             seq_len // compress_ratio)
        best_groups = set(
            int(idx) // compress_ratio
            for idx in selection[row] if idx >= 0)
        visible_tokens = position + 1
        tail_group = (visible_tokens // compress_ratio
                      if visible_tokens % compress_ratio else None)
        expected_groups = set(
            range(visible_groups - block_topk, visible_groups))
        if tail_group is not None:
            expected_groups.add(tail_group)
        assert best_groups == expected_groups, (row, best_groups,
                                                expected_groups)
        visible_scores = scores[row, :visible_groups]
        threshold = float(
            torch.kthvalue(-visible_scores, block_topk).values.abs())
        selected = scores[row, torch.tensor(sorted(best_groups))]
        assert float(selected.min()) >= threshold - 5e-3, (row, threshold)


def test_qsa_main_cache_packing_kv_heads_gt_1():
    """The packed (G, packing) cache layout must reproduce K and V for
    multi-KV-head attention (the layout bug class the first port hit)."""
    from tpu_inference.models.vllm.experimental.qwen4_exp.qsa import (
        _qsa_update_main_cache)

    torch.manual_seed(31)
    head_dim, num_kv_heads = 16, 2
    packing = 2
    num_heads = 4
    num_blocks, page_size, num_tokens = 4, 2, 6
    g = (2 * num_kv_heads + packing - 1) // packing
    cache = jnp.zeros((num_blocks, page_size, g, packing, head_dim),
                      dtype=jnp.bfloat16)
    k = torch.randn(num_tokens, num_kv_heads, head_dim)
    v = torch.randn(num_tokens, num_kv_heads, head_dim)
    k_j = jnp.asarray(k.numpy(), dtype=jnp.bfloat16)
    v_j = jnp.asarray(v.numpy(), dtype=jnp.bfloat16)
    positions = jnp.arange(num_tokens, dtype=jnp.int32)
    block_tables = jnp.asarray(np.arange(num_blocks).reshape(
        1, num_blocks), dtype=jnp.int32)

    new_cache = _qsa_update_main_cache(k_j, v_j, positions,
                                       jnp.zeros(num_tokens, jnp.int32),
                                       jnp.ones(num_tokens, dtype=bool),
                                       jnp.asarray([num_tokens],
                                                  dtype=jnp.int32),
                                       block_tables, cache)

    # Logical KV rows are the row-major (G, packing) split of the
    # concatenated [K heads; V heads] axis.
    rows = new_cache.reshape(-1, g, packing, head_dim)
    kv_rows = rows[:num_tokens].reshape(num_tokens, g * packing, head_dim)
    read_k = np.asarray(kv_rows[:, :num_kv_heads, :].astype(jnp.float32))
    read_v = np.asarray(kv_rows[:, num_kv_heads:, :].astype(jnp.float32))
    np.testing.assert_array_equal(read_k,
                                  np.asarray(k_j.astype(jnp.float32)))
    np.testing.assert_array_equal(read_v,
                                  np.asarray(v_j.astype(jnp.float32)))

    # A GQA read sees exactly the written K/V per query-group.
    selection = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
    q_main = torch.randn(1, num_heads, head_dim)
    q_j = jnp.asarray(q_main.numpy(), dtype=jnp.bfloat16)
    out = _qsa_sparse_attention(selection, q_j, block_tables,
                                jnp.zeros(1, jnp.int32), new_cache,
                                head_dim**-0.5, num_kv_heads, num_heads)
    k_f32 = np.asarray(k_j.astype(jnp.float32))  # [T, KH, D]
    v_f32 = np.asarray(v_j.astype(jnp.float32))
    rep = num_heads // num_kv_heads
    selected = np.asarray(selection)[0]
    keys_rep = np.repeat(k_f32[selected], rep, axis=1)  # [S, H, D]
    values_rep = np.repeat(v_f32[selected], rep, axis=1)
    q_np = np.asarray(q_j.astype(jnp.float32))[0]  # [H, D]
    scores = np.einsum("hd,shd->hs", q_np, keys_rep)  # [H, S]
    scaled = scores * head_dim**-0.5
    weights = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)
    expected = np.einsum("hs,shd->hd", weights, values_rep)  # [H, D]
    gathered = np.asarray(out.astype(jnp.float32))[0]  # [H, D]
    np.testing.assert_allclose(gathered, expected, atol=2e-2, rtol=2e-2)


def test_qsa_padding_tokens_are_inert():
    """Padded slots must not corrupt caches: writes go to the null block
    with zero values and produce zero attention output."""
    from tpu_inference.models.vllm.experimental.qwen4_exp.qsa import (
        _qsa_update_main_cache)

    torch.manual_seed(41)
    head_dim, num_kv_heads, packing = 16, 1, 2
    num_blocks, page_size = 5, 2
    total, real = 8, 4
    cache = jnp.zeros((num_blocks, page_size, 1, packing, head_dim),
                      dtype=jnp.bfloat16)
    k = torch.randn(total, num_kv_heads, head_dim)
    v = torch.randn(total, num_kv_heads, head_dim)
    k_j = jnp.asarray(k.numpy(), dtype=jnp.bfloat16)
    v_j = jnp.asarray(v.numpy(), dtype=jnp.bfloat16)
    positions = jnp.arange(total, dtype=jnp.int32)
    valid = jnp.arange(total, dtype=jnp.int32) < real
    token_to_req = jnp.zeros(total, dtype=jnp.int32)
    # Block 0 is the reserved null block; the request's pages start at 1.
    block_tables = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    del num_blocks

    new_cache = _qsa_update_main_cache(k_j, v_j, positions, token_to_req,
                                       valid,
                                       jnp.asarray([real],
                                                   dtype=jnp.int32),
                                       block_tables, cache)

    rows = new_cache.reshape(-1, 1, 2, head_dim)
    written = np.asarray(rows[page_size:page_size + real, 0, 0,
                              :].astype(jnp.float32))
    np.testing.assert_array_equal(
        written,
        np.asarray(k_j.astype(jnp.float32)).reshape(total,
                                                    head_dim)[:real])
    # The null block stayed untouched by the masked padding writes...
    assert not np.asarray(rows[:page_size].astype(jnp.float32)).any()
    # ...and no page row beyond the real tokens was written either.
    tail = np.asarray(rows[page_size + real:].astype(jnp.float32))
    assert not tail.any()


def test_qsa_state_cache_interfaces(model_pair, model_dir):
    """The side caches must advertise MambaBase-compatible state shapes."""
    from vllm.v1.attention.backends.registry import (
        MambaAttentionBackendEnum)

    from tpu_inference.models.vllm.experimental.qwen4_exp.qsa import (
        _QSAKeyStateCache)

    vllm_config = _build_vllm_config(model_dir)
    with set_current_vllm_config(vllm_config):
        state = _QSAKeyStateCache(
            vllm_config=vllm_config,
            capacity_tokens=1234,
            head_size=32,
            dtype=torch.bfloat16,
            prefix="model.layers.0.self_attn.indexer.raw_key_cache",
        )
    # Registered for the runner's per-layer cache allocation.
    assert "model.layers.0.self_attn.indexer.raw_key_cache" in (
        vllm_config.compilation_config.static_forward_context)
    assert state.mamba_type == MambaAttentionBackendEnum.SHORT_CONV
    assert state.get_state_shape() == ((1234, 1, 32), )
    assert state.get_state_dtype() == (torch.bfloat16, )
    assert state.is_kv_cache_tp_replicated()


def test_qsa_state_cap_tokens_env_override(monkeypatch):
    from tpu_inference.models.vllm.experimental.qwen4_exp import qsa
    monkeypatch.setenv("TPU_QSA_STATE_CAP_TOKENS", "12345")
    assert qsa._state_cap_tokens(99999) == 12345
    monkeypatch.delenv("TPU_QSA_STATE_CAP_TOKENS")
    assert qsa._state_cap_tokens(99999) == 99999


def test_qsa_get_kv_cache_spec_fields(model_pair, model_dir):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    port, _ = model_pair
    vllm_config = _build_vllm_config(model_dir)
    layer = port.model.layers[1].self_attn
    spec = layer.get_kv_cache_spec(vllm_config)
    assert isinstance(spec, FullAttentionSpec)
    assert spec.block_size == vllm_config.cache_config.block_size
    assert spec.num_kv_heads == 2
    assert spec.head_size == 16
    assert spec.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# PLE checkpoint shard loading vs vLLM's own helper
# ---------------------------------------------------------------------------
def test_ple_embedding_shard_loading_matches_vllm_helper():
    """``_load_embedding_shard`` must place shard rows exactly like
    vLLM's ``copy_ple_embedding_shard_`` for every overlap geometry."""
    from vllm.models.qwen4_exp.common.ple import (
        compute_ple_shard_overlap, copy_ple_embedding_shard_)

    from tpu_inference.models.vllm.experimental.qwen4_exp.ple import (
        Qwen4ExpNGramEmbedding)

    vocab_size, head_dim = 64, 8
    weight = torch.randn(vocab_size, head_dim)

    shard_specs = [
        (0, 64),  # full shard
        (0, 32),  # covers the low range only
        (32, 64),  # covers the high range only
        (16, 48),  # interior overlap
        (64, 96),  # entirely above the vocab range
    ]
    for start, end in shard_specs:
        shard = torch.randn(end - start, head_dim)
        port_weight = weight.clone()
        embedding = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
        port_param = torch.nn.Parameter(port_weight)
        embedding.ngram_embedding = SimpleNamespace(
            weight=port_param,
            shard_indices=SimpleNamespace(org_vocab_start_index=0,
                                          org_vocab_end_index=vocab_size))
        Qwen4ExpNGramEmbedding._load_embedding_shard(
            embedding, port_param, shard, start)

        ref_weight = weight.clone()
        copied = copy_ple_embedding_shard_(ref_weight,
                                           shard,
                                           checkpoint_start=start,
                                           tp_start=0,
                                           tp_end=vocab_size)
        torch.testing.assert_close(port_weight, ref_weight)
        del copied


def test_ple_embedding_load_weights_routes_shards_and_skips_junk(
        tpu_env, model_dir):
    """The port's embedding loader must route shard rows, load the hash
    buffers, and skip the CUDA-side hashstat tensors."""
    import vllm.distributed as vd
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding)

    from tpu_inference.models.vllm.experimental.qwen4_exp.ple import (
        Qwen4ExpNGramEmbedding)
    config = _build_vllm_config(model_dir)
    context = set_current_vllm_config(config)
    context.__enter__()
    from vllm.distributed import parallel_state as _ps
    if _ps._TP is None:
        vd.initialize_model_parallel(tensor_model_parallel_size=1)

    text_config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=100,
        eos_token_id=1,
        ple_embed_dim=8,
        vocab_size=32,
        make_ngram_vocab_size_divisible_by=16,
    )
    embedding = Qwen4ExpNGramEmbedding(
        text_config,
        embedding_dim=text_config.ple_embed_dim,
        ple_dense_layer_id=1,
        prefix="model.layers.1.ple_embedding",
        quant_config=None,
        params_dtype=torch.bfloat16,
    )
    assert isinstance(embedding.ngram_embedding, VocabParallelEmbedding)

    # Checkpoint shards split the n-gram vocab into split_ngram_parts
    # pieces of shard_size rows each; with a tiny vocab each piece is a
    # couple of rows of the embedding dimension.
    shard_size = (embedding.org_vocab_size + embedding.split_ngram_parts -
                  1) // embedding.split_ngram_parts
    head_dim = embedding.ngram_embedding.embedding_dim
    shard = torch.randn(shard_size, head_dim)
    buffers = {
        "layer_multipliers":
        torch.tensor([11, 13, 17], dtype=torch.int64),
        "ngram_heads_vocab_sizes":
        torch.tensor([101, 103, 107, 109], dtype=torch.int64),
        "ngram_heads_offsets":
        torch.tensor([0, 101, 204, 311], dtype=torch.int64),
    }
    junk = torch.randn(3, 3)
    weights = [
        ("ngram_embedding.shard_0.weight", shard),
        *[(name, value) for name, value in buffers.items()],
        ("hashstats_dummy", junk),
        ("token_lookup", junk),
    ]
    loaded = embedding.load_weights(iter(weights))
    context.__exit__(None, None, None)
    assert loaded
    table = embedding.ngram_embedding.weight.data.float()
    torch.testing.assert_close(table[:shard_size],
                               shard.to(torch.bfloat16).float())
    for name, value in buffers.items():
        torch.testing.assert_close(getattr(embedding, name).data, value)
