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
"""TPU (torchax) Qwen4Exp (public name: Qwen3.8-Flash-Next) text model.

Port of ``vllm/models/qwen4_exp/nvidia/model.py`` to tpu-inference's torchax
path. Structure and checkpoint layout match the vLLM CUDA reference exactly:

* hybrid layers: ``linear_attention`` (Gated DeltaNet — reused from vLLM's
  ``QwenGatedDeltaNetAttention``, which tpu-inference replaces with the TPU
  JAX-kernel implementation via ``register_oot``) and ``full_attention``
  (QSA — the TPU implementation in this package);
* ``GatedResidual`` hyper-connections around every attention and MLP block,
  with a final mixer (checkpoint-native separate projections instead of the
  CUDA reference's merged low-latency GEMM packing — see
  ``hyper_connection.py``);
* PLE N-gram embedding layers at ``ple_layer_ids`` (this package's TPU
  implementation);
* ultra-sparse MoE blocks reusing vLLM's ``Qwen3NextSparseMoeBlock`` whose
  ``FusedMoE`` lowers through tpu-inference's registered MoE runner.

Not carried over (with reasons):

* ``enable_qwen4_exp_low_latency_gemm`` — CUDA CuBLAS-heuristic dispatch.
* ``Qwen4ExpForConditionalGeneration`` (vision tower + deepstack) — the
  text-only checkpoint path is ported first; multimodal support is deferred
  and documented in the support matrix.
* ``_remap_qsa_cache_scale_name`` — the TPU QSA owner carries no persistent
  ``_k_scale``/``_v_scale`` buffers; the checkpoint's cache-scale names fall
  into the ignore list instead (QSA on TPU requires a BF16 KV cache).
* ``Qwen4ExpMixtureOfExperts`` (EPLB protocol) — expert-load-balancing is
  not part of the TPU path yet.
* CUDA-graph / MTP multi-stream hidden buffer plumbing
  (``_mtp_hidden_buffer``) — the TPU runner's MTP drafter integration is
  not yet in place (see ``mtp.py``).
"""

from collections.abc import Iterable
from itertools import islice
from typing import Optional, Tuple, Union

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import \
    QwenGatedDeltaNetAttention
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import (HasInnerState, IsHybrid,
                                                   SupportsLoRA, SupportsPP)
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_next import (Qwen3NextAttention,
                                                   Qwen3NextMLP,
                                                   Qwen3NextSparseMoeBlock)
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              WeightsMapper,
                                              extract_layer_index,
                                              make_layers,
                                              make_empty_intermediate_tensors_factory,
                                              maybe_prefix)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig

from tpu_inference.logger import init_logger
from tpu_inference.models.vllm.experimental.qwen4_exp.hyper_connection import \
    GatedResidual
from tpu_inference.models.vllm.experimental.qwen4_exp.hyper_connection import \
    HyperConnectionConfig
from tpu_inference.models.vllm.experimental.qwen4_exp.ple import \
    Qwen4ExpPLELayer
from tpu_inference.models.vllm.experimental.qwen4_exp.qsa import \
    Qwen4ExpQSAAttention

logger = init_logger(__name__)

_QWEN4_EXP_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]

# Checkpoint names that map onto non-persistent runtime state or that the
# CUDA reference folds into packed modules the TPU path does not create.
_QWEN4_EXP_SKIPPED_SUBSTRS = (
    "hashstats_",
    "token_lookup",
    "hyper_connection_mixer.block_inject_weight",
)

hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper


class Qwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3Next MoE with the Qwen4Exp sequence-parallel restriction."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        parallel_config = vllm_config.parallel_config
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)


class Qwen4ExpDecoderLayer(nn.Module):
    """One hybrid decoder layer with HC-managed residual streams."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp HC does not support sequence-parallel MoE")
        self.ple: Optional[Qwen4ExpPLELayer] = None
        ple_layer_ids = config.ple_layer_ids
        if (self.layer_idx + 1) in ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx
                for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = Qwen4ExpPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            # Reuses vLLM's GDN layer; tpu-inference's registered OOT
            # subclass swaps in the TPU (JAX kernel) forward.
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif layer_type == "full_attention":
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen4ExpQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0) or 0
        absolute_layer_id = self.layer_idx + 1
        is_moe_layer = self.layer_idx not in mlp_only_layers and (
            num_experts > 0 and
            absolute_layer_id % config.decoder_sparse_step == 0)
        if is_moe_layer:
            self.mlp = Qwen4ExpSparseMoeBlock(vllm_config=vllm_config,
                                              prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "attn_hyper_connection"),
        )
        self.mlp_hyper_connection = GatedResidual(
            hc_config,
            prefix=maybe_prefix(prefix, "mlp_hyper_connection"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: Optional[torch.Tensor],
        prev_injection: Optional[torch.Tensor],
        positions: torch.Tensor,
        *,
        input_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_block_output is None:
            assert prev_injection is None
        attn_hc = self.attn_hyper_connection
        if self.ple is not None:
            # PLE adds directly to the multi-stream state, so the pending
            # HC state must be materialized before the addition.
            if prev_block_output is not None:
                hidden_states = attn_hc.combine(hidden_states,
                                                prev_block_output,
                                                prev_injection)
                prev_block_output = prev_injection = None
            if input_ids is None:
                raise RuntimeError("PLE requires raw input_ids")
            hidden_states = hidden_states + self.ple(hidden_states,
                                                     input_ids)

        # Fuse a pending combine with this HC module's mix when possible.
        if prev_block_output is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection)
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)

        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(hidden_states=block_input)
        elif self.layer_type == "full_attention":
            attn_out = self.self_attn(
                hidden_states=block_input,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")

        mlp_hc = self.mlp_hyper_connection
        hidden_states, block_input, injection = mlp_hc.combine_and_mix(
            hidden_states, attn_out, injection)
        mlp_out = self.mlp(block_input)
        return hidden_states, mlp_out, injection


class Qwen4ExpModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size,
                                                   config.hidden_size)

        def get_layer(prefix: str) -> Qwen4ExpDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        intermediate_size = config.hidden_size * config.hc_count
        self.make_empty_intermediate_tensors = \
            make_empty_intermediate_tensors_factory(
                ["hidden_states"], intermediate_size)

        self.hyper_connection_mixer: Optional[GatedResidual]
        if get_pp_group().is_last_rank:
            hc_config = HyperConnectionConfig(
                hc_count=config.hc_count,
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                hc_lowrank=config.hc_lowrank,
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=True,
            )
            self.hyper_connection_mixer = GatedResidual(
                hc_config,
                use_combine=False,
                prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
            )
        else:
            self.hyper_connection_mixer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError(
                    "pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        block_output = None
        injection = None
        last_layer = None
        for layer_idx, layer in islice(enumerate(self.layers),
                                       self.start_layer, self.end_layer):
            last_layer = layer
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
                input_ids=input_ids,
            )
        if not get_pp_group().is_last_rank:
            # PP transports one tensor, not the delayed HC tuple.
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection)
            return IntermediateTensors({"hidden_states": hidden_states})

        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        _, sample_hidden_states, _ = final_mixer.combine_and_mix(
            hidden_states, block_output, injection)
        return sample_hidden_states

    def load_weights(self,
                     weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        mapper = hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={substr: None
                                for substr in _QWEN4_EXP_SKIPPED_SUBSTRS})
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.
            copy(),
        )
        return loader.load_weights(weights, mapper=mapper)


class Qwen4ExpForCausalLM(nn.Module, HasInnerState, SupportsLoRA, SupportsPP,
                          IsHybrid):
    """TPU (torchax) Qwen4Exp text-only causal LM."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "kv_proj": ["key_proj", "value_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."})
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = Qwen4ExpModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self,
                       hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self,
                     weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.
            copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
