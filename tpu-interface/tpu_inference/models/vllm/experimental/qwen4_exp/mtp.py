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
"""TPU (torchax) Qwen4Exp MTP (Multi-Token Predictor) draft model.

Port of ``vllm/models/qwen4_exp/nvidia/mtp.py`` to the torchax path. The
draft reuses the Qwen4Exp backbone building blocks (HC / MoE / GDN / QSA via
``Qwen4ExpDecoderLayer``) but drops multimodal handling, forces PLE off
(MTP layers sit beyond ``num_hidden_layers`` so no ``ple_layer_ids`` entry
reaches them), and fuses the backbone hidden with the new-token embedding
through ``fc_embedding``/``fc_hidden`` instead of the generic MTP
``Linear(2H, H)`` fusion.

The forward emits the same two-stream contract as the reference: the
sample-ready single stream ``[T, H]`` (final-mixer collapsed) and the
pre-final-mixer multi stream ``[T, hc_count * H]`` for the next draft step.

KNOWN GAP (flagged): the TPU runner's speculative-decoding drafter selector
(``tpu_inference/runner/tpu_runner._init_speculative_decoding``) currently
instantiates only the ngram/dflash/eagle3 drafters, so serving with
``--speculative-method mtp`` requires the runner-side MTP drafter integration
before this module can be exercised end to end. The module is complete at
the model level (weights, forward contract, step-fn kwargs) so that
integration can land without further model-side changes.
"""

from collections.abc import Iterable
from typing import Optional, Tuple, Union

import torch
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              PPMissingLayer, WeightsMapper,
                                              make_empty_intermediate_tensors_factory,
                                              maybe_prefix)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig

from tpu_inference.models.vllm.experimental.qwen4_exp.hyper_connection import \
    GatedResidual
from tpu_inference.models.vllm.experimental.qwen4_exp.hyper_connection import \
    HyperConnectionConfig
from tpu_inference.models.vllm.experimental.qwen4_exp.model import (
    _QWEN4_EXP_IGNORED_MISSING_SUFFIXES, Qwen4ExpDecoderLayer,
    Qwen4ExpSparseMoeBlock, hf_to_vllm_mapper)


def _remap_mtp_weight_name(name: str) -> Optional[str]:
    """Map Qwen4Exp checkpoint paths into the standalone draft model."""
    for checkpoint_prefix in (
            "model.language_model.",
            "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


class Qwen4ExpMultiTokenPredictor(nn.Module):
    """Standalone single-layer draft backbone (reference parity)."""

    hf_to_vllm_mapper = hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        from vllm.model_executor.layers.vocab_parallel_embedding import \
            VocabParallelEmbedding
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size,
                                                   self.hidden_size)
        # The reference derives a draft-specific quant config; on TPU the
        # draft inherits the target's resolved quant config through the
        # wrapper, which mirrors what the CUDA path computes for fp8.
        draft_vllm_config = vllm_config
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=True,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            self.layers = nn.ModuleList(
                Qwen4ExpDecoderLayer(
                    draft_vllm_config,
                    layer_type="full_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                ) for idx in range(self.num_mtp_layers))

        self.pre_fc_norm_embedding = GemmaRMSNorm(self.hidden_size,
                                                  eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps)
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
        self.make_empty_intermediate_tensors = \
            make_empty_intermediate_tensors_factory(
                ["hidden_states"], self.hidden_size * self.hc_count)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> Union[torch.Tensor, IntermediateTensors, Tuple[
            torch.Tensor, torch.Tensor]]:
        hc_count = self.hc_count
        hidden_size = self.hidden_size
        prev_block_output: Optional[torch.Tensor] = None

        if get_pp_group().is_first_rank:
            assert hidden_states is not None
            if inputs_embeds is None:
                assert input_ids is not None
                inputs_embeds = self.embed_input_ids(input_ids)
            # Embedding branch: pre-norm -> fc_embedding -> [T, H].
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            inputs_embeds = self.fc_embedding(inputs_embeds)
            # Backbone hidden is the pre-final-mixer multi stream
            # [T, hc_count * H] (scheme A).
            num_tokens = hidden_states.shape[0]
            hidden_states = hidden_states.view(num_tokens, hc_count,
                                               hidden_size)
            hidden_states = self.pre_fc_norm_hidden(
                hidden_states.flatten(-2)).view(num_tokens, hc_count,
                                                hidden_size)
            hidden_states = self.fc_hidden(hidden_states)
            hidden_states = hidden_states.flatten(-2)
            prev_block_output = inputs_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=prev_block_output,
            prev_injection=None,
            positions=positions,
            input_ids=None,
        )
        if not get_pp_group().is_last_rank:
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection)
            return IntermediateTensors({"hidden_states": hidden_states})

        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection))
        return sample_hidden_states, multi_hidden

    def load_weights(self,
                     weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight":
                                None})
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.
            copy(),
        )
        return loader.load_weights(weights, mapper=mapper)


class Qwen4ExpMTP(nn.Module, SupportsPP):
    """TPU (torchax) Qwen4Exp MTP entry module (architecture Qwen4ExpMTP)."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> Union[torch.Tensor, IntermediateTensors, Tuple[
            torch.Tensor, torch.Tensor]]:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(self,
                       hidden_states: torch.Tensor,
                       spec_step_idx: int = 0) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self,
                     weights: Iterable[Tuple[str, torch.Tensor]]) -> set:

        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight

        mapper = WeightsMapper(
            orig_to_new_substr={"hyper_connection_mixer.block_inject_weight":
                                None})
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.
            copy(),
        )
        return loader.load_weights(remap_weight_names(), mapper=mapper)
