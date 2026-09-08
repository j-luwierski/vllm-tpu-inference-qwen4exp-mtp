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
"""TPU (torchax) HyperConnection / Gated Residual for Qwen4Exp.

Port of ``vllm/models/qwen4_exp/nvidia/hyperconnection.py`` (GatedResidual)
and ``vllm/models/qwen4_exp/common/hyperconnection.py``
(GroupedGemmaRMSNorm). The NVIDIA variant defers each block-output combine to
the following HC module's input RMSNorm and fuses the low-rank down and
block-inject projections into one merged linear for CUDA GEMM dispatch. On
TPU neither trick is needed, so the projections keep the checkpoint-native
separate weights (``input_mix_weight_down``, ``input_mix_weight_up``,
``block_inject_weight``) and the math is plain PyTorch that torchax lowers to
XLA. The numerics follow the eager reference in
``common/hyperconnection.py``, which the Triton kernels replicate.

Checkpoint-native weight names (no ``_EXTRA_WEIGHTS_MAPPER`` stacking needed):
``hc_norm.weight``, ``input_mix_weight_down.weight``,
``input_mix_weight_up.weight``, ``block_inject_weight.weight``. The final
mixer (``use_combine=False``) has no ``block_inject_weight``, matching the
checkpoint's skipped ``hyper_connection_mixer.block_inject_weight`` column.
"""

from dataclasses import dataclass

import torch
from torch import nn
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.linear import (ReplicatedLinear,
                                               RowParallelLinear)


@dataclass
class HyperConnectionConfig:
    """Configuration shared by all HyperConnection variants.

    Mirrors ``vllm.models.qwen4_exp.common.hyperconnection.HyperConnectionConfig``.
    """

    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    mtp_hc: bool = False
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm with a (1 + w) affine, optionally grouped.

    ``group_size=hidden_size`` normalizes each H-sized HC stream
    independently while retaining a separate affine weight for every element
    of the HC*H layout (``hc_per_branch_norm=True`` in the checkpoint).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})")
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size,
                     self.group_size))
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped *
                          torch.rsqrt(variance +
                                      self.variance_epsilon)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``mix()`` applies the grouped GemmaRMSNorm per HC stream and projects
    through a low-rank sigmoid gate to produce a single block input; it also
    emits the injection logits that the *next* ``combine`` will apply to this
    block's output (the delayed-combine scheme of the NVIDIA reference).
    ``combine_and_mix()`` fuses the pending combine of the previous block
    output with this module's input RMSNorm. ``use_combine=False`` (the final
    mixer) emits no injection.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (self.hyper_hidden_size
                     if config.hc_per_branch_norm else config.hidden_size)
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # quant_config=None: the HC glue is kept in checkpoint bf16/fp32 on
        # purpose; the NVIDIA reference also builds these unquantized.
        # Column-parallel over the hyper-hidden dim with the output gathered:
        # the gate math (silu/sigmoid on the summed projection) is unchanged,
        # but the weights stop being replicated on every chip (~553 MiB/chip
        # saved at TP=8 for the 10240x320 and 320x10240 pair).
        # Row-parallel over the hyper-hidden dim: each rank holds
        # (hc_lowrank, hyper_hidden/tp) and the partial outputs are
        # all-reduced inside the layer, so the gate math is unchanged while
        # the weights shrink ~8x versus the replicated layout.
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_mix_weight_down = RowParallelLinear(
            self.hyper_hidden_size,
            config.hc_lowrank,
            input_is_parallel=True,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=f"{prefix}.input_mix_weight_down" if prefix else
            "input_mix_weight_down",
            return_bias=False,
        )
        # Column-parallel sharding with the implicit all-reduce of the
        # partial outputs: the gate/residual math stays identical to the
        # replicated version while the weight shards to (1280, 320) per chip.
        self.input_mix_weight_up = RowParallelLinear(
            config.hc_lowrank,
            self.hyper_hidden_size,
            input_is_parallel=True,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=f"{prefix}.input_mix_weight_up" if prefix else
            "input_mix_weight_up",
            return_bias=False,
        )
        if use_combine:
            self.block_inject_weight = ReplicatedLinear(
                self.hyper_hidden_size,
                self.hc_count,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=f"{prefix}.block_inject_weight" if prefix else
                "block_inject_weight",
                return_bias=False,
            )

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    def _normalize(self, hyper_input: torch.Tensor) -> torch.Tensor:
        if self.config.hc_per_branch_norm:
            return self.hc_norm(hyper_input)
        return self.hc_norm(
            hyper_input.unflatten(-1,
                                  (self.hc_count, self.hidden_size))).flatten(
                                      -2)

    def _mix_normed(
        self, xn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Gated mean over the HC streams, plus this module's injection."""
        # The row-parallel projections shard their INPUT dim, so slice the
        # activations to this rank's span; the layer all-reduces its output.
        hh_shard = self.hyper_hidden_size // self.tp_size
        lr_shard = self.hc_lowrank // self.tp_size
        xs = xn[..., self.tp_rank * hh_shard:(self.tp_rank + 1) * hh_shard]
        gate = torch.sigmoid(
            self.input_mix_weight_up(
                torch.nn.functional.silu(self.input_mix_weight_down(xs) /
                                         self.hc_count)[...,
                                                        self.tp_rank *
                                                        lr_shard:(self.
                                                                  tp_rank +
                                                                  1) *
                                                        lr_shard]))
        block_input = (gate.unflatten(-1, (self.hc_count, self.hidden_size)) *
                       xn.unflatten(-1,
                                    (self.hc_count, self.hidden_size))).mean(
                                        dim=-2)
        injection = None
        if self.use_combine:
            injection = self.block_inject_weight(xn)
        return block_input.to(xn.dtype), injection

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = self._normalize(hidden_states)
        block_input, injection = self._mix_normed(xn)
        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is applied with unit
        weight when the pending injection is None (the first block).
        """
        hidden_states = self.combine(hidden_states, prev_block_output,
                                     prev_injection)
        xn = self._normalize(hidden_states)
        block_input, injection = self._mix_normed(xn)
        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        """Inject ``block_output`` into every HC stream of ``hidden_states``."""
        residual = hidden_states.unflatten(
            -1, (self.hc_count, self.hidden_size))
        if injection is None:
            return (residual + block_output.unsqueeze(-2)).flatten(-2).to(
                hidden_states.dtype)
        injection_weight = (2.0 * torch.sigmoid(injection /
                                                self.hc_count)).unsqueeze(-1)
        return (residual +
                block_output.unsqueeze(-2) *
                injection_weight).flatten(-2).to(hidden_states.dtype)


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
