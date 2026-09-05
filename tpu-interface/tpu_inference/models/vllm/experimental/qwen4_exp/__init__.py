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
"""TPU (torchax) Qwen4Exp — public name Qwen3.8-Flash-Next.

The architecture classes here are registered over vLLM's built-in ones (the
latter refuse to load on TPU and are CUDA/ROCm-specific) via
``tpu_inference/models/vllm/experimental/_TPU_VLLM_MODELS``.

Phase B (JAX-native / Flax NNX) status: intentionally deferred. The torchax
path already routes this model's GDN layers through the JAX GDN Pallas
kernels (``tpu_inference.kernels.gdn``) and its MoE through the registered
TPU MoE runner, so a parallel Flax NNX reimplementation of GDN + QSA + PLE +
HC + MoE would duplicate roughly five thousand lines of untestable code
without any TPU to validate either path. A JAX-native port should start
from the QSA/PLE math isolated in ``qsa.py``/``ple.py`` (all state-machine
logic is plain jnp there) once the torchax path has been validated on real
hardware.
"""

from tpu_inference.models.vllm.experimental.qwen4_exp.model import (
    Qwen4ExpDecoderLayer, Qwen4ExpForCausalLM, Qwen4ExpModel,
    Qwen4ExpSparseMoeBlock)
from tpu_inference.models.vllm.experimental.qwen4_exp.mtp import Qwen4ExpMTP

__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpMTP",
    "Qwen4ExpModel",
    "Qwen4ExpSparseMoeBlock",
]
