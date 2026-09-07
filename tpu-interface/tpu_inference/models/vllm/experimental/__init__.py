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

# Mapping of architecture name -> "module:ClassName" for the TPU-specific
# vLLM/torchax model implementations that should override vLLM's built-in
# ones. Values use the lazy string form so the (torch/tilelang) model module
# is only imported when the architecture is actually instantiated.
_TPU_VLLM_MODELS = {
    "DeepseekV4ForCausalLM":
    "tpu_inference.models.vllm.experimental.deepseek_v4:DeepseekV4ForCausalLM",
    # Qwen4Exp = Qwen3.8-Flash-Next. The text-only causal LM and the MTP
    # draft head; the multimodal wrapper stays CUDA/ROCm-only for now.
    "Qwen4ExpForCausalLM":
    "tpu_inference.models.vllm.experimental.qwen4_exp:Qwen4ExpForCausalLM",
    "Qwen4ExpMTP":
    "tpu_inference.models.vllm.experimental.qwen4_exp:Qwen4ExpMTP",
    # The released checkpoint (Qwen/Qwen3.8-Flash-Next-FP8) declares the
    # multimodal-layout architecture "Qwen4ExpForConditionalGeneration"
    # (vision_config + model.language_model.* weight prefixes). The TPU
    # torchax port serves the text-only path of that checkpoint directly, so
    # map the wrapper architecture name onto the text causal LM: its
    # load_weights already strips the "model.language_model." prefix (see
    # Qwen4ExpForCausalLM.hf_to_vllm_mapper). Without this, vLLM would fall
    # back to the CUDA/ROCm-only wrapper class and refuse to run on TPU.
    "Qwen4ExpForConditionalGeneration":
    "tpu_inference.models.vllm.experimental.qwen4_exp:Qwen4ExpForCausalLM",
}


def register_models():
    """Override vLLM's built-in model classes with TPU-specific ones.

    Called from the ``vllm.general_plugins`` entrypoint so it runs at vLLM
    startup, before any model is resolved from its architecture name.
    """
    # Direct module import: ``vllm.ModelRegistry`` is unavailable during the
    # plugin hook (top-level ``vllm`` is still mid-import there).
    from vllm.model_executor.models.registry import ModelRegistry

    for arch, model_cls in _TPU_VLLM_MODELS.items():
        ModelRegistry.register_model(arch, model_cls)
