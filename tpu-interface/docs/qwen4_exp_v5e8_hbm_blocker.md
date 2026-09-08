# Qwen3.8-Flash-Next (Qwen4Exp) on TPU v5e-8: HBM capacity blocker

Status as of 2026-09-07: the torchax port passes registration, config
parsing, model construction and weight streaming (all 131 FP8 shards,
172.76 GiB), but **cannot fit on a v5litepod-8** (8 chips x 16 GiB HBM =
128 GiB total). Verified empirically on hardware (2026-09-07):

```
per-chip placement experiment (jax mesh over 8 devices, bf16):
  MoE experts/chip  14.36 GiB  -> placed OK
  rest/chip          0.31 GiB  -> placed OK
  PLE ngram/chip     5.97 GiB  -> RESOURCE_EXHAUSTED (1.08 GiB free)
```

Checkpoint breakdown (from model.safetensors.index.json + shard headers):

| component        | size      | share |
|------------------|-----------|-------|
| MoE experts (FP8)| 114.86 GiB| 66.5% |
| PLE n-gram (FP8) |  47.75 GiB| 27.6% |
| GDN + QSA + rest |  10.15 GiB|  5.9% |
| total            | 172.76 GiB|       |

Even with perfect sharding, 172.76 GiB of weights > 128 GiB of HBM.
The engine died silently (SIGKILL-class, no Python traceback) right after
"[MoE requantization]: re-quantizing MoE weights" — the first step that
materializes the full per-chip footprint on device.

## Why the CUDA reference works but v5e-8 does not

The vLLM CUDA/ROCm reference keeps the PLE n-gram table in accelerator HBM
(80-180 GB per GPU makes this trivial). The TPU torchax port
(`tpu_inference/models/vllm/experimental/qwen4_exp/ple.py`) inherits that
design: `ngram_embedding` is a regular on-device `VocabParallelEmbedding`.
No host-offload path exists in the port (deliberately documented decisions
cover state rings and x64 hashing, but not table placement).

## What v5e-8 serving would require

At least ~45 GiB of the PLE table must leave HBM (host RAM, 377 GiB
available, is sufficient). That means a host-offload design for PLE:

- table resident in host memory (numpy/torch CPU),
- per-step: n-gram ids computed on device (the int64 hash) -> D2H ids ->
  host-side row gather -> H2D rows (batch x heads x head_dim, small),
- then the existing on-device kv_proj / gating / short-conv path.

Implications: a per-step D2H->H2D round trip (latency; likely incompatible
with straightforward graph capture -> enforce_eager or a custom splitting
op), plus torchax/jax interop plumbing. This is a design-level change, not a
patch. Alternatives (int4-quantizing the PLE table still leaves ~149 GiB on
device; dropping PLE layers changes the model) do not close the gap.

## Update 2026-09-07 (later): PLE host-offload implemented; requant transient is the last wall

The host-offload is now implemented (`ple.py`): the table lives in host RAM
as a plain numpy attribute, `gather_host()` fetches rows per step through
`jax.pure_callback` (host-side e4m3 -> f32 x scale -> bf16 dequantization),
and the load path fills the host table from the checkpoint shards. Verified
by unit tests on CPU and on the 8-chip TPU mesh (jitted: exact match).

With the table off device (~15.3 GiB/chip weights), the engine gets past
weight streaming and MoE requantization now fails with a *visible* error
instead of a silent kill:

    _process_quantized_moe_weights_impl: RESOURCE_EXHAUSTED E0101
    Error loading program 'jit__process_quantized_moe_weights_impl':
    Attempting to reserve 420.19M ... There are 203.73M free.

The 420 MB is the requant program's output allocation (split of merged
w13 into w1/w3 + w2, per chip ~503 MB) coexisting with the pre-split
weights already on device. Knobs tried: MOE_STAGE_WEIGHTS_ON_HOST=true and
VLLM_INCREMENTAL_FP8_LOADING=true + --load-format tpu_streaming_loader
(engaged, but the requant still ran after full placement).
MOE_REQUANTIZE_BLOCK_SIZE cannot help: the reservation is the output
tensors, not the block workspace.

Remaining path to green on v5e-8: run the requant transform on the host
(the staged inputs are already CPU tensors and, for this checkpoint, the
transform is a split/scale re-index - no arithmetic when block boundaries
match) so only the final tensors cross H2D; or repair the incremental
trigger so per-layer processing runs before later layers are placed.

## Update 2026-09-08: requantization now runs; the final wall is arithmetic

With MOE_STAGE_WEIGHTS_ON_HOST + VLLM_INCREMENTAL_FP8_LOADING +
--load-format tpu_streaming_loader (per-layer triggers), plus two new forks
patches — MOE_REQUANTIZE_EXPERT_CHUNK (chunked requant with host staging;
the full-size program's 420M output allocation fails against the staged
weights) and MOE_W13_REORDER_SIZE=1 (the GMM expert grouping pads each
chunk's intermediate 80 -> 128, bloating processed experts by ~60%:
processed 320/160 MiB per chip per layer vs raw 210/105) — the engine
requantizes and places layer after layer. Measured free HBM decays
~430 MiB per layer from 13.1 GiB and the run dies at layer ~30 of 48.

The remaining gap is arithmetic, not engineering: experts 14.36 GiB/chip
(fp8, raw) + non-expert weights 1.27 GiB/chip + jax/runtime overhead
~1.3 GiB/chip (measured: only 13.1 GiB free before the first MoE layer)
= ~16.9 GiB vs 16 GiB HBM. Even with zero leaks and no kernel-layout
overhead the weights alone (15.63 GiB/chip) leave ~0.37 GiB/chip, which
the runtime overhead alone exceeds. v5e-8 cannot serve this checkpoint at
FP8 with the PLE host-offload alone; it needs ~2 GiB/chip more offload
(e.g. 4-bit MoE at checkpoint level, embedding/lm_head offload+fp8, or a
streamed-expert kernel) or a larger pod.

Fixes landed in the fork (all behind env flags, defaults unchanged):
chunked requantization with host staging (MOE_REQUANTIZE_EXPERT_CHUNK),
w13 reorder-size override (MOE_W13_REORDER_SIZE), raw-parameter release
before the per-layer final H2D, and the incremental per-layer trigger
verified working (it fires per layer; the earlier 6-vs-4/expert theory
was wrong — arrivals are 6/expert even for fused-w13 checkpoints).

## Update 2026-09-08 (evening): the engine now loads and requantizes ALL 48 layers

Chain of fixes (all env-gated, defaults preserved):
- chunked requantization (MOE_REQUANTIZE_EXPERT_CHUNK=8) with per-chunk
  host staging and per-device shard placement
  (make_array_from_single_device_arrays — no full-size staging);
- explicit fp8 requant target (the streamed FusedMoE params are bf16, so
  the dtype-derived target kept the processed weights in bf16 — 2x size);
- MOE_W13_REORDER_SIZE=1 (the GMM grouping padded the intermediate
  80 -> 128 per chunk: +60% on the processed experts);
- hyper-connection projections row/column-parallel instead of replicated
  (1.9 GiB/chip -> ~75 MiB/chip across 48 layers);
- rotary cos_sin_cache trimmed to the served context (128 MiB/chip -> 2);
- token embedding table host-offloaded (QWEN4_EXP_HOST_EMBEDDING,
  ~149 MiB/chip), mirroring the PLE table offload;
- gc.collect() per processed layer (cyclic transients ~300 MiB/layer).

Result: all 48 MoE layers requantize and place; the engine reaches the
final shard_model_to_tpu cleanup with ~0.6-0.8 GiB/chip free. The
remaining failure: ONE (512, 1280, 2560) fp8 raw w13 CPU tensor still
reaches the generic replication walk, and its t2j (1.56 GiB on one device)
OOMs. The layer it belongs to is not yet identified (CLNDBG3 name logging
added; the safety net that re-processes CPU RoutedExperts layers did not
fire, so the holder is either a non-RoutedExperts module or a param the
walk sees that the safety net does not). Next step: read the CLNDBG3 name
from the next boot and route that holder through the chunked requant too.

## What DOES work on v5e-8 (verified 2026-09-07)

- full environment bootstrap (`bootstrap.sh`), TPU visibility (8 devices,
  jax 0.11.0 + libtpu 0.0.44 + torchax 0.13 + LKG vLLM),
- architecture registration incl. Qwen4ExpForConditionalGeneration ->
  text-only path, plugin selection via VLLM_PLUGINS=tpu_inference,
- config parsing, model construction, FP8 PLE construction (x64 auto-enable,
  global-scale mapping), 131-shard lazy streaming, MoE requant entry,
- keep-tpu heartbeat loop for long Kaggle sessions.

The gap is purely the v5e-8 HBM capacity vs. the checkpoint's PLE table.
