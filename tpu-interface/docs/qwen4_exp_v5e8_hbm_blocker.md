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

## What DOES work on v5e-8 (verified 2026-09-07)

- full environment bootstrap (`bootstrap.sh`), TPU visibility (8 devices,
  jax 0.11.0 + libtpu 0.0.44 + torchax 0.13 + LKG vLLM),
- architecture registration incl. Qwen4ExpForConditionalGeneration ->
  text-only path, plugin selection via VLLM_PLUGINS=tpu_inference,
- config parsing, model construction, FP8 PLE construction (x64 auto-enable,
  global-scale mapping), 131-shard lazy streaming, MoE requant entry,
- keep-tpu heartbeat loop for long Kaggle sessions.

The gap is purely the v5e-8 HBM capacity vs. the checkpoint's PLE table.
