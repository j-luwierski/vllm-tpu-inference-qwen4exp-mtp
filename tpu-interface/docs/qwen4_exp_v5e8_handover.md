# v5e-8 handover: Qwen3.8-Flash-Next loading state (2026-09-08)

## Status
All 48 MoE layers requantize and place on the TPU. The load now fails only
in the final `shard_model_to_tpu` walk on ONE leftover CPU
(512, 1280, 2560) fp8 `w13` tensor whose `t2j` needs 1.56 GiB on one device.

## Launch command (all flags are required)
```
cd /kaggle/working && bash /kaggle/working/vllm-tpu-inference-qwen4exp-mtp/bootstrap.sh
# bootstrap.sh --start runs:
PJRT_DEVICE=TPU VLLM_PLUGINS=tpu_inference \
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_USE_SHARDY_PARTITIONER=false \
  MOE_STAGE_WEIGHTS_ON_HOST=true VLLM_INCREMENTAL_FP8_LOADING=true \
  MOE_REQUANTIZE_EXPERT_CHUNK=8 MOE_W13_REORDER_SIZE=1 \
  QWEN4_EXP_HOST_EMBEDDING=true SKIP_JAX_PRECOMPILE=true \
  python3 -m vllm.entrypoints.openai.api_server \
  --model /kaggle/working/qwen3.8-flash-next --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 --safetensors-load-strategy lazy \
  --max-model-len 512 --max-num-batched-tokens 16 --max-num-seqs 1 \
  --load-format tpu_streaming_loader
```

## Flag rationale
- MOE_STAGE_WEIGHTS_ON_HOST + VLLM_INCREMENTAL_FP8_LOADING +
  tpu_streaming_loader: per-layer incremental processing (weights stay on
  CPU until their layer completes; per-chip placement).
- MOE_REQUANTIZE_EXPERT_CHUNK=8: requantize in 8-expert chunks with host
  staging (a full-size program needs 1.56 GiB transient).
- MOE_W13_REORDER_SIZE=1: skip the GMM grouping whose 128-alignment pads
  the intermediate 640 -> 1024 (+60% on the processed experts).
- QWEN4_EXP_HOST_EMBEDDING: token embedding table in host RAM
  (~149 MiB/chip), rows fetched via jax.pure_callback.
- SKIP_JAX_PRECOMPILE=true: skip the precompile/autotune pass (it needs
  hundreds of MiB of workspace); the first request compiles lazily.
- XLA_PYTHON_CLIENT_PREALLOCATE=false: on-demand HBM.

## Host-offloaded tensors (never on device)
- PLE n-gram table 47.75 GiB (ple.py host_table + gather_host).
- Token embedding table 1.19 GiB (model.py Qwen4ExpHostEmbedding).

## Next step (the last known failure)
1. Boot with the command above (load ~13 min).
2. On failure, read the `[CLNDBG3] <name> shape=... free_before=...` line
   in /kaggle/working/server.log — it names the CPU parameter that reached
   the generic replication walk with >100 MiB.
3. Route that holder through the chunked requant path (see
   cleanup_sharding._shard_named: params under `mlp.experts.` are already
   skipped; the CLNDBG3 name tells which OTHER module still holds a full
   CPU copy), or extend the skip list.
4. After the walk completes: Memory statistics -> KV cache allocation ->
   capture_model skipped -> first request compiles lazily (1-5 min) ->
   /v1/chat/completions.
5. Remove the [CHUNKDBG]/[CLNDBG] prints once green.

## Session protection
keep-tpu-loop.sh pings the TPU for 60 s every 20 min (skip when api_server
is running). Keep it alive or the session dies in ~2 h idle.
