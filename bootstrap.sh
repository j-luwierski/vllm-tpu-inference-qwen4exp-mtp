#!/usr/bin/env bash
#
# bootstrap.sh — reproducible bootstrap of the Qwen3.8-Flash-Next serving
# environment on a fresh Kaggle TPU v5e-8 (Debian) session.
#
# End state (identical to the manually-debugged working state of 2026-09-07):
#   1. fork repo cloned/pulled into /kaggle/working/vllm-tpu-inference-qwen4exp-mtp
#      (branch qwen3.8-flash-next) and installed editable,
#   2. JAX family aligned to tpu-inference 0.28.0 pins
#      (jax/jaxlib 0.11.0, libtpu 0.0.44, torchax 0.0.13, flax 0.12.8, qwix 0.1.2),
#   3. all runtime deps of vllm-tpu present (the Kaggle image misses many),
#   4. vLLM built from the tpu-inference LKG commit 7fbd44cbe0a9...
#      (the image's preinstalled vllm snapshot predates Qwen4Exp entirely and
#      lacks vllm/transformers_utils/configs/qwen4_exp.py, so the port cannot
#      even import against it),
#   5. unified model dir /kaggle/working/qwen3.8-flash-next with:
#        - symlinks to config/tokenizer JSONs (Kaggle config dataset),
#        - symlinks to all 131 safetensors shards (Kaggle fp8 dataset),
#        - model.safetensors.index.json downloaded from Hugging Face
#          (the Kaggle dataset does not ship it; fallback: generated from
#          shard headers),
#   6. (with --start) the API server launched in the background.
#
# The script is idempotent — safe to re-run after a session restart.
#
# Usage:
#   bash bootstrap.sh            # environment + model dir
#   bash bootstrap.sh --start    # ...and start the API server in background
#
set -euo pipefail

REPO_URL="https://github.com/j-luwierski/vllm-tpu-inference-qwen4exp-mtp.git"
BRANCH="qwen3.8-flash-next"
WORK="/kaggle/working"
REPO_DIR="$WORK/vllm-tpu-inference-qwen4exp-mtp"
TPU_IFACE_DIR="$REPO_DIR/tpu-interface"   # NOTE: the subdir really is named
                                          # "tpu-interface" (verified; not a
                                          # typo for "tpu-inference")
MODEL_DIR="$WORK/qwen3.8-flash-next"

# vLLM LKG commit pinned by the fork ("Update vLLM LKG to 7fbd44cb...").
VLLM_LKG_COMMIT="7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719"
VLLM_LKG_DIR="/tmp/vllm-lkg"

# Hugging Face mirror of the checkpoint (for the missing index file and, as a
# fallback, the JSON configs).
HF_BASE="https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/resolve/main"
HF_INDEX="$HF_BASE/model.safetensors.index.json"

SP="/usr/local/lib/python3.12/site-packages"   # Kaggle image python3.12

log() { echo "[bootstrap] $*"; }
die() { echo "[bootstrap] FATAL: $*" >&2; exit 1; }

START_SERVER=0
[ "${1:-}" = "--start" ] && START_SERVER=1

cd "$WORK"

# ---------------------------------------------------------------------------
# 0. Locate the Kaggle dataset mounts (paths verified on 2026-09-07; the
#    datasets/<username>/ segment IS real on Kaggle mounts, but resolve them
#    dynamically in case the mount layout differs).
# ---------------------------------------------------------------------------
CFG_DATASET="$(find /kaggle/input -maxdepth 4 -type d \
    -name 'qwen38-flash-next-config' 2>/dev/null | head -1 || true)"
WGT_DATASET="$(find /kaggle/input -maxdepth 4 -type d \
    -name 'qwen38-flash-next-fp8' 2>/dev/null | head -1 || true)"
[ -n "$CFG_DATASET" ] || die "config dataset (qwen38-flash-next-config) not found under /kaggle/input"
[ -n "$WGT_DATASET" ] || die "weights dataset (qwen38-flash-next-fp8) not found under /kaggle/input"
log "config dataset : $CFG_DATASET"
log "weights dataset: $WGT_DATASET"

# ---------------------------------------------------------------------------
# 1. Fork repo: clone or pull (all fixes are git-committed on the branch).
# ---------------------------------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
    log "cloning $REPO_URL (branch $BRANCH)"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    log "repo present; pulling latest $BRANCH"
    git -C "$REPO_DIR" fetch -q origin "$BRANCH"
    git -C "$REPO_DIR" reset -q --hard "origin/$BRANCH"
fi

# ---------------------------------------------------------------------------
# 2. Given install sequence (vllm-tpu metapackage, --no-deps per plan).
# ---------------------------------------------------------------------------
python3 -c "import importlib.metadata as md; md.version('vllm-tpu')" 2>/dev/null \
    || pip install -q --no-deps vllm-tpu==0.28.0

# ---------------------------------------------------------------------------
# 3. Runtime dependencies missing from the Kaggle image (discovered by
#    import-failure during the 2026-09-07 debugging session).
#    --no-deps everywhere nothing must touch the jax/torch/libtpu family.
# ---------------------------------------------------------------------------
log "installing missing runtime deps"
pip install -q --no-deps \
    anthropic blake3 cachetools cbor2 compressed-tensors depyf fastapi \
    ijson llguidance lm-format-enforcer mistral-common msgspec openai \
    openai-harmony outlines-core partial-json-parser \
    prometheus-fastapi-instrumentator pybase64 ray starlette tiktoken \
    watchfiles xgrammar setproctitle uvicorn gguf semver \
    loguru xprof py-cpuinfo pydantic-extra-types httpx2 apache-tvm-ffi \
    nixl google-cloud-storage pathwaysutils parameterized \
    runai-model-streamer gcsfs hypothesis sortedcontainers einshape \
    model-hosting-container-standards \
    || die "runtime deps install failed"
# These four are small and safe to resolve with deps (they pulled in
# sniffio/httpx/anyio/jiter/mcp_types/jmespath in the session).
pip install -q openai anthropic mcp mistral-common \
    opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp \
    opentelemetry-semantic-conventions-ai jmespath || die "dep-resolution install failed"

# ---------------------------------------------------------------------------
# 4. JAX family aligned to tpu-inference 0.28.0 pins.
#    The image ships jax 0.10.2 + libtpu 0.0.17, under which the installed
#    torchax cannot import ("cannot import name 'mutable_array' from
#    'jax.experimental'"), which made vLLM's platform detection fail with
#    "cannot import name 'current_platform'" and 'import vllm' crash.
#    Verified on TPU after upgrade: jax 0.11.0 still sees all 8 devices.
# ---------------------------------------------------------------------------
if ! python3 -c "import jax; assert jax.__version__ == '0.11.0', jax.__version__" 2>/dev/null; then
    log "aligning JAX family with tpu-inference 0.28.0 pins"
    pip install -q --no-deps --force-reinstall \
        jax==0.11.0 jaxlib==0.11.0 libtpu==0.0.44 \
        torchax==0.0.13 flax==0.12.8 qwix==0.1.2
fi

# tokamax 0.0.13 (tpu-inference pin; provides the GMM kernel used by the MoE)
# and its pinned OLD typeguard (2.x API — 4.x breaks jaxtyping's shape-DSL
# parsing in tokamax's decorators: "SyntaxError: invalid syntax" on import).
pip install -q --no-deps tokamax==0.0.13
pip install -q --no-deps --force-reinstall typeguard==2.13.3

# ---------------------------------------------------------------------------
# 5. vLLM at the LKG commit (built without compiled ops via
#    VLLM_TARGET_DEVICE=tpu). The image's preinstalled vllm is a stale
#    snapshot: it has no qwen4_exp config/model modules and its modelopt.py
#    predates the Role-based API the fork's nvfp4 import needs, so with it
#    vLLM falls back to the built-in CUDA/ROCm-gated Qwen4Exp wrapper and the
#    server dies with "Qwen4Exp currently supports CUDA and ROCm only".
# ---------------------------------------------------------------------------
NEED_VLLM_BUILD=1
if python3 -c "import vllm; assert 'g${VLLM_LKG_COMMIT:0:9}' in vllm.__version__" 2>/dev/null; then
    log "vLLM LKG already installed"
    NEED_VLLM_BUILD=0
fi
if [ "$NEED_VLLM_BUILD" = 1 ]; then
    log "building vLLM from LKG commit $VLLM_LKG_COMMIT"
    pip install -q --no-deps cmake ninja setuptools-rust setuptools-scm \
        vcs-versioning semantic-version
    rm -rf "$VLLM_LKG_DIR"
    mkdir -p "$VLLM_LKG_DIR"
    git -C "$VLLM_LKG_DIR" init -q
    git -C "$VLLM_LKG_DIR" remote add origin https://github.com/vllm-project/vllm.git
    git -C "$VLLM_LKG_DIR" fetch -q --depth 1 origin "$VLLM_LKG_COMMIT"
    git -C "$VLLM_LKG_DIR" checkout -q FETCH_HEAD
    # Remove the stale image snapshot first (it has no dist metadata, so pip
    # would otherwise leave orphaned modules behind); clear any vllm
    # dist-info too so importlib.metadata sees exactly one vllm.
    rm -rf "$SP/vllm"
    rm -rf "$SP"/vllm-*.dist-info 2>/dev/null || true
    VLLM_TARGET_DEVICE=tpu pip install -q --no-build-isolation --no-deps \
        "$VLLM_LKG_DIR"
    # Version check via dist metadata only: `import vllm` cannot work yet on
    # a TPU box — the TPU platform class comes from the tpu_inference plugin,
    # which is installed in the NEXT step.
    python3 -c "import importlib.metadata as md; v = md.version('vllm'); assert 'g${VLLM_LKG_COMMIT:0:9}' in v, v" \
        || die "vLLM LKG build did not produce the expected version"
fi

# ---------------------------------------------------------------------------
# 6. Editable install of the fork (tpu-interface/ is the real subdir name).
# ---------------------------------------------------------------------------
log "installing fork editable"
cd "$TPU_IFACE_DIR"
VLLM_VERSION_OVERRIDE=0.28.0 pip install -q --no-build-isolation --no-deps -e .
grep -q "tpu_inference" "$SP"/tpu_inference-*.dist-info/entry_points.txt \
    || die "fork entry point not registered"

# ---------------------------------------------------------------------------
# 7. Unified model directory (symlinks only — never copy the weights).
# ---------------------------------------------------------------------------
log "assembling model dir $MODEL_DIR"
mkdir -p "$MODEL_DIR"

# 7a. JSON configs/tokenizer: symlink from the config dataset, else download
#     from the HF mirror.
for f in config.json generation_config.json tokenizer.json tokenizer_config.json; do
    if [ ! -e "$MODEL_DIR/$f" ]; then
        if [ -f "$CFG_DATASET/$f" ]; then
            ln -s "$CFG_DATASET/$f" "$MODEL_DIR/$f"
        else
            log "downloading $f from HF (not in dataset)"
            curl -fL --retry 3 -sS "$HF_BASE/$f" -o "$MODEL_DIR/$f"
        fi
    fi
done

# 7b. Safetensors shards: symlink (never copy).
shard_count=0
for f in "$WGT_DATASET"/*.safetensors; do
    [ -e "$f" ] || continue   # glob unmatched
    b="$(basename "$f")"
    [ -e "$MODEL_DIR/$b" ] || ln -s "$f" "$MODEL_DIR/$b"
    shard_count=$((shard_count + 1))
done
[ "$shard_count" -gt 0 ] || die "no safetensors shards found in $WGT_DATASET"
log "linked $shard_count safetensors shards"

# 7c. model.safetensors.index.json — MISSING from the Kaggle dataset.
#     Prefer the official file from HF; fall back to generating it from the
#     shard headers (headers only — weights are never read/copied).
if [ ! -e "$MODEL_DIR/model.safetensors.index.json" ]; then
    log "fetching model.safetensors.index.json from HF"
    if curl -fL --retry 3 -sS "$HF_INDEX" -o "$MODEL_DIR/.index.tmp"; then
        python3 -c "import json,sys; json.load(open('$MODEL_DIR/.index.tmp'))['weight_map']" \
            && mv "$MODEL_DIR/.index.tmp" "$MODEL_DIR/model.safetensors.index.json" \
            || { log "HF index invalid; falling back to header generation"; rm -f "$MODEL_DIR/.index.tmp"; }
    fi
fi
if [ ! -e "$MODEL_DIR/model.safetensors.index.json" ]; then
    log "generating model.safetensors.index.json from shard headers"
    python3 - "$WGT_DATASET" "$MODEL_DIR" <<'PYEOF'
import json, os, struct, glob, sys
src, dst = sys.argv[1], sys.argv[2]
weight_map, total = {}, 0
for path in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
    fname = os.path.basename(path)
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
    for k, v in header.items():
        if k == "__metadata__":
            continue
        weight_map[k] = fname
        lo, hi = v.get("data_offsets", [0, 0])
        total += hi - lo
with open(os.path.join(dst, "model.safetensors.index.json"), "w") as fh:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, fh)
print(f"index: {len(weight_map)} tensors, {total/1e9:.1f} GB")
PYEOF
fi
python3 -c "import json; wm=json.load(open('$MODEL_DIR/model.safetensors.index.json'))['weight_map']; assert len(wm)>100000, len(wm)" \
    || die "model.safetensors.index.json looks wrong"

# ---------------------------------------------------------------------------
# 8. Sanity checks: TPU visible, fork importable, arch registered.
# ---------------------------------------------------------------------------
log "sanity checks"
PJRT_DEVICE=TPU python3 - <<'PYEOF' || die "TPU sanity check failed"
import jax
devices = jax.devices()
assert len(devices) == 8, devices
print("TPU OK:", devices[0])
PYEOF
python3 - <<'PYEOF' || die "fork/registration sanity check failed"
import vllm.plugins
vllm.plugins.load_general_plugins()
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
for a in ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM", "Qwen4ExpMTP"):
    assert a in archs, a
print("Qwen4Exp architectures registered OK")
PYEOF

log "bootstrap complete."
log "server start command:"
echo "  cd $WORK && PJRT_DEVICE=TPU VLLM_PLUGINS=tpu_inference \\"
echo "    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_USE_SHARDY_PARTITIONER=false \\
    MOE_REQUANTIZE_EXPERT_CHUNK=8 MOE_W13_REORDER_SIZE=1 \\
    QWEN4_EXP_HOST_EMBEDDING=true SKIP_JAX_PRECOMPILE=true \\"
echo "    python3 -m vllm.entrypoints.openai.api_server \\"
echo "    --model $MODEL_DIR --host 0.0.0.0 --port 8000 --tensor-parallel-size 8 \\"
echo "    --safetensors-load-strategy lazy --max-model-len 8192 \\
    --max-num-batched-tokens 1024 --max-num-seqs 1"
# NOTE --safetensors-load-strategy lazy: the auto "prefetch" strategy pulls the
# whole 172.78 GiB checkpoint into page cache while the weights themselves are
# also resident during MoE requantization/sharding, which OOM-killed the
# EngineCore (silent SIGKILL, no traceback) on the 377 GiB host.
# NOTE XLA_PYTHON_CLIENT_PREALLOCATE=false: weights take ~15.3 GiB/chip after
# the PLE host-offload; on-demand allocation leaves the remainder for KV cache.
# NOTE JAX_USE_SHARDY_PARTITIONER=false: jax 0.11's Shardy pipeline cannot
# lower eager/traced ops on pure_callback outputs (GSPMDSharding -> SdyArray
# conversion failure) — the PLE host-gather callback requires the legacy
# GSPMD path. --max-model-len 8192 caps the KV budget (~15.3 GiB/chip is
# already taken by weights).

# ---------------------------------------------------------------------------
# 9. Optional: start the server in the background (same command as above).
# ---------------------------------------------------------------------------
if [ "$START_SERVER" = 1 ]; then
    log "starting API server in background (log: $WORK/server.log)"
    cd "$WORK"
    nohup env PJRT_DEVICE=TPU VLLM_PLUGINS=tpu_inference \
        XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_USE_SHARDY_PARTITIONER=false \
        MOE_REQUANTIZE_EXPERT_CHUNK=8 MOE_W13_REORDER_SIZE=1 \
        QWEN4_EXP_HOST_EMBEDDING=true SKIP_JAX_PRECOMPILE=true \
        python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" --host 0.0.0.0 --port 8000 \
        --tensor-parallel-size 8 --safetensors-load-strategy lazy \
        --max-model-len 512 --max-num-batched-tokens 16 \
        --max-num-seqs 1 > "$WORK/server.log" 2>&1 &
    echo $! > "$WORK/server.pid"
    log "server pid $(cat "$WORK/server.pid"); follow with: tail -f $WORK/server.log"
fi
