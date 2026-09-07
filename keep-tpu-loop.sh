#!/usr/bin/env bash
# keep-tpu-loop.sh — run keep-tpu.py every ~20 min, but only while the vLLM
# API server is NOT running (a second jax process cannot coexist with the
# server's preallocated HBM). While the server runs, the TPU is busy anyway.
#
# Usage: nohup bash keep-tpu-loop.sh > /kaggle/working/keep-tpu-loop.log 2>&1 &
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${KEEP_TPU_INTERVAL:-1200}"   # 20 min default (Kaggle TPU-idle
                                        # timeout hit at >25 min idle)
while true; do
    if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
        echo "[keep-tpu-loop] $(date -u '+%F %T') server running -> TPU active, skip ping"
    else
        echo "[keep-tpu-loop] $(date -u '+%F %T') server not running -> pinging TPU"
        python3 "$HERE/keep-tpu.py" 2>&1 | tail -3 || echo "[keep-tpu-loop] ping failed (will retry)"
    fi
    sleep "$INTERVAL"
done
