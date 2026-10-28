#!/bin/bash
# Noise 30M speedrun (train/speedrun/run.py with the Tricks flags) on one Cloud.ru node (mlsub), under the lab
# recorder (lab_noise.py). The data are built once and kept on the shared disk; the run's config.json, metrics.jsonl,
# result.json and recorder page are copied there too. RUN_NAME names the run; arguments go to $SCRIPT (default train/val/run.py).
set -euo pipefail
if [ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]; then exit 0; fi
RUN_NAME=${RUN_NAME:?RUN_NAME required}
ROOT=/home/jovyan/shares/SR006.nfs2/veprikov/noise
OUT=logs/$RUN_NAME
SHARE=$ROOT/$RUN_NAME
GPUS=$(nvidia-smi -L | wc -l)
export NOISE_COMMIT=$(git rev-parse HEAD) LAB_RUN_WORK=zo-pretrain LAB_RUN_NAME=$RUN_NAME LAB_RUN_OFFLINE=1
export LAB_RUN_DIR=logs/lab-runs LAB_RUN_OBJECTIVE="Noise 30M speedrun, optimizer tricks: ${*:-record}"
echo "=== noise $RUN_NAME commit $NOISE_COMMIT gpus $GPUS start $(date -Is) args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
save() { mkdir -p "$SHARE" && cp -r "$OUT"/. "$SHARE"/ 2>/dev/null; cp -r logs/lab-runs "$SHARE"/ 2>/dev/null || true; echo "=== copied to $SHARE"; }
trap save EXIT
for i in 1 2 3 4 5; do pip install -q --timeout 120 uv && break; sleep 30; done
for i in 1 2 3 4 5; do UV_HTTP_TIMEOUT=120 uv sync -q && break; sleep 30; done
mkdir -p data
if [ -s "$ROOT/data30m/train.pt" ] && [ -s "$ROOT/data30m/val.pt" ]; then
  cp "$ROOT/data30m/train.pt" "$ROOT/data30m/val.pt" data/
else
  uv run utils/prepare_data.py || true
  test -s data/train.pt && test -s data/val.pt
  mkdir -p "$ROOT/data30m" && cp data/train.pt data/val.pt "$ROOT/data30m/"
fi
# a resumed run starts from the share's checkpoint of the same name
if [ -s "$SHARE/checkpoint.pt" ] && [ ! -s "$SHARE/result.json" ]; then mkdir -p "$OUT" && cp -r "$SHARE"/. "$OUT"/; fi
for v in $(env | grep -oE "^(OMPI|PMIX|PMI)[A-Za-z0-9_]*"); do unset "$v"; done
uv run torchrun --standalone --nproc_per_node=$GPUS lab_noise.py ${SCRIPT:-train/val/run.py} --output "$OUT" "$@" 2>&1
echo "=== config.json"; cat "$OUT/config.json"
echo "=== result.json"; cat "$OUT/result.json" 2>/dev/null || true
echo "=== done $(date -Is)"
