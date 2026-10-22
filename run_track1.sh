#!/bin/bash
# Noise track 1 on one Cloud.ru node (mlsub): environment from uv.lock, the fixed data, then train/val/run.py on all
# GPUs of the node. The run's own record (config.json with the full recipe, metrics.jsonl, result.json) is printed
# into the job log and copied to the shared disk, so the hyperparameters travel with the numbers.
# RUN_NAME names the run; EXTRA_ARGS go to train/val/run.py.
set -euo pipefail
# mpirun may start this once per slot; only the first copy runs, torchrun takes all GPUs itself.
if [ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]; then exit 0; fi
RUN_NAME=${RUN_NAME:?RUN_NAME required}
OUT=logs/$RUN_NAME
SHARE=/home/jovyan/shares/SR006.nfs2/veprikov/noise/$RUN_NAME
GPUS=$(nvidia-smi -L | wc -l)
echo "=== noise $RUN_NAME commit $(git rev-parse HEAD) gpus $GPUS start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
save() { mkdir -p "$SHARE" && cp -r "$OUT"/. "$SHARE"/ 2>/dev/null || true; echo "=== copied to $SHARE"; }
trap save EXIT
pip install -q uv
uv sync -q
uv run utils/prepare_data.py
uv run torchrun --standalone --nproc_per_node=$GPUS train/val/run.py --output "$OUT" ${EXTRA_ARGS:-}
echo "=== config.json"; cat "$OUT/config.json"
echo "=== result.json"; cat "$OUT/result.json"
echo "=== done $(date -Is)"
