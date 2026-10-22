#!/bin/bash
# Noise track 2 (speedrun) on one Cloud.ru node (mlsub): environment from uv.lock, the fixed data, then train/speedrun/run.py on all
# GPUs of the node. The run's own record (config.json with the full recipe, metrics.jsonl, result.json) is printed
# into the job log and copied to the shared disk, so the hyperparameters travel with the numbers.
# RUN_NAME names the run; EXTRA_ARGS and the script's own arguments go to train/speedrun/run.py.
set -euo pipefail
# mpirun may start this once per slot; only the first copy runs, torchrun takes all GPUs itself.
if [ "${OMPI_COMM_WORLD_RANK:-0}" != 0 ]; then exit 0; fi
RUN_NAME=${RUN_NAME:?RUN_NAME required}
OUT=logs/$RUN_NAME
SHARE=/home/jovyan/shares/SR006.nfs2/veprikov/noise/$RUN_NAME
GPUS=$(nvidia-smi -L | wc -l)
echo "=== noise $RUN_NAME commit $(git rev-parse HEAD) gpus $GPUS start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
save() { mkdir -p "$SHARE" && cp -r "$OUT"/. "$SHARE"/ 2>/dev/null || true; echo "=== copied to $SHARE"; }
trap save EXIT
# the node's network to PyPI drops at times: retry the installs
for i in 1 2 3 4 5; do pip install -q --timeout 120 uv && break; sleep 30; done
for i in 1 2 3 4 5; do UV_HTTP_TIMEOUT=120 uv sync -q && break; sleep 30; done
# prepare_data.py can abort at interpreter exit ("terminate called without an active exception") after it has
# written and checked the files; the run only needs the files.
uv run utils/prepare_data.py || true
test -s data/train.pt && test -s data/val.pt
# mpirun leaves OMPI/PMIX variables that torch.distributed would read; torchrun sets its own.
for v in $(env | grep -oE "^(OMPI|PMIX|PMI)[A-Za-z0-9_]*"); do unset "$v"; done
uv run torchrun --standalone --nproc_per_node=$GPUS train/speedrun/run.py --output "$OUT" ${EXTRA_ARGS:-} "$@" 2>&1
echo "=== config.json"; cat "$OUT/config.json"
echo "=== result.json"; cat "$OUT/result.json"
echo "=== done $(date -Is)"
