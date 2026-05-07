#!/bin/bash
#SBATCH -J vllm_batch_client_ED
#Set job requirements
#SBATCH -N 1
#SBATCH -t 10:10:00

set -u  # catch unset vars; don't use -e because we want to keep going on per-file errors

# Loading modules
module purge
module load 2025
module load CUDA/12.8.0
module load cuDNN/9.10.1.4-CUDA-12.8.0

# Manage environment
source $HOME/.venv/bin/activate

# Env for HF / downloader (same as serve/client)
export HF_HOME=$TMPDIR/huggingface
export UV_LINK_MODE=copy
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_READ_TIMEOUT=300

echo "Starting batch inference job at $(date)"

# -------------------------
# Arguments
# -------------------------
INPUT_DIR=$1
OUTPUT_DIR=$2
SERVER_INFO=$3

if [ -z "${INPUT_DIR:-}" ] || [ -z "${OUTPUT_DIR:-}" ] || [ -z "${SERVER_INFO:-}" ]; then
  echo "Usage: sbatch batch_client.slurm <input_dir> <output_dir> <server_info.json>"
  echo ""
  echo "  <input_dir>        Folder containing .jsonl input files"
  echo "  <output_dir>       Folder where predictions will be written"
  echo "  <server_info.json> Path to server_info_<JOB_ID>.json produced by serve.slurm"
  exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
  echo "ERROR: input dir does not exist: $INPUT_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# -------------------------
# Wait for server_info.json to appear (serve job may not have written it yet)
# -------------------------
echo "Waiting for server info file: $SERVER_INFO"
WAIT_SECS=0
MAX_WAIT=12800   
while [ ! -f "$SERVER_INFO" ]; do
  if [ $WAIT_SECS -ge $MAX_WAIT ]; then
    echo "ERROR: timed out waiting for $SERVER_INFO after ${MAX_WAIT}s"
    exit 1
  fi
  sleep 10
  WAIT_SECS=$((WAIT_SECS + 10))
done
echo "Found server info after ${WAIT_SECS}s"

NODE=$(python -c "import json; print(json.load(open('$SERVER_INFO'))['node'])")
PORT=$(python -c "import json; print(json.load(open('$SERVER_INFO'))['port'])")

export VLLM_URL="http://${NODE}:${PORT}/v1/chat/completions"
export CONCURRENCY=${CONCURRENCY:-24}

# Health endpoint is at /health on the vLLM OpenAI server
HEALTH_URL="http://${NODE}:${PORT}/health"

echo "Using server: $VLLM_URL"
echo "Health URL:   $HEALTH_URL"

# -------------------------
# Wait for server to be ready (model loaded)
# -------------------------
echo "Waiting for vLLM server to become healthy..."
WAIT_SECS=0
MAX_WAIT=1800   # 30 minutes for model load
until curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; do
  if [ $WAIT_SECS -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server did not become healthy after ${MAX_WAIT}s"
    exit 1
  fi
  sleep 10
  WAIT_SECS=$((WAIT_SECS + 10))
  echo "  ...still waiting (${WAIT_SECS}s)"
done
echo "Server is healthy after ${WAIT_SECS}s"

# -------------------------
# Loop over input files
# -------------------------
shopt -s nullglob
# Accept .jsonl (and .json) inputs. Adjust the glob if you use a different extension.
INPUT_FILES=( "$INPUT_DIR"/*.jsonl "$INPUT_DIR"/*.json )

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
  echo "ERROR: no .jsonl or .json files found in $INPUT_DIR"
  exit 1
fi

echo "Found ${#INPUT_FILES[@]} input file(s) in $INPUT_DIR"

# Marker directory for "this file is fully done" (folder-level resume).
# Per-sample resume inside a file is already handled by async_client.py via load_completed_indices().
DONE_DIR="$OUTPUT_DIR/.done"
mkdir -p "$DONE_DIR"

for INPUT_FILE in "${INPUT_FILES[@]}"; do
  BASENAME=$(basename "$INPUT_FILE")
  STEM="${BASENAME%.*}"
  OUTPUT_FILE="$OUTPUT_DIR/${STEM}.predictions.jsonl"
  DONE_MARKER="$DONE_DIR/${BASENAME}.done"

  echo ""
  echo "========================================"
  echo "File:   $INPUT_FILE"
  echo "Output: $OUTPUT_FILE"
  echo "========================================"

  if [ -f "$DONE_MARKER" ]; then
    echo "[Skip] Already completed (marker exists): $DONE_MARKER"
    continue
  fi

  # async_client.py itself resumes from any existing OUTPUT_FILE by reading completed indices.
  START_TS=$(date +%s)
  python $HOME/RAGED/scripts/async_client.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"
  RC=$?
  END_TS=$(date +%s)
  ELAPSED=$((END_TS - START_TS))

  if [ $RC -eq 0 ]; then
    touch "$DONE_MARKER"
    echo "[Done] $BASENAME in ${ELAPSED}s"
  else
    echo "[Fail] $BASENAME exited with code $RC after ${ELAPSED}s — will be retried on next run"
  fi
done

echo ""
echo "Finished batch at $(date)"
