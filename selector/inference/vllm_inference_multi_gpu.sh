#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
MODEL_NAME=""
INPUT_DIR=""
OUTPUT_DIR=""
SUFFIX="_predictions"
MAX_TOKENS=200
TEMPERATURE=0.01
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=8000
CHUNK_SIZE=2500
ENFORCE_EAGER=false
TENSOR_PARALLEL_SIZE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_SCRIPT="$SCRIPT_DIR/vllm_inference_multi_gpu.py"

# --- Argument parsing ---
usage() {
    echo "Usage: $0 --model <model_name> --input_dir <dir> --output_dir <dir> [options]"
    echo ""
    echo "Required:"
    echo "  --model                  HuggingFace model name or local path"
    echo "  --input_dir              Folder containing .jsonl input files"
    echo "  --output_dir             Folder to write output .jsonl files"
    echo ""
    echo "Optional:"
    echo "  --suffix                 Output filename suffix (default: _predictions)"
    echo "  --max_tokens             Max tokens to generate (default: 2000)"
    echo "  --temperature            Sampling temperature (default: 0.01)"
    echo "  --gpu_memory_utilization GPU memory fraction for vLLM (default: 0.9)"
    echo "  --max_model_len          Max model context length (default: 8000)"
    echo "  --chunk_size             Inference chunk size (default: 2500)"
    echo "  --tensor_parallel_size   Number of GPUs for tensor parallelism (default: 1)"
    echo "  --enforce_eager          Disable CUDA graphs — slower, for debugging"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                  MODEL_NAME="$2";                shift 2 ;;
        --input_dir)              INPUT_DIR="$2";                 shift 2 ;;
        --output_dir)             OUTPUT_DIR="$2";                shift 2 ;;
        --suffix)                 SUFFIX="$2";                    shift 2 ;;
        --max_tokens)             MAX_TOKENS="$2";                shift 2 ;;
        --temperature)            TEMPERATURE="$2";               shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2";    shift 2 ;;
        --max_model_len)          MAX_MODEL_LEN="$2";             shift 2 ;;
        --chunk_size)             CHUNK_SIZE="$2";                shift 2 ;;
        --tensor_parallel_size)   TENSOR_PARALLEL_SIZE="$2";      shift 2 ;;
        --enforce_eager)          ENFORCE_EAGER=true;             shift 1 ;;
        --help)                   usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# --- Validation ---
[[ -z "$MODEL_NAME" ]]  && { echo "Error: --model is required";      usage; }
[[ -z "$INPUT_DIR" ]]   && { echo "Error: --input_dir is required";  usage; }
[[ -z "$OUTPUT_DIR" ]]  && { echo "Error: --output_dir is required"; usage; }
[[ -d "$INPUT_DIR" ]]   || { echo "Error: input_dir '$INPUT_DIR' does not exist"; exit 1; }

# FIX: Check inference script exists before starting the loop
[[ -f "$INFERENCE_SCRIPT" ]] || { echo "Error: inference script not found at $INFERENCE_SCRIPT"; exit 1; }

mkdir -p "$OUTPUT_DIR"

# FIX: Use an array for extra flags instead of a plain string
EXTRA_FLAGS=()
[[ "$ENFORCE_EAGER" == true ]] && EXTRA_FLAGS+=(--enforce_eager)

# --- Trap: clean up partial output on unexpected interrupt ---
# FIX: Trap only INT/TERM so normal exits don't trigger cleanup
OUTPUT_FILE=""
cleanup() {
    if [[ -n "$OUTPUT_FILE" && -f "$OUTPUT_FILE" ]]; then
        echo "Interrupted — removing partial output: $OUTPUT_FILE"
        rm -f "$OUTPUT_FILE"
    fi
    exit 130
}
trap cleanup INT TERM

# --- Main loop ---
TOTAL=0
SKIPPED=0
COMPLETED=0
FAILED=0

shopt -s nullglob
INPUT_FILES=("$INPUT_DIR"/*.jsonl)
shopt -u nullglob

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    echo "No .jsonl files found in $INPUT_DIR"
    exit 0
fi

TOTAL=${#INPUT_FILES[@]}
echo "================================================"
echo "Model:      $MODEL_NAME"
echo "Input dir:  $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Suffix:     $SUFFIX"
echo "TP size:    $TENSOR_PARALLEL_SIZE"
echo "Files:      $TOTAL"
echo "================================================"

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    BASENAME=$(basename "$INPUT_FILE" .jsonl)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}${SUFFIX}.jsonl"  # kept in scope for trap

    echo ""
    echo "[$(date '+%H:%M:%S')] Processing: ${BASENAME}.jsonl"

    if [[ -f "$OUTPUT_FILE" ]]; then
        echo "  → Skipping: output already exists at $OUTPUT_FILE"
        (( SKIPPED++ )) || true
        OUTPUT_FILE=""  # don't let trap delete a pre-existing file
        continue
    fi

    # FIX: Replaced `accelerate launch` with plain python — vLLM manages its own GPU setup
    if python "$INFERENCE_SCRIPT" \
            --model_name "$MODEL_NAME" \
            --input_file "$INPUT_FILE" \
            --output_file "$OUTPUT_FILE" \
            --max_tokens "$MAX_TOKENS" \
            --temperature "$TEMPERATURE" \
            --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
            --max_model_len "$MAX_MODEL_LEN" \
            --chunk_size "$CHUNK_SIZE" \
            --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
            "${EXTRA_FLAGS[@]}"; then
        echo "  ✓ Done: $OUTPUT_FILE"
        (( COMPLETED++ )) || true
        OUTPUT_FILE=""  # success — trap should not delete it
    else
        # FIX: Capture exit code before it gets overwritten
        EXIT_CODE=$?
        echo "  ✗ Failed: ${BASENAME}.jsonl (exit code $EXIT_CODE)"
        (( FAILED++ )) || true
        [[ -f "$OUTPUT_FILE" ]] && rm -f "$OUTPUT_FILE"
        OUTPUT_FILE=""  # already cleaned up, trap should not touch it
    fi
done

# --- Summary ---
echo ""
echo "================================================"
echo "Summary:"
echo "  Total:     $TOTAL"
echo "  Completed: $COMPLETED"
echo "  Skipped:   $SKIPPED"
echo "  Failed:    $FAILED"
echo "================================================"

[[ $FAILED -gt 0 ]] && exit 1 || exit 0
