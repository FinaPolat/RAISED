#!/bin/bash
# Evaluate every prompts file in INPUT_DIR against its matching predictions
# file in PRED_DIR, writing per-file results to SCORE_DIR.
#
# Usage:
#   ./evaluate_folder.sh <INPUT_DIR> <PRED_DIR> <SCORE_DIR>
#
# Expects prediction files to follow the naming convention produced by
# batch_client.slurm:  <stem>.jsonl  ->  <stem>.predictions.jsonl

set -u

INPUT_DIR=${1:-}
PRED_DIR=${2:-}
SCORE_DIR=${3:-}

if [ -z "$INPUT_DIR" ] || [ -z "$PRED_DIR" ] || [ -z "$SCORE_DIR" ]; then
    echo "Usage: $0 <INPUT_DIR> <PRED_DIR> <SCORE_DIR>"
    echo ""
    echo "  <INPUT_DIR>  Folder of prompts .jsonl files (with index, correct_answer, candidates)"
    echo "  <PRED_DIR>   Folder of predictions .jsonl files (named <stem>.predictions.jsonl)"
    echo "  <SCORE_DIR>  Folder where per-file evaluation JSONs will be written"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: input dir does not exist: $INPUT_DIR"
    exit 1
fi
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: prediction dir does not exist: $PRED_DIR"
    exit 1
fi

mkdir -p "$SCORE_DIR"

# Resolve the evaluator path: same dir as this script by default, or override
# with EVALUATOR=/path/to/evaluate_output.py ./evaluate_folder.sh ...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATOR=${EVALUATOR:-"$SCRIPT_DIR/evaluate_output.py"}

if [ ! -f "$EVALUATOR" ]; then
    echo "ERROR: evaluator not found at $EVALUATOR"
    echo "       Set EVALUATOR=/path/to/evaluate_output.py to override."
    exit 1
fi

shopt -s nullglob
PROMPT_FILES=( "$INPUT_DIR"/*.jsonl )

if [ ${#PROMPT_FILES[@]} -eq 0 ]; then
    echo "ERROR: no .jsonl files found in $INPUT_DIR"
    exit 1
fi

echo "Found ${#PROMPT_FILES[@]} prompt file(s) in $INPUT_DIR"
echo "Evaluator: $EVALUATOR"
echo ""

n_ok=0
n_skip=0
n_fail=0

for prompts in "${PROMPT_FILES[@]}"; do
    stem=$(basename "$prompts" .jsonl)
    preds="$PRED_DIR/${stem}.predictions.jsonl"
    result="$SCORE_DIR/${stem}_results.json"

    echo "----------------------------------------"
    echo "Prompts:     $prompts"
    echo "Predictions: $preds"
    echo "Results:     $result"

    if [ ! -f "$preds" ]; then
        echo "[skip] no predictions file for $stem"
        n_skip=$((n_skip + 1))
        continue
    fi

    python "$EVALUATOR" \
        --prompts_file  "$prompts" \
        --LLM_answers   "$preds" \
        --output_folder "$SCORE_DIR" \
        --experiment    "${stem}_results.json"
    rc=$?

    if [ $rc -eq 0 ]; then
        n_ok=$((n_ok + 1))
    else
        echo "[fail] evaluator exited with code $rc on $stem"
        n_fail=$((n_fail + 1))
    fi
done

echo ""
echo "========================================"
echo "Evaluated: $n_ok"
echo "Skipped:   $n_skip (no predictions file)"
echo "Failed:    $n_fail"
echo "Results in: $SCORE_DIR"