#!/bin/bash
#SBATCH -J vllm_inference_ED
#Set job requirements
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=2

#Loading modules
module purge
module load 2025
module load CUDA/12.8.0
module load cuDNN/9.10.1.4-CUDA-12.8.0

#Setup paths and data
cp -r $HOME/llm_alignment_for_ED "$TMPDIR"
cd $TMPDIR/llm_alignment_for_ED

#manage environment
source $HOME/.venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install vllm --torch-backend=cu128
uv pip install unsloth unsloth_zoo bitsandbytes transformers datasets accelerate peft trl
uv pip install -U "triton>=3.3.1"
uv pip install tensorboard huggingface-hub[hf_transfer]

# TMPDIR is local to the node (fastest). We put the model cache there.
export HF_HOME=$TMPDIR/huggingface
export UV_LINK_MODE=copy
export HF_HUB_ENABLE_HF_TRANSFER=1   # Use the turbo downloader
export HF_HUB_READ_TIMEOUT=300       # Be patient if HF is slow (5 mins)

echo "Starting inference job at $(date)"

# ================================================================
# SLURM Inference Job — QWEN3-32B bf16 on 2x A100
# Usage:
#   sbatch run_vllm_inference.slurm \
#     --model Qwen/Qwen3-32B \
#     --input_dir data/inputs \
#     --output_dir data/outputs \
#     --tensor_parallel_size 2
# ================================================================

set -euo pipefail

# --- Environment setup ---
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $SLURM_JOB_GPUS"
echo "Started:     $(date)"
echo ""

# Create log directory if it doesn't exist
mkdir -p logs

# --- Run the inference loop ---
bash vllm_inference_multi_gpu.sh "$@"

echo ""
echo "Finished: $(date)"
