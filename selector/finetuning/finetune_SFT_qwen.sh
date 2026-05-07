#!/bin/bash
#SBATCH -J finetune_llms_SFT
#Set job requirements
#SBATCH -N 1
#SBATCH -t 00:40:00
#SBATCH -p gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END


#Loading modules
module purge
module load 2025
module load CUDA/12.8.0
module load cuDNN/9.10.1.4-CUDA-12.8.0

#Setup paths and data
cp -r $HOME/llm_alignment_for_ED "$TMPDIR"
cd $TMPDIR/llm_alignment_for_ED

#manage environment
uv cache clean

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

echo "Starting training job at $(date)"

#Execute the Python program.
accelerate launch --num_processes=1 \
                    --num_machines=1 \
                    --mixed_precision="bf16" \
                    SFT_ED.py \
                    --epochs 1 \
                    --LLM_model "Qwen/Qwen3-8B" \
                    --train_data random_train/completion_style/first_1K/prompts_completion_style_train.jsonl \
                    --val_data random_train/completion_style/first_1K/prompts_completion_style_val.jsonl \
                    --ftuned_model_name "RAISED_Qwen" \
                    --logging_dir "$TMPDIR/1K_RAISED_Qwen3_8B" \

Copy output directory from scratch to home
cp -r "$TMPDIR"/1K_RAISED_Qwen3_8B $HOME/llm_alignment_for_ED/training_logs

echo "Job finished at $(date)"