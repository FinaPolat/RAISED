#!/bin/bash
#SBATCH -J disambiguate_with_OpenAI
#Set job requirements
#SBATCH -N 1
#SBATCH -t 00:30:00

#Loading modules
module purge
module load 2025

source $HOME/.venv/bin/activate

uv pip install openai tqdm

# Setup paths and data
cp -r $HOME/VerbalizED/prompting "$TMPDIR"
cd $TMPDIR/prompting
cp -r $HOME/llm_alignment_for_ED/prompts/test/compiled_api_prompts "$TMPDIR/prompting"



INPUT_DIR="$TMPDIR/prompting/compiled_api_prompts"
OUTPUT_DIR="$HOME/llm_alignment_for_ED/prompts/test/raged_gpt5_4_mini_output"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.jsonl; do
    filename=$(basename "$file")
    outfile="$OUTPUT_DIR/$filename"

    echo "Processing $file -> $outfile"

    python link_openai_gpt5.py \
        --input "$file" \
        --output "$outfile" \

done

echo "All files processed."