import json
import os
import argparse
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="vLLM Batch Inference")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8000)
    parser.add_argument("--enforce_eager", action="store_true", help="Disable CUDA graphs (slower, use for debugging)")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                        help="Number of GPUs for pipeline parallelism (use instead of TP for BnB quantized models)")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization method (e.g. 'bitsandbytes', 'awq', 'fp8'). None = auto-detect from config.")
    parser.add_argument("--load_format", type=str, default="auto",
                        help="Weight load format. Use 'bitsandbytes' for BnB checkpoints.")
    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # 1. Load data
    print(f"Loading data from {args.input_file}...")
    ds = load_dataset("json", data_files={"test": args.input_file})["test"]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("Sample:", ds[0])

    # FIX #6 – Validate output dir exists before loading the model
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # FIX #7 – Resume by collecting already-written indices instead of counting lines
    completed_indices = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed_indices.add(json.loads(line)["index"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(completed_indices)} samples already completed...")

    # 3. Initialize vLLM
    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        quantization=args.quantization,
        load_format=args.load_format,
    )

    # 4. Run inference in chunks, appending to output
    total = len(ds)
    print(f"Total samples: {total}. Processing in chunks of {args.chunk_size}...")

    with open(args.output_file, "a", encoding="utf-8") as f:
        for i in range(0, total, args.chunk_size):
            chunk_ds = ds.select(range(i, min(i + args.chunk_size, total)))

            # FIX #7 – Skip rows already present in the output file
            pending_indices = [j for j in range(len(chunk_ds)) if chunk_ds[j]["index"] not in completed_indices]
            if not pending_indices:
                print(f"Chunk {i // args.chunk_size + 1}: all samples already completed, skipping.")
                continue

            pending_ds = chunk_ds.select(pending_indices)
            chunk_messages = pending_ds["messages"]

            # FIX #10 – Log chunk progress
            chunk_num = i // args.chunk_size + 1
            total_chunks = (total + args.chunk_size - 1) // args.chunk_size
            print(f"Chunk {chunk_num}/{total_chunks}: processing {len(pending_ds)} samples (indices {i}–{min(i + args.chunk_size, total) - 1})...")

            # FIX #4 – Error handling around inference
            try:
                outputs = llm.chat(chunk_messages, sampling_params=sampling_params, use_tqdm=True)
            except Exception as e:
                print(f"Chunk {chunk_num}/{total_chunks} failed during inference: {e}. Skipping.")
                continue

            for j, output in enumerate(outputs):
                row = pending_ds[j]
                res = {
                    "index": row["index"],
                    "mention": row["mention"],
                    "prediction": output.outputs[0].text.strip(),
                    "correct_answer": row["correct_answer"],
                    "candidates": row["candidates"],
                }
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

            # FIX #3 – Flush after each chunk to avoid data loss on crash
            f.flush()
            os.fsync(f.fileno())

            torch.cuda.empty_cache()

    print(f"Done! Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
