import asyncio
import aiohttp
import json
import os
import random
from datasets import load_dataset

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1/chat/completions")
CONCURRENCY = int(os.environ.get("CONCURRENCY", 32))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 5))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", 1.5))


# -------------------------
# Resume logic
# -------------------------
def load_completed_indices(output_file):
    completed = set()
    if not os.path.exists(output_file):
        return completed

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                completed.add(json.loads(line)["index"])
            except Exception:
                continue

    print(f"[Resume] Found {len(completed)} completed samples")
    return completed


# -------------------------
# Retry wrapper
# -------------------------
async def fetch_with_retry(session, payload, idx):
    delay = 1.0

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(
                VLLM_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")

                result = await resp.json()
                text = result["choices"][0]["message"]["content"]
                return idx, text

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                return idx, f"ERROR: {str(e)}"

            # exponential backoff + jitter
            sleep_time = delay * (BACKOFF_BASE ** attempt) * (1 + random.random())
            print(f"[Retry] idx={idx} attempt={attempt+1} sleeping={sleep_time:.2f}s error={e}")
            await asyncio.sleep(sleep_time)


# -------------------------
# Worker
# -------------------------
async def worker(queue, session, file_lock, output_file):
    while True:
        item = await queue.get()
        if item is None:
            break

        idx, messages = item

        payload = {
            "model": "Qwen/Qwen3-32B",
            "messages": messages,
            "temperature": 0.01,
            "max_tokens": 2000,
        }

        idx, text = await fetch_with_retry(session, payload, idx)

        # atomic write with lock
        async with file_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"index": idx, "prediction": text}, ensure_ascii=False) + "\n")

        queue.task_done()


# -------------------------
# Main
# -------------------------
async def main(input_file, output_file):
    ds = load_dataset("json", data_files={"test": input_file})["test"]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    completed = load_completed_indices(output_file)

    queue = asyncio.Queue()

    total = 0
    skipped = 0

    for row in ds:
        idx = row["index"]
        if idx in completed:
            skipped += 1
            continue

        queue.put_nowait((idx, row["messages"]))
        total += 1

    print(f"[Queue] Pending: {total}, Skipped: {skipped}")

    file_lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)

    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [
            asyncio.create_task(worker(queue, session, file_lock, output_file))
            for _ in range(CONCURRENCY)
        ]

        await queue.join()

        for _ in workers:
            await queue.put(None)

        await asyncio.gather(*workers)

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    asyncio.run(main(args.input_file, args.output_file))