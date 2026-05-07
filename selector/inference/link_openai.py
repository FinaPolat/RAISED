import json
import asyncio
import random
import argparse
import logging
from tqdm import tqdm
from openai import AsyncOpenAI, BadRequestError, APIError, RateLimitError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key="your_api_key_here")  # set your API key here


def parse_args():
    parser = argparse.ArgumentParser(description="Async Entity Linking")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=30)  # this can be tuned
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def sanitize_messages(messages):
    clean = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)

        content = content.replace("\x00", " ").strip()
        clean.append({"role": role, "content": content})
    return clean


# FIX 1: messages is now a local variable that can be reassigned across retries.
# FIX 2: BadRequestError attempts sanitization once, then retries; permanent errors exit early.
# FIX 3: asyncio.wait_for enforces a per-request timeout so hung calls don't block semaphore slots.
# FIX 4: errors are flagged with "error": True in the output dict.
async def run_inference(example, args):
    messages = list(example["messages"])  # local copy — safe to reassign
    sanitized = False

    for attempt in range(args.max_retries):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                ),
                timeout=args.timeout,
            )
            return response.choices[0].message.content.strip(), False  # (pred, is_error)

        except (RateLimitError, APIError):
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.warning("Rate/API error on attempt %d for index %s, retrying in %.1fs",
                           attempt + 1, example["index"], wait)
            await asyncio.sleep(wait)

        except BadRequestError as e:
            if sanitized:
                # Already tried sanitizing — this request is unrecoverable.
                logger.error("BadRequestError persists after sanitization for index %s: %s",
                             example["index"], e)
                return "Bad request — unrecoverable", True
            logger.warning("BadRequestError for index %s, sanitizing and retrying: %s",
                           example["index"], e)
            messages = sanitize_messages(messages)  # reassign so next attempt uses clean messages
            sanitized = True
            # Don't sleep — retry immediately with the sanitized messages.

        except asyncio.TimeoutError:
            logger.warning("Timeout on attempt %d for index %s", attempt + 1, example["index"])
            await asyncio.sleep(2 ** attempt)

        except Exception as e:
            logger.warning("Unexpected error on attempt %d for index %s: %s",
                           attempt + 1, example["index"], e)
            await asyncio.sleep(2 ** attempt)

    return "Failed after max retries", True


# FIX 5: asyncio.Lock passed in and used to serialise file writes.
async def worker(example, args, sem, outfile, write_lock):
    async with sem:
        pred, is_error = await run_inference(example, args)

        output = {
            "index": example["index"],
            "mention": example["mention"],
            "prediction": pred,
            "correct_answer": example.get("correct_answer"),
            "candidates": example.get("candidates", []),
            "error": is_error,
        }

        if is_error:
            logger.error("index %s failed: %s", example["index"], pred)
        else:
            logger.debug("index %s: %s -> %s", example["index"], example["mention"], pred)

        async with write_lock:
            outfile.write(json.dumps(output, ensure_ascii=False) + "\n")
            outfile.flush()


async def main():
    args = parse_args()

    processed_ids = set()
    try:
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                processed_ids.add(json.loads(line)["index"])
        logger.info("Resuming: %d already processed", len(processed_ids))
    except FileNotFoundError:
        pass

    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    data = [ex for ex in data if ex["index"] not in processed_ids]
    logger.info("Remaining examples: %d", len(data))

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    with open(args.output, "a", encoding="utf-8") as outfile:
        tasks = [
            asyncio.create_task(worker(example, args, sem, outfile, write_lock))
            for example in data
        ]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await f


if __name__ == "__main__":
    asyncio.run(main())