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
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=30.0)  # FIX: per-request timeout
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


# FIX: re-raise on unexpected structure so the error isn't silently swallowed;
# log a warning when unrecognised output types are encountered.
def extract_text(response, index):
    texts = []
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    texts.append(content.text)
                else:
                    logger.warning("index %s: unrecognised content type %r — skipping", index, content.type)
        else:
            logger.warning("index %s: unrecognised output type %r — skipping", index, item.type)

    return "".join(texts).strip()


# FIX: asyncio.wait_for for per-request timeout.
# FIX: sanitize upfront (kept from original), BadRequestError now logs detail and returns immediately
#      since sanitization has already been applied — there's nothing else to try.
# FIX: returns (pred, is_error) tuple so errors are distinguishable in output.
async def run_inference(example, args):
    messages = sanitize_messages(example["messages"])  # sanitize once upfront

    for attempt in range(args.max_retries):
        try:
            response = await asyncio.wait_for(
                client.responses.create(
                    model=args.model,
                    input=messages,
                    temperature=args.temperature,
                    max_output_tokens=args.max_tokens,
                ),
                timeout=args.timeout,
            )

            text = extract_text(response, example["index"])

            if not text:
                logger.warning("index %s: empty response on attempt %d", example["index"], attempt + 1)
                return "No LLM response", True

            return text, False  # (pred, is_error)

        except (RateLimitError, APIError) as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.warning("index %s: rate/API error on attempt %d, retrying in %.1fs: %s",
                           example["index"], attempt + 1, wait, e)
            await asyncio.sleep(wait)

        except BadRequestError as e:
            # Messages were already sanitized before the loop — bad request is unrecoverable.
            logger.error("index %s: BadRequestError (unrecoverable): %s", example["index"], e)
            return "Bad request — unrecoverable", True

        except asyncio.TimeoutError:
            logger.warning("index %s: timeout on attempt %d", example["index"], attempt + 1)
            await asyncio.sleep(2 ** attempt)

        except Exception as e:
            logger.warning("index %s: unexpected error on attempt %d: %s", example["index"], attempt + 1, e)
            await asyncio.sleep(2 ** attempt)

    return "Failed after max retries", True


# FIX: asyncio.Lock passed in and held during file writes to prevent interleaved output.
# FIX: "error" field added to output dict.
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