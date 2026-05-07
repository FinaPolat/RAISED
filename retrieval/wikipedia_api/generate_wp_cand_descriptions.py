import os
import argparse
from tqdm import tqdm
import json
from retry import retry
import requests


def read_jsonL(file):
    data = []
    with open(file, "r", encoding='utf-8') as read_file:
        lines = read_file.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data

def write_jsonL(data, file):
    with open(file, "w", encoding='utf-8') as write_file:
        for line in data:
            json.dump(line, write_file, ensure_ascii=False)
            write_file.write("\n")

def write_json(data, file):
    with open(file, "w", encoding='utf-8') as write_file:
        json.dump(data, write_file, ensure_ascii=False, indent=4)


@retry(tries=3, delay=5, max_delay=60)
def fetch_wikipedia_page_data(page_ids, language="en"):
    """
    Fetch Wikipedia title, summary, and disambiguation status for up to 50 page IDs.
    Returns all keys as **strings**.

    Args:
        page_ids (list[int or str]): Wikipedia page IDs.
        language (str): Language code.

    Returns:
        dict: {
            "page_id": {
                "title": str,
                "summary": str,
                "is_disambiguation": bool
            } or None if missing
        }
    """
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "WikipediaCandidateFetcher/1.0 (contact: your_email@example.com)"
    }

    results = {}
    BATCH = 50

    for i in range(0, len(page_ids), BATCH):
        batch = page_ids[i:i + BATCH]
        batch_str = [str(pid) for pid in batch]

        params = {
            "action": "query",
            "pageids": "|".join(batch_str),
            "prop": "extracts|pageprops",
            "exintro": True,
            "explaintext": True,
            "format": "json"
        }

        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})

            for pid in batch_str:   # keys remain strings
                page = pages.get(pid)

                if (not page) or ("missing" in page):
                    results[pid] = None
                    continue

                results[pid] = {
                    "wikipedia_title": page.get("title", ""),
                    "wikipedia_summary": page.get("extract", ""),
                    "disambiguation": "disambiguation" in page.get("pageprops", {})
                }

        except Exception as e:
            print(f"Batch error: {e}")
            for pid in batch_str:
                results[pid] = None

    return results

def main():
    parser = argparse.ArgumentParser(description="Generate candidate descriptions.")
    parser.add_argument("--cand_file", type=str, default= "finetuning_LLMs_for_ED/sampled_data/first_is_disambig.jsonl", help="Path to the candidate list file.")
    parser.add_argument("--output_dir", type=str, default= "finetuning_LLMs_for_ED/sampled_data/cand_desc", help="Directory to save the output files.")
    parser.add_argument("--dataset_name", type=str, default="first_is_disambig", help="Name of the dataset")
    args = parser.parse_args()

    args = parser.parse_args()
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Output folder {args.output_dir} created.")

    # Read the candidate list
    data = read_jsonL(args.cand_file)

    unique_candidates = dict()
    for i in tqdm(data, desc="Processing entries", total=len(data)):
        cands = i.get("candidates", [])
        #print(f"Processing entry with {len(cands)} candidates.")
        #print(cands)
        if not cands:
            print(f"Warning: No candidates found for entry {i}.")
        else:
            descriptions = fetch_wikipedia_page_data(cands)
            #print(f"Fetched descriptions for entry {i}.")
            #print(descriptions)
            unique_candidates.update(descriptions)
        #unique_candidates.update(i["candidates"])
    print(f"Unique candidates found: {len(unique_candidates)}")
    #print("Processing candidates...")

    # Save the candidate descriptions
    output_file = os.path.join(args.output_dir, f"cand_descriptions_wp_{args.dataset_name}.json")
    write_json(unique_candidates, output_file)
    print(f"Candidate descriptions saved to {output_file}")

if __name__ == "__main__":
    main()