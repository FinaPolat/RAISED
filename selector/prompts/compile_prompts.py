import os
import re
import json
import argparse
import difflib
from html import unescape
from tqdm import tqdm


def read_jsonL(file):
    data = []
    with open(file, "r", encoding='utf-8') as read_file:
        for line in read_file:
            data.append(json.loads(line))
    return data


def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_jsonL(data, file):
    with open(file, "w", encoding='utf-8') as write_file:
        for line in data:
            json.dump(line, write_file, ensure_ascii=False)
            write_file.write("\n")


def write_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def format_candidates(reranking_output):
    lines = []
    for candidate_dict in reranking_output:
        for label, info in candidate_dict.items():
            candidate_id = info.get("id", "N/A")
            description = info.get("description", "No description available")
            readable_label = label.replace("_", " ")
            lines.append(f"{candidate_id} -- {readable_label}: {description}")
    return "\n".join(lines)


def get_ground_truth_id(ground_truth):
    for label, info in ground_truth.items():
        return info.get("id", None)
    return None


def find_context_file(context_folder, dataset_name):
    for fname in os.listdir(context_folder):
        if dataset_name in fname and fname.endswith(".jsonl"):
            return os.path.join(context_folder, fname)
    return None


def normalize_mention(mention):
    mention = unescape(mention)
    mention = mention.lower()
    mention = re.sub(r"[^a-z0-9]", "", mention)
    return mention


def build_context_lookup(context_data):
    lookup = {}
    dataset_to_mentions = {}

    for item in context_data:
        dataset = item.get("dataset", "")
        mention_raw = item.get("mention", "")
        mention = normalize_mention(mention_raw)

        key = (dataset, mention)

        lookup.setdefault(key, []).append(item.get("context", ""))
        dataset_to_mentions.setdefault(dataset, set()).add(mention)

    # convert sets to lists for fuzzy matching
    dataset_to_mentions = {
        k: list(v) for k, v in dataset_to_mentions.items()
    }

    return lookup, dataset_to_mentions


def fuzzy_match_mention(target, candidates, threshold=0.85):
    matches = difflib.get_close_matches(target, candidates, n=1, cutoff=threshold)
    return matches[0] if matches else None


#################################################################################################

def process_file(input_file, template, system_prompt, output_folder, context_folder=None):
    input_data = read_json(input_file)
    input_file_name = os.path.splitext(os.path.basename(input_file))[0]
    print(f"\n[{input_file_name}] Loaded {len(input_data)} items", flush=True)

    context_lookup = {}
    dataset_to_mentions = {}

    if context_folder:
        dataset_name = input_file_name.replace("_entity_results", "")
        context_file = find_context_file(context_folder, dataset_name)

        if context_file:
            context_data = read_jsonL(context_file)
            context_lookup, dataset_to_mentions = build_context_lookup(context_data)
            print(f"  Context file : {context_file} ({len(context_data)} items)", flush=True)
        else:
            print(f"  WARNING: No context JSONL found for dataset '{dataset_name}'", flush=True)

    prompts = []
    data = []
    missing_context_items = []

    consume_counter = {}

    stats = {
        "exact": 0,
        "fuzzy": 0,
        "missing": 0
    }

    for idx, item in tqdm(enumerate(input_data), desc=f"  Compiling {input_file_name}", total=len(input_data)):
        mention = item["mention"]
        ground_truth = item.get("ground_truth", {})
        reranking_output = item.get("reranking_output", [])

        dataset = item.get("dataset", f"test_{input_file_name.replace('_entity_results', '')}")

        normalized = normalize_mention(mention)
        key = (dataset, normalized)

        match_type = "missing"

        if context_lookup and key in context_lookup:
            occurrence = consume_counter.get(key, 0)
            context = context_lookup[key][min(occurrence, len(context_lookup[key]) - 1)]
            consume_counter[key] = occurrence + 1
            match_type = "exact"

        elif context_lookup:
            candidates = dataset_to_mentions.get(dataset, [])
            fuzzy_hit = fuzzy_match_mention(normalized, candidates)

            if fuzzy_hit:
                fuzzy_key = (dataset, fuzzy_hit)
                occurrence = consume_counter.get(fuzzy_key, 0)
                context = context_lookup[fuzzy_key][min(occurrence, len(context_lookup[fuzzy_key]) - 1)]
                consume_counter[fuzzy_key] = occurrence + 1
                match_type = "fuzzy"
            else:
                missing_context_items.append({
                    "index": idx,
                    "dataset": dataset,
                    "mention": mention
                })
                context = item.get("context", "")
                match_type = "missing"
        else:
            context = item.get("context", "")

        stats[match_type] += 1

        correct_answer = get_ground_truth_id(ground_truth)

        candidate_ids = [
            info.get("id")
            for candidate_dict in reranking_output
            for info in candidate_dict.values()
            if isinstance(info, dict)
        ]

        if correct_answer not in candidate_ids:
            correct_answer = "None of the candidates"

        candidates_str = (
            format_candidates(reranking_output)
            if reranking_output
            else f"No candidates have been found for {mention}"
        )

        prompt = template["formatter"].format(
            mention=mention,
            context=context,
            candidates=candidates_str
        )

        data.append({
            "index": idx,
            "candidates": candidate_ids,
            "correct_answer": correct_answer,
            "mention": mention,
            "context": context,
            "match_type": match_type,
            "candidates": candidates_str,
            "prompt": prompt
        })

        prompts.append({
            "index": idx,
            "mention": mention,
            "correct_answer": correct_answer,
            "candidates": candidate_ids,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        })

    # logging
    print("\n  Match Statistics:")
    print(f"    Exact   : {stats['exact']}")
    print(f"    Fuzzy   : {stats['fuzzy']}")
    print(f"    Missing : {stats['missing']}")
    print(f"    Coverage: {(stats['exact'] + stats['fuzzy']) / len(input_data):.2%}")

    if missing_context_items:
        missing_file = os.path.join(output_folder, f"missing_context_{input_file_name}.json")
        write_json(missing_context_items, missing_file)
        print(f"  Missing context log: {missing_file}")

    test_file = os.path.join(output_folder, f"{input_file_name}_prompts.jsonl")
    write_jsonL(prompts, test_file)
    print(f"  Prompts written to: {test_file}")

    inspect_file = os.path.join(output_folder, f"data_to_inspect_{input_file_name}.json")
    write_json(data[:5], inspect_file)
    print(f"  Inspection sample: {inspect_file}")

    return len(input_data), len(prompts)


def main():
    parser = argparse.ArgumentParser(description='Compile test prompts from entity results files')
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--context_folder', type=str, default=None)
    parser.add_argument('--prompt_template', type=str,
                        default="VerbalizED/prompting/ED_template_can_reject.json")
    parser.add_argument('--output_folder', type=str,
                        default="VerbalizED/prompting/compiled_prompts")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    system_prompt = (
        "You are an expert entity disambiguation agent. "
        "Follow the instructions carefully and do exactly as the user asks you to do. "
        "Do not provide any additional information or explanations. "
        "Only provide the output as requested by the user."
    )

    template = read_json(args.prompt_template)

    input_files = sorted([
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if f.endswith(".json")
    ])

    if not input_files:
        print(f"No JSON files found in {args.input_folder}")
        return

    total_items, total_prompts = 0, 0

    for input_file in input_files:
        n_items, n_prompts = process_file(
            input_file, template, system_prompt, args.output_folder,
            context_folder=args.context_folder
        )
        total_items += n_items
        total_prompts += n_prompts

    print("\n=== Overall Summary ===")
    print(f"  Files processed : {len(input_files)}")
    print(f"  Total items     : {total_items}")
    print(f"  Total prompts   : {total_prompts}")


if __name__ == "__main__":
    main()