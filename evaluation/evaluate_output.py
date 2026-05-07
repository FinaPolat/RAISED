import argparse
import os
import re
from tqdm import tqdm
import json


def read_jsonL(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def extract_answer(text: str) -> str:
    """
    Extracts the final answer from a string that may contain <think>...</think> reasoning.
    Returns the content after the closing </think> tag, or the full text if no think tag exists.
    """
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_answer(answer):
    """Normalize an answer string without over-stripping."""
    if '/think>' in answer:
        answer = extract_answer(answer)
    else:
        answer = answer.strip()

    return answer.replace('\n', '').replace('"', '').replace("'", "").replace("*", "").strip()


def merge_llm_answers(prompts_file, llm_answers_file):
    """
    Join the prompts file (has correct_answer + candidates) with the predictions
    file (has prediction) on the shared `index` field. This is robust to the
    async client producing predictions in a different order than the prompts.
    """
    prompts = read_jsonL(prompts_file)
    llm_answers = read_jsonL(llm_answers_file)

    print(f"Number of prompt items:     {len(prompts)}")
    print(f"Number of LLM answer items: {len(llm_answers)}")

    # Build an index -> prediction map (last write wins if duplicates exist)
    pred_by_idx = {}
    dup_count = 0
    for item in llm_answers:
        idx = item["index"]
        if idx in pred_by_idx:
            dup_count += 1
        pred_by_idx[idx] = item["prediction"]
    if dup_count:
        print(f"[Warn] {dup_count} duplicate predictions found; kept the last one for each index.")

    merged_data = []
    none_count = 0
    missing_count = 0

    for item in tqdm(prompts, desc="Merging LLM answers", total=len(prompts)):
        idx = item["index"]

        if idx not in pred_by_idx:
            # No prediction written for this prompt — treat as empty -> "None"
            item["prediction"] = "None"
            missing_count += 1
            none_count += 1
            merged_data.append(item)
            continue

        raw_answer = pred_by_idx[idx]
        answer = normalize_answer(raw_answer)

        if not answer:
            item["prediction"] = "None"
            none_count += 1
        else:
            item["prediction"] = answer

        merged_data.append(item)

    if missing_count:
        print(f"[Warn] {missing_count} prompts had no matching prediction (treated as 'None').")
    print(f"Number of empty LLM answers: {none_count}")
    return merged_data


def evaluate_performance(dataset):
    all_results = {}

    # In-KG Counters
    inkg_tp = 0
    inkg_fp = 0
    inkg_fn = 0
    inkg_actual = 0

    # NOC Counters
    noc_tp = 0
    noc_fn = 0
    noc_fp = 0
    noc_actual = 0

    for item in dataset:
        correct_answer = str(item["correct_answer"])
        candidates = item["candidates"]
        llm_answer = str(item["prediction"]).lower()

        candidates_str = [str(c) for c in candidates]
        is_gt_noc = correct_answer not in candidates_str
        correct_answer_str = "none" if is_gt_noc else correct_answer.lower()

        # In-KG Metrics
        if not is_gt_noc:
            inkg_actual += 1
            if "none" in llm_answer:
                inkg_fn += 1
            elif llm_answer == correct_answer_str:
                inkg_tp += 1
            else:
                inkg_fp += 1

        # NOC Metrics
        if is_gt_noc:
            noc_actual += 1
            if "none" in llm_answer:
                noc_tp += 1
            else:
                noc_fn += 1
        else:
            if "none" in llm_answer:
                noc_fp += 1

    # In-KG F1
    inkg_prec = inkg_tp / (inkg_tp + inkg_fp) if (inkg_tp + inkg_fp) > 0 else 0.0
    inkg_rec = inkg_tp / (inkg_tp + inkg_fn) if (inkg_tp + inkg_fn) > 0 else 0.0
    inkg_f1 = 2 * (inkg_prec * inkg_rec) / (inkg_prec + inkg_rec) if (inkg_prec + inkg_rec) > 0 else 0.0

    # NOC F1
    pred_nocs = noc_tp + noc_fp
    noc_prec = noc_tp / pred_nocs if pred_nocs > 0 else 0.0
    actual_nocs = noc_tp + noc_fn
    noc_rec = noc_tp / actual_nocs if actual_nocs > 0 else 0.0
    noc_f1 = 2 * (noc_prec * noc_rec) / (noc_prec + noc_rec) if (noc_prec + noc_rec) > 0 else 0.0

    # Overall EL F1
    precision = (inkg_tp) / (inkg_tp + inkg_fp + noc_fn) if (inkg_tp + inkg_fp + noc_fn) > 0 else 0.0
    recall = (inkg_tp) / (inkg_tp + inkg_fn + noc_actual) if (inkg_tp + inkg_fn + noc_actual) > 0 else 0.0
    overall_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    all_results = {
        "In-KG F1": round(inkg_f1, 3),
        "In-KG Precision": round(inkg_prec, 3),
        "In-KG Recall": round(inkg_rec, 3),
        "In-KG TP": inkg_tp,
        "In-KG FP": inkg_fp,
        "In-KG FN": inkg_fn,
        "In-KG Actual Count": inkg_actual,
        "NOC F1": round(noc_f1, 3),
        "NOC Precision": round(noc_prec, 3),
        "NOC Recall": round(noc_rec, 3),
        "NOC TP": noc_tp,
        "NOC FP": noc_fp,
        "NOC FN": noc_fn,
        "NOC Actual Count": noc_actual,
        "total_samples": len(dataset),
        "Overall Precision": round(precision, 3),
        "Overall Recall": round(recall, 3),
        "Overall F1": round(overall_f1, 3),
    }

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Path to the original prompts file (has index, correct_answer, candidates, messages)")
    parser.add_argument("--LLM_answers", type=str, required=True,
                        help="Path to the LLM predictions file (has index, prediction)")
    parser.add_argument("--output_folder", type=str, default="candidate_generation/bm25/scores",
                        help="Path to save the evaluation results")
    parser.add_argument("--experiment", type=str, default="bm25_gpt_wned-wiki_results.json",
                        help="Name of the experiment for output file naming")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    merged_data = merge_llm_answers(args.prompts_file, args.LLM_answers)
    evaluation_results = evaluate_performance(merged_data)
    output_file = os.path.join(args.output_folder, args.experiment)
    write_json(evaluation_results, output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
