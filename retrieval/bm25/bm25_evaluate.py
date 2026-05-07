import json
from pathlib import Path

RETRIEVED_FOLDER = Path("candidate_generation/bm25/retrieved/")
RESULTS_PATH     = Path("candidate_generation/bm25/evaluation_results.jsonl")

retrieved_files = sorted(RETRIEVED_FOLDER.glob("*.jsonl"))
print(f"Found {len(retrieved_files)} retrieved files.")

all_results = []
dataset_stats = {}

for retrieved_file in retrieved_files:
    dataset = retrieved_file.stem  # filename without .jsonl
    results = []

    with open(retrieved_file) as f:
        for line in f:
            r = json.loads(line)
            correct   = r["correct_answer"]
            retrieved = r["retrieved"]
            results.append({
                "mention": r["mention"],
                "correct": correct,
                "hit@1":   correct in retrieved[:1],
                "hit@8":   correct in retrieved[:8],
                "hit@16":  correct in retrieved[:16],
            })

    n = len(results)
    stats = {
        "dataset":  dataset,
        "n":        n,
        "recall@1":  sum(r["hit@1"]  for r in results) / n,
        "recall@8":  sum(r["hit@8"]  for r in results) / n,
        "recall@16": sum(r["hit@16"] for r in results) / n,
    }
    dataset_stats[dataset] = stats
    all_results.extend(results)

    print(f"  [{dataset}]  n={n:6d}  "
          f"R@1={stats['recall@1']:.1%}  "
          f"R@8={stats['recall@8']:.1%}  "
          f"R@16={stats['recall@16']:.1%}")

# ── Overall ───────────────────────────────────────────────────────────────────
total = len(all_results)
overall = {
    "dataset":   "overall",
    "n":         total,
    "recall@1":  sum(r["hit@1"]  for r in all_results) / total,
    "recall@8":  sum(r["hit@8"]  for r in all_results) / total,
    "recall@16": sum(r["hit@16"] for r in all_results) / total,
}
print(f"\n  [overall]    n={total:6d}  "
      f"R@1={overall['recall@1']:.1%}  "
      f"R@8={overall['recall@8']:.1%}  "
      f"R@16={overall['recall@16']:.1%}")

# ── Write results ─────────────────────────────────────────────────────────────
with open(RESULTS_PATH, "w") as f:
    for stats in dataset_stats.values():
        f.write(json.dumps(stats) + "\n")
    f.write(json.dumps(overall) + "\n")

print(f"\nResults written to {RESULTS_PATH}")