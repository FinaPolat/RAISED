import json
import bm25s
import Stemmer
from pathlib import Path

CORPUS_PATH = Path("candidate_generation/zelda/entity_descriptions.jsonl")
INDEX_PATH  = Path("candidate_generation/bm25/wiki_bm25_index")
IDS_PATH    = Path("candidate_generation/bm25/wiki_bm25_ids.json")
QUERIES_FOLDER = Path("candidate_generation/test_data/")
OUTPUT_FOLDER  = Path("candidate_generation/bm25/retrieved/")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

query_files = sorted(QUERIES_FOLDER.glob("*.jsonl"))
print(f"Found {len(query_files)} query files.")

# ── Load ──────────────────────────────────────────────────────────────────────
with open(IDS_PATH) as f:
    ids = json.load(f)

id_to_idx = {wid: i for i, wid in enumerate(ids)}

retriever = bm25s.BM25.load(INDEX_PATH)
stemmer   = Stemmer.Stemmer("english")

# ── Run ───────────────────────────────────────────────────────────────────────
for query_file in query_files:
    output_file = OUTPUT_FOLDER / query_file.name
    print(f"Processing {query_file.name} ...", end=" ", flush=True)
    with open(query_file) as fin, open(output_file, "w") as fout:
        for line in fin:
            q = json.loads(line)

            query  = f"{q['mention']} {q['mention']} {q['context'].replace('#', '')}"
            tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer)
            scores = retriever.get_scores(tokens.ids[0])  # <-- fix here

            candidate_indices = [id_to_idx[cid] for cid in q["candidates"] if cid in id_to_idx]
            ranked_ids = sorted(candidate_indices, key=lambda i: scores[i], reverse=True)
            retrieved  = [ids[i] for i in ranked_ids][:16]

            fout.write(json.dumps({
                "mention":        q["mention"],
                "correct_answer": q["correct_answer"],
                "retrieved":      retrieved,
            }) + "\n")

print("Done.")