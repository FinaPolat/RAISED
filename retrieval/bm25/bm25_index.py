import json
import bm25s
import Stemmer

CORPUS_PATH = "candidate_generation/zelda/entity_descriptions.jsonl"
INDEX_PATH  = "candidate_generation/bm25/wiki_bm25_index"
IDS_PATH    = "candidate_generation/bm25/wiki_bm25_ids.json"

# ── Load JSONL ───────────────────────────────────────────────────────────────
corpus = []
with open(CORPUS_PATH) as f:
    for line in f:
        corpus.append(json.loads(line))

ids   = [entry["wikipedia_id"] for entry in corpus]
texts = [
    " ".join(filter(None, [
        entry.get("wikipedia_title", ""),
        entry.get("description", ""),
    ]))
    for entry in corpus
]

# ── Index ────────────────────────────────────────────────────────────────────
stemmer   = Stemmer.Stemmer("english")
tokenized = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)

retriever = bm25s.BM25()
retriever.index(tokenized)

# ── Save ─────────────────────────────────────────────────────────────────────
retriever.save(INDEX_PATH)
with open(IDS_PATH, "w") as f:
    json.dump(ids, f)

print(f"Indexed {len(ids):,} entries.")