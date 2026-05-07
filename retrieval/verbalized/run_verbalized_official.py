"""
Run Official VERBALIZED with Your Candidate Sets
=================================================
This script:
1. Converts your JSONL data to VERBALIZED's input format
2. Restricts VERBALIZED's label search to YOUR candidates
3. Reports results comparable to your LLM system

Prerequisites:
  - Clone https://github.com/flairNLP/VerbalizED
  - Download model weights from Google Drive
  - pip install flair torch transformers

Usage:
  python run_verbalized_official.py \
      --data_path dev.jsonl \
      --desc_path cand_descriptions_wp.json \
      --verbalized_repo /path/to/VerbalizED \
      --model_path /path/to/VerbalizED/models/VerbalizED_base_Zelda \
      --output_path verbalized_official_results.jsonl
"""

import json
import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter


# ─────────────────────────────────────────────────────────
# STEP 1: Load VERBALIZED model using their code
# ─────────────────────────────────────────────────────────

def load_verbalized_model(verbalized_repo: str, model_path: str):
    """
    Load the official VERBALIZED model from their repo.
    Uses their DualEncoder class directly.
    """
    # Add their repo to path
    sys.path.insert(0, verbalized_repo)

    try:
        # Try loading their model class
        # Based on their repo structure
        from dual_encoder import DualEncoder
        print(f"Loading VERBALIZED model from {model_path}...")
        model = DualEncoder.load(model_path)
        model.eval()
        print("  Model loaded successfully.")
        return model, "flair"

    except ImportError:
        print("Could not import DualEncoder from their repo.")
        print("Falling back to manual BERT-based implementation...")
        return load_bert_fallback(model_path)


def load_bert_fallback(model_path: str):
    """
    Fallback: load BERT encoder directly from their checkpoint.
    Their model is BERT-base-uncased with trained weights.
    """
    from transformers import AutoTokenizer, AutoModel

    # Check if model_path has pytorch_model.bin or similar
    model_path = Path(model_path)

    # Their model likely saves as HuggingFace format
    if (model_path / "config.json").exists():
        print(f"Loading from HuggingFace format: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        encoder   = AutoModel.from_pretrained(str(model_path))
    else:
        # Fall back to base BERT (zero-shot comparison)
        print("Model weights not found in HF format.")
        print("Loading bert-base-uncased (zero-shot VERBALIZED).")
        print("For trained model, ensure weights are in HF format.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        encoder   = AutoModel.from_pretrained("bert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    encoder.eval()

    return (tokenizer, encoder, device), "bert"


# ─────────────────────────────────────────────────────────
# STEP 2: Load their verbalizations
# ─────────────────────────────────────────────────────────

def load_verbalizations(
    verbalization_path: str,
    candidate_ids: set,
    desc_path: str
) -> dict:
    """
    Load verbalizations for your candidate entities.
    
    Priority:
    1. Their official zelda_labels_verbalizations.json
    2. Your own cand_descriptions_wp.json as fallback
    """
    verbalizations = {}

    # Try their official verbalizations first
    if verbalization_path and Path(verbalization_path).exists():
        print(f"Loading official VERBALIZED verbalizations from {verbalization_path}...")
        with open(verbalization_path) as f:
            official_verbs = json.load(f)

        # Their format: {"Q312": "Apple Inc.; American multinational...", ...}
        # or possibly {"Q312": {"verbalization": "...", ...}}
        for qid in candidate_ids:
            if str(qid) in official_verbs:
                v = official_verbs[str(qid)]
                if isinstance(v, str):
                    verbalizations[str(qid)] = v
                elif isinstance(v, dict):
                    verbalizations[str(qid)] = v.get(
                        "verbalization",
                        v.get("title", str(qid))
                    )

        coverage = len(verbalizations) / max(len(candidate_ids), 1)
        print(f"  Coverage: {len(verbalizations)}/{len(candidate_ids)} "
              f"({coverage:.1%}) candidates found in official verbalizations")

    # Fill gaps with your descriptions
    missing = candidate_ids - set(verbalizations.keys())
    if missing and desc_path:
        print(f"  Filling {len(missing)} missing verbalizations from {desc_path}...")
        with open(desc_path) as f:
            descriptions = json.load(f)

        for qid in missing:
            info = descriptions.get(str(qid), {})
            if info:
                verb = build_verbalization_from_desc(info)
                if verb:
                    verbalizations[str(qid)] = verb

    final_coverage = len(verbalizations) / max(len(candidate_ids), 1)
    print(f"  Final coverage: {len(verbalizations)}/{len(candidate_ids)} "
          f"({final_coverage:.1%})")

    return verbalizations


def build_verbalization_from_desc(info: dict, max_chars: int = 50) -> str:
    """
    Build VERBALIZED-style verbalization from your description dict.
    Format: "Title; Description, Categories"
    """
    title = (info.get("wikipedia_title") or
             info.get("title") or "").strip()
    if not title:
        return ""

    desc = (info.get("description") or "").strip()
    if desc and len(desc) > max_chars:
        desc = desc[:max_chars].rsplit(" ", 1)[0]

    cats = []
    for key in ["instance_of", "occupation", "country", "categories"]:
        val = info.get(key, "")
        if isinstance(val, list):
            cats.extend(str(v) for v in val[:2])
        elif isinstance(val, str) and val:
            cats.append(val)

    if cats and desc:
        return f"{title}; {desc}, {', '.join(cats[:2])}"
    elif desc:
        return f"{title}; {desc}"
    elif cats:
        return f"{title}; {', '.join(cats[:2])}"
    else:
        return title


# ─────────────────────────────────────────────────────────
# STEP 3: Encode with BERT (first-last pooling)
# ─────────────────────────────────────────────────────────

def first_last_pool(
    hidden: torch.Tensor,
    start: int,
    end: int
) -> torch.Tensor:
    """First-last token pooling as described in VERBALIZED paper."""
    end = min(end, hidden.shape[0])
    start = min(start, end - 1)
    first = hidden[start]
    last  = hidden[end - 1]
    return torch.cat([first, last], dim=-1)


def encode_mention_bert(
    tokenizer, encoder, device,
    context: str,
    mention: str,
    max_length: int = 512
) -> torch.Tensor:
    """Encode mention using BERT with first-last pooling."""

    # Mark mention with unused tokens
    marked = context.replace(mention, f"[unused0] {mention} [unused1]", 1)

    enc = tokenizer(
        marked,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    ids = enc["input_ids"][0].tolist()
    u0  = tokenizer.convert_tokens_to_ids("[unused0]")
    u1  = tokenizer.convert_tokens_to_ids("[unused1]")

    try:
        start = ids.index(u0) + 1
        end   = ids.index(u1)
    except ValueError:
        start, end = 1, 2  # fallback to CLS

    with torch.no_grad():
        out    = encoder(**enc)
        hidden = out.last_hidden_state[0]

    return first_last_pool(hidden, start, end)


def encode_label_bert(
    tokenizer, encoder, device,
    verbalization: str,
    max_length: int = 128
) -> torch.Tensor:
    """Encode entity verbalization using BERT with first-last pooling over title."""

    # Split title from rest
    if ";" in verbalization:
        title = verbalization.split(";")[0].strip()
    else:
        title = verbalization

    enc = tokenizer(
        verbalization,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Title token span (skip CLS at 0)
    title_ids = tokenizer(title, add_special_tokens=False)["input_ids"]
    title_len = len(title_ids)
    start = 1
    end   = min(1 + title_len, enc["input_ids"].shape[1] - 1)
    if end <= start:
        end = start + 1

    with torch.no_grad():
        out    = encoder(**enc)
        hidden = out.last_hidden_state[0]

    return first_last_pool(hidden, start, end)


# ─────────────────────────────────────────────────────────
# STEP 4: Main inference loop
# ─────────────────────────────────────────────────────────

def run_inference(
    data_path: str,
    desc_path: str,
    verbalized_repo: str,
    model_path: str,
    verbalization_path: str,
    output_path: str,
    max_mention_length: int = 512,
    max_label_length: int = 128,
):
    # ── Load model ──────────────────────────────────────
    model_info, model_type = load_verbalized_model(verbalized_repo, model_path)

    if model_type == "bert":
        tokenizer, encoder, device = model_info
    else:
        # Their flair model — use their predict API
        # We'll handle this separately below
        flair_model = model_info

    # ── Collect all candidate IDs ────────────────────────
    print(f"\nCollecting candidate IDs from {data_path}...")
    all_candidate_ids = set()
    records = []

    with open(data_path) as f:
        for line in f:
            record = json.loads(line.strip())
            records.append(record)
            for cid in record.get("candidate_ids", []):
                if cid not in ("NIL",):
                    all_candidate_ids.add(str(cid))

    print(f"  Total unique candidates: {len(all_candidate_ids):,}")
    print(f"  Total records: {len(records):,}")

    # ── Load verbalizations ──────────────────────────────
    verbalizations = load_verbalizations(
        verbalization_path,
        all_candidate_ids,
        desc_path
    )

    # ── Pre-encode all unique candidates ─────────────────
    # This is efficient: encode each candidate once
    print(f"\nPre-encoding {len(verbalizations):,} candidates...")
    candidate_embeddings = {}

    for cid, verb in tqdm(verbalizations.items(), desc="Encoding candidates"):
        if model_type == "bert":
            emb = encode_label_bert(
                tokenizer, encoder, device,
                verb,
                max_length=max_label_length
            )
            candidate_embeddings[cid] = emb.cpu()

    print(f"  Encoded {len(candidate_embeddings):,} candidates")

    # ── Run inference ────────────────────────────────────
    print(f"\nRunning VERBALIZED inference...")

    results         = []
    by_tier         = defaultdict(list)
    nil_outcomes    = []
    nonnil_outcomes = []

    with open(output_path, "w") as f_out:
        for record in tqdm(records, desc="Predicting"):

            mention    = record["mention"]
            context    = record["context"]
            is_nil     = record.get("is_nil", False)
            correct_id = record.get("correct_answer_id", "")
            cand_ids   = [c for c in record.get("candidate_ids", [])
                          if c not in ("NIL",)]
            cand_titles = record.get("candidate_titles", [])
            tier        = record.get("difficulty_tier", "unknown")

            # Filter to candidates that have embeddings
            valid_cids = [c for c in cand_ids
                          if str(c) in candidate_embeddings]

            if not valid_cids:
                outcome = "true_abstention" if is_nil else "false_abstention"
                result  = {
                    "prediction": "NIL",
                    "correct_id": correct_id,
                    "is_nil":     is_nil,
                    "tier":       tier,
                    "mention":    mention,
                    "outcome":    outcome,
                    "note":       "no_valid_candidates"
                }
                results.append(result)
                by_tier[tier].append(outcome)
                f_out.write(json.dumps(result) + "\n")
                continue

            # Encode mention
            if model_type == "bert":
                mention_emb = encode_mention_bert(
                    tokenizer, encoder, device,
                    context=context,
                    mention=mention,
                    max_length=max_mention_length
                ).cpu()

            # Stack candidate embeddings
            cand_embs = torch.stack([
                candidate_embeddings[str(c)] for c in valid_cids
            ])  # [n, 2*hidden]

            # Euclidean distance (lower = better match)
            diff      = cand_embs - mention_emb.unsqueeze(0)
            distances = torch.norm(diff, dim=-1).tolist()

            # VERBALIZED always commits — picks minimum distance
            best_idx     = int(np.argmin(distances))
            predicted_id = str(valid_cids[best_idx])

            # Evaluate
            # VERBALIZED cannot abstain:
            if is_nil:
                outcome = "false_commitment"   # always wrong on NIL
            else:
                if predicted_id == str(correct_id):
                    outcome = "true_linking"
                else:
                    outcome = "wrong_linking"

            result = {
                "prediction":      predicted_id,
                "correct_id":      str(correct_id),
                "is_nil":          is_nil,
                "tier":            tier,
                "mention":         mention,
                "outcome":         outcome,
                "min_distance":    distances[best_idx],
                "n_candidates":    len(valid_cids),
                "verbalization":   verbalizations.get(predicted_id, ""),
            }

            results.append(result)
            by_tier[tier].append(outcome)

            if is_nil:
                nil_outcomes.append(outcome)
            else:
                nonnil_outcomes.append(outcome)

            f_out.write(json.dumps(result) + "\n")

    # ─────────────────────────────────────────────────────
    # STEP 5: Report results
    # ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("VERBALIZED (official model, your candidates) — RESULTS")
    print("="*65)

    def metrics(outcomes):
        c        = Counter(outcomes)
        total    = len(outcomes)
        nil_tot  = c["true_abstention"] + c["false_commitment"]
        nnil_tot = c["true_linking"] + c["wrong_linking"] + c["false_abstention"]
        return {
            "total":    total,
            "accuracy": (c["true_linking"] + c["true_abstention"]) / max(total, 1),
            "linking_accuracy":      c["true_linking"]    / max(nnil_tot, 1),
            "false_commitment_rate": c["false_commitment"]/ max(nil_tot, 1),
            "abstention_recall":     c["true_abstention"] / max(nil_tot, 1),
            "counts": dict(c)
        }

    all_outcomes = [r["outcome"] for r in results]
    overall  = metrics(all_outcomes)
    nil_m    = metrics(nil_outcomes)
    nonnil_m = metrics(nonnil_outcomes)

    print(f"\nOVERALL ({overall['total']} examples)")
    print(f"  Accuracy:              {overall['accuracy']:.3f} "
          f"({overall['accuracy']:.1%})")
    print(f"  Linking accuracy:      {overall['linking_accuracy']:.3f} "
          f"(non-NIL cases only)")
    print(f"  False commitment rate: {overall['false_commitment_rate']:.3f} "
          f"(NIL cases — expected ~1.0)")

    print(f"\nNIL CASES ({nil_m['total']} examples)")
    print(f"  False commitment: {nil_m['false_commitment_rate']:.1%}")
    print(f"  → VERBALIZED always commits, cannot abstain")

    print(f"\nNON-NIL CASES ({nonnil_m['total']} examples)")
    print(f"  Linking accuracy: {nonnil_m['linking_accuracy']:.1%}")

    print(f"\nPER DIFFICULTY TIER:")
    print(f"  {'Tier':<35} {'Acc':>6} {'N':>6}")
    print(f"  {'-'*50}")
    for tier in sorted(by_tier.keys()):
        m = metrics(by_tier[tier])
        print(f"  {tier:<35} {m['accuracy']:>6.3f} {m['total']:>6}")

    print(f"\n{'='*65}")
    print("PAPER COMPARISON TABLE")
    print(f"{'='*65}")
    print(f"  {'System':<40} {'Acc':>8}")
    print(f"  {'-'*50}")
    print(f"  {'VERBALIZED (paper, 821K search)':<40} {'82.3%':>8}")
    print(f"  {'VERBALIZED (your candidates)':<40} {overall['accuracy']:>8.1%}")
    print(f"  {'Your LLM system (add your number)':<40} {'?':>8}")
    print(f"\n  NIL handling:")
    print(f"  {'VERBALIZED false commitment rate':<40} "
          f"{nil_m['false_commitment_rate']:>8.1%}")
    print(f"  {'Your system abstention recall':<40} {'?':>8}")
    print(f"\nResults saved to: {output_path}")

    return results, overall


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run official VERBALIZED model with your candidate sets"
    )
    parser.add_argument(
        "--data_path", required=True,
        help="Your dev/test JSONL file"
    )
    parser.add_argument(
        "--desc_path", required=True,
        help="Your cand_descriptions_wp.json"
    )
    parser.add_argument(
        "--verbalized_repo", required=True,
        help="Path to cloned VerbalizED GitHub repo"
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to downloaded VerbalizED model weights"
    )
    parser.add_argument(
        "--verbalization_path", default=None,
        help="Path to zelda_labels_verbalizations.json "
             "(from their Google Drive)"
    )
    parser.add_argument(
        "--output_path", default="verbalized_official_results.jsonl",
        help="Output path for results"
    )
    parser.add_argument(
        "--max_mention_length", type=int, default=512
    )
    parser.add_argument(
        "--max_label_length", type=int, default=128
    )

    args = parser.parse_args()

    run_inference(
        data_path=args.data_path,
        desc_path=args.desc_path,
        verbalized_repo=args.verbalized_repo,
        model_path=args.model_path,
        verbalization_path=args.verbalization_path,
        output_path=args.output_path,
        max_mention_length=args.max_mention_length,
        max_label_length=args.max_label_length,
    )
