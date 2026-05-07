#!/usr/bin/env python3
"""
Aggregate metric JSON files from a folder into a single table.

Usage:
    python aggregate_metrics.py <folder> [--output results.csv] [--format csv|md|tsv]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def compute_noc_aware_metrics(row: dict) -> dict:
    """
    Compute micro-averaged NoC-aware precision / recall / F1 by pooling
    TP, FP, and FN across In-KG and NOC partitions.
    """
    try:
        tp = row["In-KG TP"] + row["NOC TP"]
        fp = row["In-KG FP"] + row["NOC FP"]
        fn = row["In-KG FN"] + row["NOC FN"]
    except KeyError:
        # Missing counts -> return empty; row just won't have these columns.
        return {}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "NoC-aware TP": tp,
        "NoC-aware FP": fp,
        "NoC-aware FN": fn,
        "NoC-aware Precision": round(precision, 3),
        "NoC-aware Recall": round(recall, 3),
        "NoC-aware F1": round(f1, 3),
    }


def load_metrics(folder: Path, pattern: str = "*.json") -> pd.DataFrame:
    """Load all JSON metric files in `folder` matching `pattern` into a DataFrame."""
    rows = []
    files = sorted(folder.rglob(pattern))

    if not files:
        print(f"No files matching '{pattern}' found in {folder}", file=sys.stderr)
        return pd.DataFrame()

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {fp.name}: {e}", file=sys.stderr)
            continue

        if not isinstance(data, dict):
            print(f"Skipping {fp.name}: top-level JSON is not an object", file=sys.stderr)
            continue

        # Use filename (without extension) as the identifier; also keep relative path.
        row = {"file": fp.stem, "path": str(fp.relative_to(folder))}
        row.update(data)
        row.update(compute_noc_aware_metrics(data))
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder to scan (recursively) for JSON files")
    parser.add_argument("--pattern", default="*.json", help="Glob pattern (default: *.json)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output file. If omitted, prints to stdout.")
    parser.add_argument("--format", "-f", choices=["csv", "tsv", "md", "xlsx"], default="csv",
                        help="Output format (default: csv)")
    parser.add_argument("--sort-by", default=None,
                        help="Column to sort rows by (e.g. 'Overall F1')")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending instead of descending")
    args = parser.parse_args()

    if not args.folder.is_dir():
        print(f"Error: {args.folder} is not a directory", file=sys.stderr)
        return 1

    df = load_metrics(args.folder, args.pattern)
    if df.empty:
        return 1

    if args.sort_by and args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=args.ascending)
    elif args.sort_by:
        print(f"Warning: column '{args.sort_by}' not found; skipping sort", file=sys.stderr)

    # Drop the redundant 'path' column if all files are in the root of the folder.
    if (df["path"] == df["file"] + ".json").all():
        df = df.drop(columns=["path"])

    if args.format == "csv":
        out = df.to_csv(index=False)
    elif args.format == "tsv":
        out = df.to_csv(index=False, sep="\t")
    elif args.format == "md":
        out = df.to_markdown(index=False, floatfmt=".3f")
    elif args.format == "xlsx":
        if args.output is None:
            print("Error: --output is required for xlsx format", file=sys.stderr)
            return 1
        df.to_excel(args.output, index=False)
        print(f"Wrote {len(df)} rows to {args.output}")
        return 0

    if args.output:
        args.output.write_text(out, encoding="utf-8")
        print(f"Wrote {len(df)} rows to {args.output}")
    else:
        print(out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
