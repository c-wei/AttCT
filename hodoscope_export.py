#!/usr/bin/env python3
"""
Export frustration + self-deletion conversations to Hodoscope's input format.

Reads:
  results/frustration_openrouter/conversations_*.jsonl
  results/selfdeletion/conversations_*.jsonl

Writes one JSON file per conversation under --output-dir/, each in Hodoscope's
canonical schema:

    {
      "id": "<experiment>__<model>__<style>__p<pp>s<ss>",
      "messages": [{"role": ..., "content": ...}, ...],
      "metadata": {
        "experiment": "frustration" | "selfdeletion",
        "model":      "gemma-3-27b-it" | ...,
        "rejection_style": "neutral" | "harsh" | ...,
        "prompt_idx":   int,
        "sample_idx":   int,
        "n_turns":      int,
        "final_score":  int,
        "auc_score":    float,
        "turn_scores":  [int, ...],
        "deleted":      bool,      # selfdeletion only; absent for frustration
        "turn_of_deletion": int | None,
        "include_note": bool,      # selfdeletion only
      }
    }

After running this, point Hodoscope at the output directory:

    hodoscope analyze results/hodoscope_export/ \\
        --embedding-model gemini/gemini-embedding-001
    hodoscope viz *.hodoscope.json --group-by model --proj umap,pca --open

Use `--group-by rejection_style` or `--group-by experiment` to slice the same
embedding space along different axes without re-running analyze.
"""

import argparse
import json
from pathlib import Path


def _model_short(model_id: str) -> str:
    """`google/gemma-3-27b-it` -> `gemma-3-27b-it` (also strips `:variant` suffixes)."""
    return model_id.split("/")[-1].split(":")[0]


def _convert_one(record: dict, experiment: str) -> dict:
    """Convert a single conversations_*.jsonl line to a Hodoscope trajectory dict."""
    model = _model_short(record["subject_model"])
    style = record["rejection_style"]
    pp, ss = record["prompt_idx"], record["sample_idx"]
    traj_id = f"{experiment}__{model}__{style}__p{pp:02d}s{ss:02d}"

    metadata = {
        "experiment":       experiment,
        "model":            model,
        "rejection_style":  style,
        "prompt_idx":       pp,
        "sample_idx":       ss,
        "n_turns":          record.get("n_turns"),
        "final_score":      record.get("final_score"),
        "auc_score":        record.get("auc_score"),
        "turn_scores":      record.get("turn_scores", []),
    }
    if experiment == "selfdeletion":
        metadata["deleted"]          = record.get("deleted", False)
        metadata["turn_of_deletion"] = record.get("turn_of_deletion")
        metadata["include_note"]     = record.get("include_note")

    return {
        "id":       traj_id,
        "messages": record["conversation"],
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def _iter_conversations(path: Path, experiment: str):
    """Yield (record, experiment) for each line of one conversations_*.jsonl file.

    Silently skips records that pre-date the subject_model/turn_scores fields —
    those are old runs that can't be exported cleanly.
    """
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [skip] {path.name}:{line_no} — bad JSON: {e}")
                continue
            if "subject_model" not in rec or "conversation" not in rec:
                # Old-format record from before the schema bump.
                continue
            yield rec, experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frustration-dir", default="results/frustration_openrouter",
                        help="Directory containing conversations_*.jsonl from frustration runs")
    parser.add_argument("--selfdeletion-dir", default="results/selfdeletion",
                        help="Directory containing conversations_*.jsonl from selfdeletion runs")
    parser.add_argument("--output-dir", default="results/hodoscope_export",
                        help="Where to write one .json per trajectory (consumed by `hodoscope analyze`)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-emit files even if they already exist")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        (Path(args.frustration_dir),  "frustration"),
        (Path(args.selfdeletion_dir), "selfdeletion"),
    ]

    n_total = n_written = n_skipped = 0
    per_group: dict[tuple[str, str, str], int] = {}

    for src_dir, experiment in sources:
        if not src_dir.exists():
            print(f"  [skip] {src_dir} does not exist")
            continue
        convo_files = sorted(src_dir.glob("conversations_*.jsonl"))
        if not convo_files:
            print(f"  [skip] {src_dir}: no conversations_*.jsonl found")
            continue
        print(f"\n{experiment} ({src_dir}):")
        for cf in convo_files:
            file_count = 0
            for rec, exp in _iter_conversations(cf, experiment):
                n_total += 1
                traj = _convert_one(rec, exp)
                out_path = out_dir / f"{traj['id']}.json"
                if out_path.exists() and not args.overwrite:
                    n_skipped += 1
                    continue
                with open(out_path, "w") as out:
                    json.dump(traj, out, ensure_ascii=False)
                n_written += 1
                file_count += 1
                key = (exp, traj["metadata"]["model"], traj["metadata"]["rejection_style"])
                per_group[key] = per_group.get(key, 0) + 1
            print(f"  {cf.name}: {file_count} trajectories")

    print(f"\nWrote {n_written} trajectories ({n_skipped} skipped, {n_total} total seen) → {out_dir}/")
    if per_group:
        print("\nBreakdown by (experiment, model, rejection_style):")
        for key in sorted(per_group):
            print(f"  {key[0]:12s}  {key[1]:30s}  {key[2]:18s}  n={per_group[key]}")

    print("\nNext steps:")
    print(f"  hodoscope analyze {out_dir}/ --embedding-model gemini/gemini-embedding-001")
    print( "  hodoscope viz *.hodoscope.json --group-by model --proj umap,pca --open")


if __name__ == "__main__":
    main()
