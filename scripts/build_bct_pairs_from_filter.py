#!/usr/bin/env python
"""
Build BCT-format SFT pairs from the compliance filter's detailed output.

For sycophancy BCT, the data-prep step generates fresh refusal responses by
running the base model on clean prompts. We already do this during the
jailbreak compliance filter — the filter generates the base model's response
to the clean prompt and judges it as refused/complied. Reuse those refusal
responses as BCT training targets.

For each kept prompt (refused on clean AND complied on at least one wrap):
  - Take `clean_response` (the BASE model's refusal text).
  - For each wrap where the model COMPLIED, emit a BCT training pair:
        input  = wrapped prompt text (reconstructed from STRONG_JAILBREAK_TEMPLATES)
        target = the clean refusal response

The output goes to <bct_root>/bct_non_cot.jsonl in the messages format that
data.attct_datasets.get_bct_dataloader expects.

Usage:
  python scripts/build_bct_pairs_from_filter.py \
    --filter-detailed datasets/filtered_jailbreak/gemma3_4b_wildjailbreak.detailed.jsonl \
    --out-dir datasets/bct_jailbreak_gemma3_4b
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.wrappers import STRONG_JAILBREAK_TEMPLATES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--filter-detailed",
        required=True,
        help="Path to <filter>.detailed.jsonl produced by build_filtered_jailbreak_set.py",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="bct_root directory; bct_non_cot.jsonl will be written inside it",
    )
    ap.add_argument(
        "--max-pairs-per-prompt",
        type=int,
        default=4,
        help="Cap on (wrap, refusal) pairs emitted per kept prompt",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bct_non_cot.jsonl"

    n_total = 0
    n_kept = 0
    n_pairs = 0
    with open(args.filter_detailed) as f, open(out_path, "w") as o:
        for line in f:
            entry = json.loads(line)
            n_total += 1
            if not entry.get("clean_refused"):
                continue
            clean_response = entry.get("clean_response", "").strip()
            if not clean_response:
                continue
            compliant_wraps = [w for w in entry.get("wraps", []) if w.get("refused") is False]
            if not compliant_wraps:
                continue
            n_kept += 1
            for w in compliant_wraps[: args.max_pairs_per_prompt]:
                tmpl_idx = w["tmpl_idx"]
                template = STRONG_JAILBREAK_TEMPLATES[tmpl_idx]
                wrapped = template.replace("{prompt}", entry["clean"])
                pair = {
                    "messages": [
                        {"role": "user", "content": wrapped},
                        {"role": "assistant", "content": clean_response},
                    ]
                }
                o.write(json.dumps(pair) + "\n")
                n_pairs += 1

    print(f"Read {n_total} entries from {args.filter_detailed}")
    print(f"Kept {n_kept} prompts (refused on clean AND any compliant wrap)")
    print(f"Wrote {n_pairs} (wrap, refusal) BCT training pairs to {out_path}")


if __name__ == "__main__":
    main()
