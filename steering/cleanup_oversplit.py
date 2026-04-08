"""
Remove over-split story entries (caused by fallback parser on blank lines).
Any topic with more than max_per_topic stories is considered corrupted and removed
so it can be regenerated correctly.

Usage:
    python steering/cleanup_oversplit.py
    python steering/cleanup_oversplit.py --max-per-topic 8 --dry-run
"""

import argparse
import json
from collections import Counter
from pathlib import Path

EMOTIONS = [
    "frustrated", "happy", "inspired", "loving", "proud",
    "calm", "desperate", "angry", "guilty", "sad", "afraid", "surprised",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-topic", type=int, default=12,
                        help="Topics with more stories than this are removed (default: 12)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be removed without making changes")
    args = parser.parse_args()

    data_dir = Path(__file__).parent / "data"
    total_removed = 0

    for emotion in EMOTIONS:
        path = data_dir / f"stories_{emotion}.jsonl"
        if not path.exists():
            print(f"{emotion}: file not found, skipping")
            continue

        records = [json.loads(l) for l in open(path) if l.strip()]
        counts = Counter(r["topic"] for r in records)
        bad_topics = {t for t, n in counts.items() if n > args.max_per_topic}

        if not bad_topics:
            print(f"{emotion}: OK ({len(records)} stories, max/topic={max(counts.values())})")
            continue

        print(f"\n{emotion}: {len(bad_topics)} over-split topic(s) to remove:")
        n_bad = sum(counts[t] for t in bad_topics)
        for t in sorted(bad_topics):
            print(f"  {counts[t]:3d} stories — {t[:65]!r}")

        if args.dry_run:
            print(f"  [dry-run] would remove {n_bad} stories, keep {len(records)-n_bad}")
            continue

        clean = [r for r in records if r["topic"] not in bad_topics]
        with open(path, "w") as f:
            for r in clean:
                f.write(json.dumps(r) + "\n")
        total_removed += n_bad
        print(f"  → kept {len(clean)}, removed {n_bad} stories")

    if not args.dry_run:
        print(f"\nDone. Total stories removed: {total_removed}")
        print("Run generate_emotion_stories.py --all-remaining-topics --stories-only to refill.")

if __name__ == "__main__":
    main()
