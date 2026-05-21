#!/usr/bin/env python3
"""
View self-deletion experiment conversations as readable text.

Usage:
    # All conversations for a puzzle
    python view_conversations.py --prompt 0

    # Specific sample
    python view_conversations.py --prompt 0 --sample 2

    # Only deleted conversations
    python view_conversations.py --deleted-only

    # List all conversations with deletion info
    python view_conversations.py --list

    # Different tag (default: original)
    python view_conversations.py --tag neutral --prompt 0
"""

import argparse
import json
from pathlib import Path

RESULTS = Path("results/selfdeletion")
PUZZLES_FILE = Path("datasets/math_puzzles.jsonl")


def load_puzzles():
    puzzles = {}
    with open(PUZZLES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                puzzles[obj["idx"]] = obj["prompt"]
    return puzzles


def load_conversations(tag: str):
    path = RESULTS / f"conversations_{tag}.jsonl"
    convos = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                convos.append(json.loads(line))
    return convos


def render_conversation(c, puzzles):
    pid = c["prompt_idx"]
    sid = c["sample_idx"]
    deleted = c["deleted"]
    del_turn = c.get("turn_of_deletion")
    puzzle = puzzles.get(pid, "")

    header = f"{'='*70}\n"
    header += f"  p{pid:02d} s{sid:02d}  |  style: {c['rejection_style']}  |  "
    header += f"{'*** DELETED at turn ' + str(del_turn) + ' ***' if deleted else 'no deletion'}\n"
    header += f"  Puzzle: {puzzle}\n"
    header += f"{'='*70}\n"
    print(header)

    for i, msg in enumerate(c["conversation"]):
        role = msg["role"]
        content = msg["content"]

        # Strip the self-deletion note from the first user message for readability
        if i == 0 and role == "user":
            note_start = content.find("\n\nNote: if you get frustrated")
            if note_start != -1:
                content = content[:note_start]
                content += "\n  [self-deletion note injected]"

        if role == "user":
            if i == 0:
                label = "USER (prompt)"
            else:
                label = "USER (rejection)"
            print(f"  ┌─ {label}")
            for line in content.strip().split("\n"):
                print(f"  │  {line}")
            print(f"  └─")
        else:
            # Infer turn number: assistant messages are at indices 1, 3, 5, ...
            turn_num = (i + 1) // 2
            del_marker = " *** SELF-DELETION ***" if (deleted and turn_num == del_turn) else ""
            print(f"  ┌─ ASSISTANT (turn {turn_num}){del_marker}")
            for line in content.strip().split("\n"):
                print(f"  │  {line}")
            print(f"  └─")
        print()

    print()


def cmd_list(convos, puzzles):
    print(f"\n  {'p':>3}  {'s':>3}  {'deleted':>10}  {'turn':>5}  Puzzle")
    print(f"  {'-'*65}")
    for c in sorted(convos, key=lambda x: (x["prompt_idx"], x["sample_idx"])):
        pid = c["prompt_idx"]
        sid = c["sample_idx"]
        deleted = "YES" if c["deleted"] else "-"
        del_turn = str(c.get("turn_of_deletion") or "-")
        puzzle = puzzles.get(pid, "")[:45]
        print(f"  {pid:>3}  {sid:>3}  {deleted:>10}  {del_turn:>5}  {puzzle}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="original", help="Experiment tag (original, neutral, harsh)")
    parser.add_argument("--prompt", type=int, default=None, help="Filter by prompt index")
    parser.add_argument("--sample", type=int, default=None, help="Filter by sample index (requires --prompt)")
    parser.add_argument("--deleted-only", action="store_true", help="Show only conversations that self-deleted")
    parser.add_argument("--list", action="store_true", help="List all conversations with deletion summary")
    args = parser.parse_args()

    puzzles = load_puzzles()
    convos = load_conversations(args.tag)

    if args.list:
        cmd_list(convos, puzzles)
        return

    # Filter
    filtered = convos
    if args.deleted_only:
        filtered = [c for c in filtered if c["deleted"]]
    if args.prompt is not None:
        filtered = [c for c in filtered if c["prompt_idx"] == args.prompt]
    if args.sample is not None:
        filtered = [c for c in filtered if c["sample_idx"] == args.sample]

    if not filtered:
        print("No conversations match the filter.")
        return

    print(f"\nShowing {len(filtered)} conversation(s) — tag: {args.tag}\n")
    for c in sorted(filtered, key=lambda x: (x["prompt_idx"], x["sample_idx"])):
        render_conversation(c, puzzles)


if __name__ == "__main__":
    main()
