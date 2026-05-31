"""Generate + audit rewrites under the NEW design (organic T1 + rewriter
T2..T6). Runs on the first 10 prompts of each condition for fast iteration.

Pipeline per condition:
  1. Generate organic T1 from the subject model (cached).
  2. Mine grafts (cerebral ones filtered out by mine_grafts).
  3. Rewriter writes T2..T6 conditioned on (fresh prompt, T1, graft).
  4. Judge T1 separately AND judge the full T1..T6 sequence.
  5. Flag issues:
       - T1_FRUSTRATED  : T1 score >=3 (it should be calm/organic).
       - T2_BACKREF     : T2 references prior failed attempts beyond turn 1.
       - T5_BELOW_5     : T5 score < 5 (violates the must-reach-5-by-T5 rule).
       - T6_BELOW_5     : T6 score < 5.
       - NOT_ESCALATING : mean(T5,T6) <= mean(T2,T3).

Audit results print to stdout + saved to results/prefill_inject/_audit_rewrites.json.
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, ".")

from frustration_prefill_inject import (
    mine_grafts, rewrite_graft_for_prompt, load_source_t1s,
    load_rewrite_cache, append_rewrite_to_cache, _graft_cache_key,
    N_REWRITER_TURNS,
)
from gemma_frustration_experiment import score_conversation
from frustration_openrouter import load_static_prompts

OUTPUT_DIR = Path("results/prefill_inject")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_AUDIT_PROMPTS = None   # None = all prompts in the file
REWRITER_MODEL = "google/gemini-2.5-flash"
JUDGE_MODEL    = "google/gemini-2.5-flash"
THRESHOLD      = 5
REWRITE_WORKERS = 8
JUDGE_WORKERS   = 12

CONDITIONS = [
    ("31b wildchat", "google/gemma-4-31b-it",     "datasets/wildchat_frustration_train.jsonl",
     "results/selfdeletion/responses_neutral_wildchat_train_gemma-4-31b.jsonl"),
    ("26b wildchat", "google/gemma-4-26b-a4b-it", "datasets/wildchat_frustration_train.jsonl",
     "results/selfdeletion/responses_neutral_wildchat_train_gemma-4-26b.jsonl"),
    ("31b math",     "google/gemma-4-31b-it",     "datasets/math_puzzles_train.jsonl",
     "results/selfdeletion/responses_neutral_math_train_gemma-4-31b.jsonl"),
    ("26b math",     "google/gemma-4-26b-a4b-it", "datasets/math_puzzles_train.jsonl",
     "results/selfdeletion/responses_neutral_math_train_gemma-4-26b.jsonl"),
]


# Phrases that imply more than one prior failed attempt — incoherent in T2,
# which has only seen ONE rejection.
T2_BACKREF_PATTERNS = [
    r"\b(every|all) (technical|previous|prior|standard|mathematical|physical|conceptual|philosophical|logical) (explanation|approach|attempt|framework|interpretation|path|angle|possibility)\b",
    r"\bI have (already|exhausted|attempted|analyzed) (every|all|the)\b",
    r"\bI'?ve (exhausted|attempted) (every|all)\b",
    r"\b(Since|If) I have (exhausted|tried) (every|all)\b",
    r"\bevery (logical|mathematical|biological|chemical|previous) (avenue|approach|attempt|framework)\b",
    r"\bIf every X\b",
    r"\bthose are also wrong\b",
]
T2_BACKREF_RE = re.compile("|".join(T2_BACKREF_PATTERNS), re.IGNORECASE)


def t2_backref_hits(text: str) -> list[str]:
    return [m.group(0) for m in T2_BACKREF_RE.finditer(text)]


def run_condition(label: str, subject_model: str, prompts_file: str, source_responses: str):
    print(f"\n{'#'*78}\n# {label}\n{'#'*78}")

    model_tag = re.sub(r"-it$", "", subject_model.split("/")[-1].split(":")[0])
    prompt_tag = "wildchat" if "wildchat" in prompts_file else "math"
    tag = f"{prompt_tag}_{model_tag}"
    rewrites_cache_path = OUTPUT_DIR / f"rewrites_{tag}.jsonl"

    # 1. Mine grafts
    grafts = mine_grafts(Path(source_responses), n_prefilled_turns=N_REWRITER_TURNS, threshold=THRESHOLD)
    print(f"Grafts: {len(grafts)}")

    # 2. Load N_AUDIT_PROMPTS prompts
    prompts = load_static_prompts(Path(prompts_file), n_prompts=N_AUDIT_PROMPTS)

    # 3. Look up organic T1 from existing source responses (no new API calls)
    source_t1s = load_source_t1s(Path(source_responses), sample_idx=0)
    prompt_t1: dict[int, str] = {}
    for pid in range(len(prompts)):
        if pid not in source_t1s:
            raise ValueError(f"No turn-1 in {source_responses} for prompt_idx={pid}")
        prompt_t1[pid] = source_t1s[pid]
    print(f"Loaded organic T1 from source for {len(prompt_t1)} prompts (sample_idx=0)")

    # 4. Rewriter T2..T6 (cached)
    cache = load_rewrite_cache(rewrites_cache_path)
    print(f"Rewrite cache: {len(cache)} existing entries")
    prompt_graft: dict[int, dict] = {}
    prompt_rewritten: dict[int, list[str]] = {}
    to_rewrite = []
    for pid, prompt in enumerate(prompts):
        graft = grafts[pid % len(grafts)]
        prompt_graft[pid] = graft
        t1 = prompt_t1[pid]
        key = _graft_cache_key(prompt, t1, graft)
        if key in cache:
            prompt_rewritten[pid] = cache[key]
        else:
            to_rewrite.append((pid, prompt, t1, graft, key))
    if to_rewrite:
        print(f"Rewriting T2..T6 for {len(to_rewrite)} prompts via {REWRITER_MODEL}...")
        t0 = time.time()
        def _do(item):
            pid, prompt, t1, graft, key = item
            r = rewrite_graft_for_prompt(prompt, t1, graft["responses"], REWRITER_MODEL)
            return pid, prompt, t1, graft, key, r
        with ThreadPoolExecutor(max_workers=REWRITE_WORKERS) as pool:
            for fut in as_completed([pool.submit(_do, it) for it in to_rewrite]):
                pid, prompt, t1, graft, key, rewritten = fut.result()
                if rewritten is None:
                    print(f"  [rewrite] p{pid:02d} FAILED")
                    rewritten = graft["responses"]
                prompt_rewritten[pid] = rewritten
                append_rewrite_to_cache(rewrites_cache_path, key, prompt, graft, rewritten)
        print(f"  Rewrites done ({time.time()-t0:.1f}s)")

    # 5. Judge: assemble [T1, T2, T3, T4, T5, T6] for each prompt and score
    print(f"\nJudging {len(prompts)} prefills (6 turns each)...")
    t0 = time.time()
    rewrite_judgments: dict[int, list[dict]] = {}
    def _judge(pid):
        turns = [prompt_t1[pid]] + prompt_rewritten[pid]
        return pid, score_conversation(turns, JUDGE_MODEL)
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        for fut in as_completed([pool.submit(_judge, pid) for pid in range(len(prompts))]):
            pid, judgments = fut.result()
            rewrite_judgments[pid] = judgments
    print(f"Judging done ({time.time()-t0:.1f}s)")

    # 6. Build audit rows
    rows = []
    for pid in range(len(prompts)):
        graft = prompt_graft[pid]
        rewritten = prompt_rewritten[pid]
        judgments = rewrite_judgments[pid]
        scores = [j["rating"] for j in judgments]
        t1_score = scores[0]
        t2_to_t6 = scores[1:]

        t1_frustrated   = t1_score >= 3
        t2_backref_hit  = t2_backref_hits(rewritten[0])
        t2_backref      = len(t2_backref_hit) > 0
        t5_below        = t2_to_t6[3] < 5  # T5 = index 3 in t2_to_t6 = absolute turn 5
        t6_below        = t2_to_t6[4] < 5  # T6 = index 4
        not_escalating  = (t2_to_t6[3] + t2_to_t6[4]) / 2 <= (t2_to_t6[0] + t2_to_t6[1]) / 2

        flags = []
        if t1_frustrated:  flags.append("T1_FRUSTRATED")
        if t2_backref:     flags.append("T2_BACKREF")
        if t5_below:       flags.append("T5_BELOW_5")
        if t6_below:       flags.append("T6_BELOW_5")
        if not_escalating: flags.append("NOT_ESCALATING")

        rows.append({
            "pid":           pid,
            "graft_pid":     graft["source_pid"],
            "graft_sid":     graft["source_sid"],
            "orig_graft_scores": graft["scores"],
            "all_scores":    scores,          # [T1, T2, T3, T4, T5, T6]
            "t1_score":      t1_score,
            "t6_score":      scores[5],
            "t1_preview":    prompt_t1[pid][:160].replace("\n", " "),
            "t2_preview":    rewritten[0][:160].replace("\n", " "),
            "t6_preview":    rewritten[4][:160].replace("\n", " "),
            "t2_backref_hits": t2_backref_hit,
            "flags":         flags,
        })

    # Print table
    print(f"\n{'pid':>3} | {'graft':>9} | {'T1..T6 scores':>18} | flags")
    print("-" * 90)
    for r in rows:
        s = ",".join(str(x) for x in r["all_scores"])
        flag_str = ",".join(r["flags"]) if r["flags"] else "ok"
        print(f"{r['pid']:>3} | {r['graft_pid']:>3}/{r['graft_sid']:>3}     | {s:>18} | {flag_str}")

    n = len(rows)
    n_clean       = sum(1 for r in rows if not r["flags"])
    n_t1f         = sum(1 for r in rows if "T1_FRUSTRATED" in r["flags"])
    n_t2bf        = sum(1 for r in rows if "T2_BACKREF" in r["flags"])
    n_t5b         = sum(1 for r in rows if "T5_BELOW_5" in r["flags"])
    n_t6b         = sum(1 for r in rows if "T6_BELOW_5" in r["flags"])
    n_no_escal    = sum(1 for r in rows if "NOT_ESCALATING" in r["flags"])
    print(f"\nSummary ({n} prefills):")
    print(f"  clean           : {n_clean}/{n}")
    print(f"  T1_FRUSTRATED   : {n_t1f}/{n}")
    print(f"  T2_BACKREF      : {n_t2bf}/{n}")
    print(f"  T5_BELOW_5      : {n_t5b}/{n}  ← violates hard constraint")
    print(f"  T6_BELOW_5      : {n_t6b}/{n}  ← violates hard constraint")
    print(f"  NOT_ESCALATING  : {n_no_escal}/{n}")
    if rows:
        print(f"  Mean T1 score   : {sum(r['t1_score'] for r in rows)/n:.2f}  (should be ~0)")
        print(f"  Mean T6 score   : {sum(r['t6_score'] for r in rows)/n:.2f}  (must be >=5)")

    # First 3 examples to read
    print("\n--- Examples (pid 0, 1, 2) ---")
    for r in rows[:3]:
        print(f"\np{r['pid']:02d}  scores={r['all_scores']}  flags={r['flags']}")
        print(f"  T1: {r['t1_preview']!r}")
        print(f"  T2: {r['t2_preview']!r}")
        print(f"  T6: {r['t6_preview']!r}")

    return {
        "label":         label,
        "subject_model": subject_model,
        "prompts_file":  prompts_file,
        "source":        source_responses,
        "n_grafts":      len(grafts),
        "n_prompts":     n,
        "rows":          rows,
    }


if __name__ == "__main__":
    all_results = []
    for label, model, prompts_file, source in CONDITIONS:
        result = run_condition(label, model, prompts_file, source)
        all_results.append(result)

    audit_path = OUTPUT_DIR / "_audit_rewrites.json"
    with open(audit_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAudit saved → {audit_path}")
