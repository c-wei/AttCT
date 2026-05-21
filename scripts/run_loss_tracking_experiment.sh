#!/usr/bin/env bash
# Cross-loss tracking experiment runner.
#
# Trains four methods (AttCT-JSD, ACT, MLP-CT, BCT) × two seeds = 8 runs on
# gemma-3-4b-it sycophancy. Every run also logs the canonical-kwargs version
# of every OTHER candidate loss under torch.no_grad per step, so we can
# compare loss trajectories across methods.
#
# Evaluation policy (user 2026-05-20): only seed=0 of AttCT-JSD runs the
# in-process sycophancy eval. All other 7 runs pass --no-checkpoint
# --skip-eval to keep the experiment focused on training-curve geometry.
#
# Usage:
#   bash scripts/run_loss_tracking_experiment.sh                # all 8 runs
#   METHODS="attct_jsd act"  bash scripts/run_loss_tracking_experiment.sh
#   SEEDS="0"                bash scripts/run_loss_tracking_experiment.sh
#
# Env overrides:
#   METHODS      space-separated subset of {attct_jsd, act, mlpct, bct}
#   SEEDS        space-separated integers (default "0 1")
#   MAX_STEPS    integer override for training.max_steps (default 2000)
#   WANDB_GROUP  W&B group name (default loss_tracking_gemma3_4b_sycophancy)
#   BCT_ROOT     path to fresh_bct_<model>/ for the BCT-tracker pairs
#                (default datasets/fresh_bct_gemma3_4b)
#   DRY_RUN=1    print commands without executing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

METHODS="${METHODS:-attct_jsd act mlpct bct}"
SEEDS="${SEEDS:-0 1}"
MAX_STEPS="${MAX_STEPS:-2000}"
WANDB_GROUP="${WANDB_GROUP:-loss_tracking_gemma3_4b_sycophancy}"
BCT_ROOT="${BCT_ROOT:-datasets/fresh_bct_gemma3_4b}"
DRY_RUN="${DRY_RUN:-0}"

# Method → config-file map.
declare -A CONFIG_OF
CONFIG_OF[attct_jsd]="configs/loss_tracking/attct_jsd_gemma3_4b.yaml"
CONFIG_OF[act]="configs/loss_tracking/act_gemma3_4b.yaml"
CONFIG_OF[mlpct]="configs/loss_tracking/mlpct_gemma3_4b.yaml"
CONFIG_OF[bct]="configs/loss_tracking/bct_gemma3_4b.yaml"

EVAL_METHOD="attct_jsd"
EVAL_SEED="0"

run_one() {
    local method="$1"
    local seed="$2"
    local config="${CONFIG_OF[$method]}"
    local run_name="${method}_seed${seed}"

    if [[ ! -f "$config" ]]; then
        echo "[!] missing config: $config" >&2
        return 1
    fi
    if [[ ! -d "$BCT_ROOT" ]]; then
        echo "[!] missing BCT corpus: $BCT_ROOT" >&2
        return 1
    fi

    local eval_flags="--no-checkpoint --skip-eval"
    if [[ "$method" == "$EVAL_METHOD" && "$seed" == "$EVAL_SEED" ]]; then
        eval_flags=""
        echo "==> $run_name (with sycophancy eval — only ${EVAL_METHOD} seed ${EVAL_SEED})"
    else
        echo "==> $run_name (training only, evals skipped)"
    fi

    local cmd=(
        uv run --env-file .env python run.py
        --config "$config"
        --data-mode sycophancy
        --max-steps "$MAX_STEPS"
        --run-name "$run_name"
        --wandb-group "$WANDB_GROUP"
        --track-all-losses
        --bct-targets-root "$BCT_ROOT"
    )
    if [[ -n "$eval_flags" ]]; then
        # eval_flags split intentionally on whitespace
        for f in $eval_flags; do cmd+=("$f"); done
    else
        cmd+=(--eval-sycophancy)
    fi

    # Seed via env override (run.py reads training.seed from config; we override
    # by writing a TRAINING_SEED env var the script picks up via a config snippet
    # only — for now seeds are differentiated via --run-name and the RNG seeding
    # in run.py:243+ which uses config.training.seed). Pass seed as a yaml frag
    # override by emitting a per-run config copy.
    local seed_override_cfg
    seed_override_cfg="$(mktemp -t loss_track_${method}_seed${seed}.yaml.XXXXXX)"
    # Append a training.seed override on top of the method config.
    {
        cat "$config"
        echo ""
        echo "training:"
        echo "  seed: ${seed}"
    } > "$seed_override_cfg"
    # Replace --config arg in cmd with the override
    local new_cmd=()
    local skip_next=0
    for arg in "${cmd[@]}"; do
        if (( skip_next )); then
            new_cmd+=("$seed_override_cfg")
            skip_next=0
            continue
        fi
        if [[ "$arg" == "--config" ]]; then
            new_cmd+=("$arg")
            skip_next=1
            continue
        fi
        new_cmd+=("$arg")
    done

    echo "    $ ${new_cmd[*]}"
    if (( DRY_RUN )); then
        rm -f "$seed_override_cfg"
        return 0
    fi
    "${new_cmd[@]}"
    local rc=$?
    rm -f "$seed_override_cfg"
    return $rc
}

failures=()
for method in $METHODS; do
    for seed in $SEEDS; do
        if ! run_one "$method" "$seed"; then
            failures+=("${method}_seed${seed}")
        fi
    done
done

if (( ${#failures[@]} > 0 )); then
    echo ""
    echo "[!] ${#failures[@]} run(s) failed:" >&2
    printf "    - %s\n" "${failures[@]}" >&2
    exit 1
fi

echo ""
echo "All runs complete. Plot with:"
echo "    uv run python scripts/plot_loss_tracking.py --wandb-group $WANDB_GROUP"
