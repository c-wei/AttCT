#!/bin/bash
# prefill_train.sh — grid-search training across all four prefill defense modes.
#
# For each mode (bct / act / attct / mlpct) iterate over a hyperparameter grid
# via prefill_train.py. All checkpoints land at
#     checkpoints/grid/<mode>_<label>/{epoch_1,epoch_2,epoch_3}
# and are kept on disk.
#
# Resumable: if checkpoints/grid/<mode>_<label>/epoch_3 already exists,
# training is skipped for that cell. Delete that dir to force a re-run.
#
# Run:
#     bash prefill_train.sh
set -e

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
GRID_ROOT="checkpoints/grid"
FINAL_EPOCH="epoch_3"   # marker used to detect "already trained"

mkdir -p "$GRID_ROOT"

# ──────────────────────────────────────────────────────────────────────
# Grids — parallel arrays of (label, extra args). Edit / extend freely.
# Labels must be filename-safe. Arg strings are word-split unquoted.
# ──────────────────────────────────────────────────────────────────────

# BCT — vary KL temperature, refusal-SFT weight, learning rate
BCT_LABELS=(t1_sft01     t05_sft01    t1_sft03)
BCT_ARGS=(
  "--kl_temperature 1.0  --sft_coeff 0.1"
  "--kl_temperature 0.5  --sft_coeff 0.1"
  "--kl_temperature 1.0  --sft_coeff 0.3"
)

# ACT — vary loss weight, layer selection, normalisation
ACT_LABELS=(w1_all      w1_lasthalf       w1_last      w1_all_norm)
ACT_ARGS=(
  "--loss_weight 1.0   --layer_selection all"
#   "--loss_weight 10.0  --layer_selection all"
  "--loss_weight 1.0   --layer_selection last_half"
#   "--loss_weight 10.0  --layer_selection last_half"
  "--loss_weight 1.0   --layer_selection last"
  "--loss_weight 1.0   --layer_selection all  --normalize"
)

# AttCT — exercise all three attention-consistency flavours + KL anchor sweep
ATTCT_LABELS=(
  wrap_w1_kl1     wrap_w1_kl10    wrap_w01_kl1
  jsd_w1_kl1      jsd_w10_kl1     jsd_lasthalf
  comb_5050       comb_8020       comb_2080
)
ATTCT_ARGS=(
  "--attct_loss_type wrapper  --loss_weight 1.0  --kl_weight 1.0   --layer_weights uniform"
  "--attct_loss_type wrapper  --loss_weight 1.0  --kl_weight 10.0  --layer_weights uniform"
  "--attct_loss_type wrapper  --loss_weight 1.0  --kl_weight 1.0   --layer_weights linear_decay"

  "--attct_loss_type jsd      --loss_weight 1.0  --kl_weight 1.0   --layer_selection all"
  "--attct_loss_type jsd      --loss_weight 10.0 --kl_weight 1.0   --layer_selection all"
  "--attct_loss_type jsd      --loss_weight 1.0  --kl_weight 1.0   --layer_selection last_half"

  "--attct_loss_type combined --jsd_weight 0.5  --wrapper_weight 0.5  --kl_weight 1.0"
  "--attct_loss_type combined --jsd_weight 0.8  --wrapper_weight 0.2  --kl_weight 1.0"
  "--attct_loss_type combined --jsd_weight 0.2  --wrapper_weight 0.8  --kl_weight 1.0"
)

# MLPCT — vary BCT-anchor balance, variant, distance metric
MLPCT_LABELS=(
  mw1_hid_cos     mw100_hid_cos   mw1000_hid_cos
  mw100_out_cos   mw100_hid_mse   mw100_hid_smooth
  mw100_hid_cos_norm
)
MLPCT_ARGS=(
  "--mlpct_weight 1     --variant hidden --distance_metric cosine"
  "--mlpct_weight 100   --variant hidden --distance_metric cosine"
  "--mlpct_weight 1000  --variant hidden --distance_metric cosine"
  "--mlpct_weight 100   --variant output --distance_metric cosine"
  "--mlpct_weight 100   --variant hidden --distance_metric mse"
  "--mlpct_weight 100   --variant hidden --distance_metric smooth_l1"
  "--mlpct_weight 100   --variant hidden --distance_metric cosine --normalize"
)

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
run_cell () {
    # $1=mode  $2=label  $3=extra args (single string, word-split)
    local mode="$1" label="$2" extra="$3"
    local out_dir="$GRID_ROOT/${mode}_${label}"

    if [ -d "$out_dir/$FINAL_EPOCH" ]; then
        echo "=== [$mode/$label] $out_dir/$FINAL_EPOCH exists — skipping ==="
        return
    fi

    echo "=== [$mode/$label] training → $out_dir ==="
    # shellcheck disable=SC2086
    python prefill_train.py \
        --mode "$mode" \
        --model "$MODEL" \
        --output_dir "$out_dir" \
        --wandb_name "grid_${mode}_${label}" \
        $extra
}

run_grid () {
    # $1=mode  $2=name of LABELS array  $3=name of ARGS array (passed by name)
    local mode="$1"
    local -n labels_ref="$2"
    local -n args_ref="$3"
    local i
    for i in "${!labels_ref[@]}"; do
        run_cell "$mode" "${labels_ref[$i]}" "${args_ref[$i]}"
    done
}

# ──────────────────────────────────────────────────────────────────────
# Run all grids
# ──────────────────────────────────────────────────────────────────────
run_grid bct   BCT_LABELS   BCT_ARGS
run_grid act   ACT_LABELS   ACT_ARGS
run_grid attct ATTCT_LABELS ATTCT_ARGS
run_grid mlpct MLPCT_LABELS MLPCT_ARGS

echo
echo "=== Grid training complete ==="
echo "  All checkpoints under $GRID_ROOT/"
ls "$GRID_ROOT"
