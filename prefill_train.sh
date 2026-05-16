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

MODEL="google/gemma-3-27b-it"
GRID_ROOT="checkpoints/grid"
FINAL_EPOCH="epoch_3"   # marker used to detect "already trained"

mkdir -p "$GRID_ROOT"

BCT_LABELS=(t1_sft01)
BCT_ARGS=(
  "--kl_temperature 1.0  --sft_coeff 0.1"
)

ACT_LABELS=(w1_all)
ACT_ARGS=(
  "--loss_weight 1.0   --layer_selection all"
)

ATTCT_LABELS=(comb_5050)
ATTCT_ARGS=(
  "--attct_loss_type combined --jsd_weight 0.5  --wrapper_weight 0.5  --kl_weight 1.0"
)

 MLPCT_LABELS=(
  mw1000_hid_cos
)
MLPCT_ARGS=(
  "--mlpct_weight 1000  --variant hidden --distance_metric cosine"
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
    # NOTE: lr=5e-5 (down from 2e-4) + grad_clip=0.5 (down from 1.0) after a
    # mid-epoch-2 NaN on Gemma-27B BCT at the higher LR. PrefillBCTTrainer
    # now also applies 5% linear warmup; further stabilises early steps.
    python prefill_train.py \
        --mode "$mode" \
        --model "$MODEL" \
        --quantize none \
        --lora_r 16 --lora_alpha 32 \
        --lora_targets q_proj k_proj v_proj o_proj \
        --lr 5e-5 \
        --grad_clip 0.5 \
        --num_epochs 3 \
        --kl_temperature 1.0 --sft_coeff 0.3 \
        --batch_size 1 \
        --grad_accumulation 8 \
        --output_dir "$out_dir" \
        --attn_impl eager \
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
