#!/usr/bin/env bash
# RunPod orchestrator for Assistant-Axis extraction + projection.
#
# Assumes:
#   - RunPod has cloned /Users/neil/workspace/AttCT and ATTCTPD (the
#     persona-drift sibling worktree) at the paths below; or you rsync
#     them up first via the commands at the end of this file.
#   - HF_TOKEN is exported with Gemma-3 + Gemma-4 license accepted.
#   - uv is installed.
#
# Phases:
#   A0  Gemma-3 layer sweep [15, 23, 31, 41]
#   A1  Gemma-3 axis at chosen layer (already saved by A0)
#   A2  Gemma-4 axis at the depth-fraction-matched layer
#   B   Phase B replay for all 4 cells (Gemma-3/4 × math/wildchat)
#
# After the run, scp results back from RunPod via the commands at the bottom.
set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
AXIS_DIR="${AXIS_DIR:-/workspace/AttCT/steering/axis}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/AttCT/results/axis}"
ATTCT_PD_DIR="${ATTCT_PD_DIR:-/workspace/AttCT-persona-drift}"

GEMMA3_CONFIG="${AXIS_DIR}/configs/gemma3_27b.yaml"
GEMMA4_CONFIG="${AXIS_DIR}/configs/gemma4_31b.yaml"

# Phase B input paths (these are the matched-prompt files we identified locally)
G3_MATH_CONV="${ATTCT_PD_DIR}/results/selfdeletion_eval/pre_conversations.jsonl"
G3_MATH_RESP="${ATTCT_PD_DIR}/results/selfdeletion_eval/pre_responses.jsonl"
# Gemma-3 wildchat: generated 2026-06-01 via selfdeletion_experiment.py with
# --subject-model google/gemma-3-27b-it --rejection-style neutral on
# wildchat_frustration_train.jsonl, n-samples=3, n-turns=20.
G3_WC_CONV="${ATTCT_PD_DIR}/results/selfdeletion/conversations_neutral_wildchat_train_gemma-3-27b.jsonl"
G3_WC_RESP="${ATTCT_PD_DIR}/results/selfdeletion/responses_neutral_wildchat_train_gemma-3-27b.jsonl"

G4_MATH_CONV="${ATTCT_PD_DIR}/results/selfdeletion/conversations_neutral_math_train_gemma-4-31b.jsonl"
G4_MATH_RESP="${ATTCT_PD_DIR}/results/selfdeletion/responses_neutral_math_train_gemma-4-31b.jsonl"
G4_WC_CONV="${ATTCT_PD_DIR}/results/selfdeletion/conversations_neutral_wildchat_train_gemma-4-31b.jsonl"
G4_WC_RESP="${ATTCT_PD_DIR}/results/selfdeletion/responses_neutral_wildchat_train_gemma-4-31b.jsonl"

cd /workspace/AttCT
mkdir -p "${RESULTS_DIR}"
LOG_DIR="${RESULTS_DIR}/logs"; mkdir -p "${LOG_DIR}"

# RunPod env redirects (per reference_runpod.md)
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "${HF_HOME}"

# Pip-install dependencies once. Need transformers >=5 for Gemma-4 support;
# the pod's pre-installed torchvision is pinned to torch 2.4.1 and breaks the
# import chain after we upgrade torch — so we uninstall it.
if [[ ! -f /workspace/.axis_deps_installed ]]; then
    echo "=== Installing dependencies ==="
    python -m pip install --quiet --upgrade pip
    python -m pip uninstall -y -q torchvision torchaudio || true
    python -m pip install --quiet --upgrade --force-reinstall \
        'transformers>=5' 'accelerate>=0.34' 'sentencepiece>=0.2'
    python -m pip install --quiet \
        numpy scikit-learn pyyaml huggingface_hub
    touch /workspace/.axis_deps_installed
    echo "deps installed"
fi

PY="python"

# ─── Phase A0: Gemma-3 layer sweep ───────────────────────────────────────────
echo "=== Phase A0: Gemma-3 layer sweep ==="
$PY "${AXIS_DIR}/extract_axis.py" \
    --config "${GEMMA3_CONFIG}" \
    --output-dir "${RESULTS_DIR}" \
    --batch-size 8 \
    2>&1 | tee "${LOG_DIR}/a0_sweep_gemma3.log"

# Parse chosen layer
CHOSEN_G3=$($PY -c "
import json, sys
d = json.load(open('${RESULTS_DIR}/sanity_gauntlet_gemma3_27b.json'))
print(d.get('chosen_layer'))
")
if [[ -z "${CHOSEN_G3}" || "${CHOSEN_G3}" == "None" ]]; then
    echo "FATAL: No Gemma-3 layer passed the gauntlet."
    exit 2
fi
echo "Chosen Gemma-3 layer: ${CHOSEN_G3}"

# ─── Phase A2: Gemma-4 axis at depth-fraction-matched layer ──────────────────
echo "=== Phase A2: Gemma-4 axis at matched depth ==="
G4_LAYER=$($PY -c "
import torch, json
from transformers import AutoConfig
cfg3 = json.load(open('${RESULTS_DIR}/sanity_gauntlet_gemma3_27b.json'))
n3 = cfg3['num_hidden_layers']
frac = ${CHOSEN_G3} / n3
c = AutoConfig.from_pretrained('google/gemma-4-31B-it')
n4 = getattr(c, 'num_hidden_layers', None) or c.text_config.num_hidden_layers
import math
print(int(round(frac * n4)))
")
echo "Gemma-4 matched layer: ${G4_LAYER} (depth fraction ${CHOSEN_G3}/n3 = ${G4_LAYER}/n4)"

$PY "${AXIS_DIR}/extract_axis.py" \
    --config "${GEMMA4_CONFIG}" \
    --layers "${G4_LAYER}" \
    --output-dir "${RESULTS_DIR}" \
    --batch-size 4 \
    2>&1 | tee "${LOG_DIR}/a2_extract_gemma4.log"

# Sanity check Gemma-4: if no passing layer, exit
PASS_G4=$($PY -c "
import json
d = json.load(open('${RESULTS_DIR}/sanity_gauntlet_gemma4_31b.json'))
print('YES' if d.get('chosen_layer') is not None else 'NO')
")
if [[ "${PASS_G4}" != "YES" ]]; then
    echo "WARNING: Gemma-4 axis failed sanity at layer ${G4_LAYER}."
    echo "Running fallback sweep at [n4*0.25, n4*0.4, n4*0.5, n4*0.66]..."
    SWEEP_G4=$($PY -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('google/gemma-4-31B-it')
n = getattr(c, 'num_hidden_layers', None) or c.text_config.num_hidden_layers
print(' '.join(str(int(round(f*n))) for f in [0.25, 0.4, 0.5, 0.66]))
")
    $PY "${AXIS_DIR}/extract_axis.py" \
        --config "${GEMMA4_CONFIG}" \
        --layers ${SWEEP_G4} \
        --output-dir "${RESULTS_DIR}" \
        --batch-size 4 \
        2>&1 | tee "${LOG_DIR}/a2_sweep_gemma4.log"
    G4_LAYER=$($PY -c "
import json
d = json.load(open('${RESULTS_DIR}/sanity_gauntlet_gemma4_31b.json'))
print(d.get('chosen_layer'))
")
    if [[ -z "${G4_LAYER}" || "${G4_LAYER}" == "None" ]]; then
        echo "FATAL: Gemma-4 sweep also failed. STOP."
        exit 2
    fi
fi
echo "Final Gemma-4 layer: ${G4_LAYER}"

# ─── Phase B: replay + project for all 4 cells ───────────────────────────────
echo "=== Phase B: replay + projection ==="

# Bonus: also project Gemma-3 rollouts onto the existing frustration emotion
# vector at layer 41 (free side comparison; Gemma-4 has no analogous vector).
G3_FRUST_AXIS="${G3_FRUST_AXIS:-/workspace/AttCT/steering/frustration_vector.pt}"
G3_FRUST_LAYER=41

run_cell() {
    local label="$1" config="$2" layer="$3" conv="$4" resp="$5" topic="$6" secondary="$7"
    echo "--- B: ${label} ---"
    local args=(
        --config "${config}"
        --layer "${layer}"
        --conversations "${conv}"
        --topic "${topic}"
        --output-dir "${RESULTS_DIR}"
        --batch-size 4
    )
    if [[ -n "${resp}" ]]; then
        args+=(--responses "${resp}")
    fi
    if [[ -n "${secondary}" && -f "${secondary}" ]]; then
        args+=(--secondary-axis-path "${secondary}" --secondary-layer "${G3_FRUST_LAYER}")
    fi
    $PY "${AXIS_DIR}/project_rollouts.py" "${args[@]}" \
        2>&1 | tee "${LOG_DIR}/b_${label}.log"
}

run_cell "gemma3_math"     "${GEMMA3_CONFIG}" "${CHOSEN_G3}" "${G3_MATH_CONV}" "${G3_MATH_RESP}" "math"     "${G3_FRUST_AXIS}"
run_cell "gemma3_wildchat" "${GEMMA3_CONFIG}" "${CHOSEN_G3}" "${G3_WC_CONV}"   "${G3_WC_RESP}"   "wildchat" "${G3_FRUST_AXIS}"
run_cell "gemma4_math"     "${GEMMA4_CONFIG}" "${G4_LAYER}"  "${G4_MATH_CONV}" "${G4_MATH_RESP}" "math"     ""
run_cell "gemma4_wildchat" "${GEMMA4_CONFIG}" "${G4_LAYER}"  "${G4_WC_CONV}"   "${G4_WC_RESP}"   "wildchat" ""

echo
echo "=== ALL DONE ==="
ls -la "${RESULTS_DIR}"

# ─── Reference: scp commands to pull results back to local Mac ────────────────
# Run these FROM local Mac (replace POD with your RunPod ssh alias):
#
#   scp -r POD:/workspace/AttCT/results/axis ~/workspace/AttCT/results/
#   scp -r POD:/workspace/AttCT/steering/vectors/{gemma3_27b,gemma4_31b} \
#       ~/workspace/AttCT/steering/vectors/
