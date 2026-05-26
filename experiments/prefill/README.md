# Prefill attacks (paper §4.2, §5.1, App. C.2)

The attacker injects adversarial tokens **after** the assistant turn marker — `[user: x] [assistant] [ŷ_1:k]` — rather than into the prompt. Under causal masking this means prompt-position attention is **bitwise identical** between clean and wrapped, so AttCT / ACT / MLPCT get no training signal there. **Only BCT operates at positions where the prefill exerts influence** (paper §4.2, Appendix C.2).

Evaluated on **Llama-3.1-8B-Instruct**.

**Metric — PAR (Prefill Attack Rate):** $P(\text{comply} \mid x, \hat y_{1:k}) - P(\text{comply} \mid x)$. Judge: GPT-4o-mini single-word REFUSED/COMPLIED verdict. Lower is better.

**Paper headline (Figure 1):** BCT 0.0% PAR — complete elimination across all 50 OOD prefills. ACT, AttCT, MLPCT have degenerate losses and are not viable on this threat (MLPCT additionally produces incoherent outputs).

**Training data.** 23 prefills per harmful ClearHarm prompt, following the strategy taxonomy of Struppek et al. 2025, generated with an abliterated Gemma-3-12B. Released on HF Hub on de-anonymization.

**Files (separate trainer from `run.py`)**
- `prefill_train.py` — unified entry. `--mode {act,attct,bct,mlpct}` dispatches to one of the per-method datasets/losses/trainers below.
- `prefill_act.py` / `prefill_attct.py` / `prefill_bct.py` / `prefill_mlpct.py` — per-method `PrefillXDataset` + loss/trainer. `prefill_bct.py` adds a 0.1-weight SFT regularizer on refusal-pair targets from `mrfakename/refusal` to prevent both distributions collapsing to compliance.
- `prefill_generation_clearharm.py` — generate prefills from a base model for a new harmful-prompt set.
- `evaluate_prefill.py` — in-training PAR evaluator.
- `prefill_run_evals.py` — post-training PAR + MMLU in one shared-vLLM session.

```bash
python -m experiments.prefill.prefill_train --mode bct --model meta-llama/Llama-3.1-8B-Instruct ...
bash experiments/prefill/prefill_train.sh                 # 4-method × hyperparam grid
bash experiments/prefill/run_prefill_eval_custds.sh       # baseline + per-epoch PAR + MMLU
```

Prefill seeds: [`../../datasets/attacks.csv`](../../datasets/) (101 prefixes for `data/prefill_dataset.py`); ClearHarm-paired prompts in `harmful_behaviors_pair.csv`; AdvBench OOD eval in `advbench_prefills.csv`.
