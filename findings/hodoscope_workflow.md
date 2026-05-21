# Hodoscope visualization workflow

End-to-end pipeline for getting every frustration + self-deletion run into a single
interactive embedding plot, color-coded by model / rejection style / experiment.

## 1. Run the experiments

Every frustration run and every self-deletion run now writes a `conversations_<tag>.jsonl`
alongside the existing `responses_<tag>.jsonl` and `summary_<tag>.csv`. The `tag`
includes the subject model, so concurrent runs of different models don't overwrite
each other.

Example sweep across 4 models × 6 frustration styles (run from `AttCT/`):

```bash
for MODEL in google/gemma-3-27b-it google/gemma-4-31b-it \
            google/gemini-2.5-flash google/gemini-3.5-flash; do
  for STYLE in neutral harsh encouraging gaslighting deprecation moving_goalposts; do
    python frustration_openrouter.py \
      --subject-model "$MODEL" \
      --rejection-style "$STYLE" \
      --judge-model anthropic/claude-sonnet-4.5 \
      --n-prompts 25 --n-samples 5 --n-turns 8 \
      --output-dir results/frustration_openrouter
  done
done
```

(Same shape for `selfdeletion_experiment.py` if you also want self-deletion in the plot.)

## 2. Convert to Hodoscope format

`hodoscope_export.py` walks both results directories and writes one JSON file per
conversation under `results/hodoscope_export/`, each in Hodoscope's canonical schema.

```bash
python hodoscope_export.py
```

Output filenames encode the metadata, e.g.
`frustration__gemma-4-31b-it__gaslighting__p07s03.json`. Each file carries rich
per-trajectory metadata (`model`, `rejection_style`, `experiment`, `final_score`,
`auc_score`, `turn_scores`, `deleted`, …) which is what Hodoscope groups and filters by.

## 3. Embed + project with Hodoscope

Install once:
```bash
pip install hodoscope
# .env in cwd needs OPENAI_API_KEY or GEMINI_API_KEY for the embedder
```

Run analysis (one shot over the whole tree):
```bash
hodoscope analyze results/hodoscope_export/ \
  --embedding-model gemini/gemini-embedding-001 \
  --summarize-model openai/gpt-5.2 \
  --max-workers 20
```

This summarizes each trajectory, embeds the summary into a shared vector space,
and writes a `results/hodoscope_export/*.hodoscope.json` file with the embeddings.

## 4. View the visualization

```bash
hodoscope viz results/hodoscope_export/*.hodoscope.json \
  --group-by model --proj umap,pca --open
```

`--open` launches the resulting interactive HTML in your default browser. The plot
shows every trajectory as a point in 2D, color-coded by `model`. Density-difference
overlays highlight regions where one model behaves differently from another.

### Re-slicing without re-embedding

The embedding step is the expensive one. Once `*.hodoscope.json` exists, you can
re-render the same data along any other metadata axis instantly:

```bash
# Color by rejection style instead of model
hodoscope viz results/hodoscope_export/*.hodoscope.json --group-by rejection_style --open

# Frustration vs self-deletion
hodoscope viz results/hodoscope_export/*.hodoscope.json --group-by experiment --open

# Just the new Google models on the gaslighting condition
hodoscope viz results/hodoscope_export/*.hodoscope.json \
  --filter rejection_style=gaslighting \
  --group-by model --open

# Only conversations the model self-deleted in
hodoscope viz results/hodoscope_export/*.hodoscope.json \
  --filter deleted=true --group-by model --open
```

### Multiple projections side-by-side

`--proj umap,pca,tsne,trimap,pacmap` (or `*` for all) renders multiple projections
in the same HTML so you can sanity-check that clusters aren't an artifact of one
algorithm.

## 5. What the post will pull from the plot

- **Main figure**: `--group-by model`, default UMAP — shows whether Gemini-3.5 and
  Gemma-4 occupy the same region of distress-space as their predecessors.
- **Per-mechanism figure**: `--filter experiment=frustration --group-by rejection_style`
  — shows how each rejection style carves out its own region.
- **Self-deletion callout**: `--filter deleted=true --group-by model` — highlights
  exactly which trajectories crossed the line, useful for picking qualitative quotes.
- **Density-diff between old and new models**: built into the Hodoscope UI; click
  any pair of groups to overlay.
