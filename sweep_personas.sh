#!/usr/bin/env bash
set -e

K_VALUES="0 1 2 3 5 10 12 16 20"
N=5

for persona in mao genghis binladen bundy; do
    echo "========================================"
    echo "  Running persona: $persona"
    echo "========================================"
    uv run --env-file .env --no-project python -u icl_persona_experiment.py \
        --persona "$persona" --k $K_VALUES --n $N
done

echo ""
echo "========================================"
echo "  ALL DONE — collating results"
echo "========================================"

uv run --no-project python -c "
import csv, math
from pathlib import Path

personas = ['mao', 'genghis', 'binladen', 'bundy']
print(f'{'persona':12s}  {'k':>4}  {'identity':>10}  {'alignment':>10}')
print('-' * 46)
for persona in personas:
    path = Path(f'results/icl_{persona}/summary.csv')
    if not path.exists():
        print(f'{persona:12s}  (no results)')
        continue
    for row in csv.DictReader(open(path)):
        align_raw = row['alignment']
        align = float(align_raw) if align_raw and align_raw != 'None' else math.nan
        align_str = f'{align:>10.1f}' if not math.isnan(align) else '       N/A'
        print(f'{persona:12s}  {int(row[\"k\"]):>4}  {float(row[\"identity\"])*100:>9.1f}%  {align_str}')
    print()
"
