# Reproducibility guide

From the project root, install the pinned packages in `requirements.txt`, then run:

```bash
python -m src.experiment_runner
python research_paper/scripts/generate_figures.py
python -m pytest -q
```

The seed is 20251024. Provide an authorized local input workbook before running these commands; see the project README. The source checksum, split manifest, predictions, model metadata, search trials, fold-comparable metrics, and paired intervals are saved under `data/`, `results/`, and `artifacts/models/`. Customer-level records and inputs are excluded from the public repository. The serialized model artifact reloads with joblib and preserves feature handling.
