## Overview

This repository now contains a minimal-yet-extensible pipeline for training an LSTM
time-series model on the meteorological observations delivered as Excel files in
`data/`. The code paths are organised as follows:

- `src/config.py` – reusable dataclasses that describe paths, data schema, and
  training hyper-parameters.
- `src/data.py` – utilities to ingest the Excel files, clean the columns, and turn
  the long series into overlapping windows for model training.
- `src/model.py` – implementation of a stacked LSTM forecaster baseline.
- `scripts/prepare_data.py` – command-line script that converts the raw Excel files
  into a single processed CSV under `data/processed/`.
- `scripts/train_lstm.py` – reference training entry point that handles
  splitting/normalisation, trains the model, and saves checkpoints in
  `models/checkpoints/`.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements include `pandas`, `numpy`, `torch`, and `openpyxl` (needed for
reading `.xlsx` files).

## Typical workflow

1. Place the Excel files in `data/` (already present in this repository).
2. Generate a cleaned CSV (resampled every 6 hours by default):
   ```bash
   python scripts/prepare_data.py
   ```
3. Train the baseline LSTM (use `--station <name>` to pick a single station):
   ```bash
   python scripts/train_lstm.py --epochs 30 --station Greatwall-MeteorologicalObservation-1985-2022
   ```

Model checkpoints and normalisation metadata are stored at
`models/checkpoints/lstm_<station>.pt`. You can adjust the resampling frequency,
window size, or other hyper-parameters via the dataclasses in `src/config.py` or
the CLI overrides (see `--help` for each script).

### Evaluate & plot

After training, run:

```bash
python scripts/evaluate_lstm.py \
  --checkpoint models/checkpoints/lstm_all.pt \
  --plot plots/lstm_holdout.png \
  --station Greatwall-MeteorologicalObservation-1985-2022
```

The script reloads the model, computes MAE/RMSE on the hold-out windows, and
produces an actual-vs-predicted plot (first forecast step) saved under `plots/`.
Use `--max-points` to limit the number of points drawn when the hold-out set is
large. If Matplotlib warns about missing Chinese glyphs, supply a font that
contains them, e.g.

```bash
python scripts/evaluate_lstm.py \
  --font-file /usr/share/fonts/truetype/noto-cjk/NotoSansCJK-Regular.ttc
```

or point to any `.ttf/.otf` file available on your system.
