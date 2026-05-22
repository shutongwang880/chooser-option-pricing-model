# Chooser Option Pricing Pipeline

End-to-end quantitative toolkit for JPMorgan Chase (JPM) equity options: Monte Carlo pricing for Chooser-style payoffs, Black–Scholes–Merton (BSM) benchmarking against listed calls, and supervised ML extensions for volatility and price forecasting.

## Overview

The pipeline ingests market and macro inputs (spot, dividends, rates, VIX, realized volatility), builds a cleaned daily feature table, evaluates a BSM theoretical price against a liquid listed call, and trains two complementary ML routes:

1. **Volatility → BSM** — predict next-day 21-day rolling realized volatility, annualize, and price the evaluation call with BSM.
2. **Direct price** — regress BSM “teacher” call prices on a dense calendar using features that exclude same-day realized vol.

Chooser option valuation (`chooser_option_mc`) is provided for replication of the course reference model; error metrics for Weeks 4–5 are computed against the **listed vanilla call**, not the Chooser payoff.

## Features

- European call pricing with continuous dividend yield (`black_scholes_call`)
- Two-stage GBM Monte Carlo for a simplified Chooser contract (`chooser_option_mc`)
- Automated data pull and cleaning (Yahoo Finance, FRED, optional Alpha Vantage)
- Listed-option evaluation panel with MAE / RMSE vs BSM
- Chronological train / validation / test splits (70% / 15% / 15%)
- Model zoo: Random Forest, gradient boosting (sklearn / XGBoost), optional LSTM (vol), linear / HistGBDT / MLP (price)
- JSON metrics and CSV panel predictions under `data/`

## Requirements

- Python 3.9+
- See `requirements.txt` for core dependencies
- Optional LSTM: `pip install -r requirements-optional-lstm.txt` (TensorFlow)

On macOS, XGBoost may require OpenMP: `brew install libomp`.

## Installation

```bash
git clone <repository-url>
cd chooser_option_project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional API keys for full historical collection:

```bash
export FRED_API_KEY="your_key"
export ALPHA_VANTAGE_KEY="your_key"
```

Alternatively, place keys in a local `config.py` (gitignored).

## Usage

Run stages in order from the project root:

```bash
python3 week1_data_collection.py      # raw market series → data/raw_dataset.csv
python3 week2_data_cleaning.py          # features → data/cleaned_dataset.csv
python3 week4_baseline_evaluation.py  # BSM vs listed call → data/week4_*
python3 week5_run_ml.py                 # ML pipelines → data/week5_*
```

### Evaluation contract

Default listed proxy (Yahoo OCC):

| Field | Value |
|-------|--------|
| Symbol | `JPM260522C00320000` |
| Type | American-style call (priced as European BSM baseline) |
| Strike | 320 |
| Expiry | 2026-05-22 |

## Project structure

```
.
├── chooser_pricing.py           # BSM + Chooser Monte Carlo
├── config.json                  # Paper benchmark & simulation defaults
├── week1_data_collection.py
├── week2_data_cleaning.py
├── week4_cme_data.py            # Listed option helpers
├── week4_baseline_evaluation.py
├── week5_run_ml.py              # ML entry point
├── week5_ml/                    # Features, splits, models, metrics
├── data/                        # CSV / JSON artifacts
└── requirements.txt
```

## Configuration

`config.json` holds paper replication parameters (spot, rate, dividend yield, volatility), Chooser contract terms (`X`, `T1`, `T2`), simulation size, and column mappings for the cleaned dataset.

## Outputs

| Path | Description |
|------|-------------|
| `data/cleaned_dataset.csv` | Modeling panel with returns, rolling vol, macro features |
| `data/week4_evaluation_panel.csv` | Option panel aligned with underlying features |
| `data/week4_metrics.json` | BSM baseline MAE / RMSE |
| `data/week5_metrics.json` | Per-model test metrics and selection metadata |
| `data/week5_panel_predictions.csv` | Panel predictions from both ML tracks |

## Model selection

Models are chosen by **lowest validation MAE** on the chronological split. Production panel scoring uses:

- **Approach I:** Random Forest volatility → BSM
- **Approach II:** Histogram-based gradient boosting for call price

Held-out test metrics for all candidates are stored in `data/week5_metrics.json` under `test_metrics_by_model`.

## Data limitations

- Listed prices come from a **Yahoo Finance OCC proxy**, not official exchange feeds.
- Dividend yield from Yahoo may be percent-scaled; the pipeline normalizes to decimal form before pricing.
- The 23-row evaluation panel uses dates **after** the main training calendar; panel metrics are out-of-time checks, not in-sample fits.

## License

MIT — see [LICENSE](LICENSE).
