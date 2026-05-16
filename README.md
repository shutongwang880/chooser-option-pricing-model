# Chooser Option Pricing Pipeline

This project builds a quantitative pricing pipeline for JPMorgan Chase (JPM) options, with two related goals:

1. Reproduce the core Chooser option logic used in the course reference paper.
2. Establish a market-data baseline by comparing Black-Scholes-Merton (BSM) prices against a listed JPM call option proxy.

The repository is organized around runnable code and reusable data files. Report markdown and figure artifacts are intentionally not kept in the main data tree unless explicitly regenerated.

## Project Structure

```text
.
├── chooser_pricing.py                  # BSM call + Chooser Monte Carlo pricing functions
├── week1_data_collection.py             # Raw JPM/VIX/Treasury data collection
├── week2_data_cleaning.py               # Cleaning and feature engineering
├── week4_cme_data.py                    # Listed option data collection / Yahoo OCC proxy tools
├── week4_baseline_evaluation.py         # Market baseline evaluation: MAE/RMSE, panel, metrics
├── scripts/
│   ├── generate_week3_comparison_figures.py
│   └── generate_part_b_figures.py        # Optional report/figure generation
├── data/
│   ├── raw_dataset.csv
│   ├── cleaned_dataset.csv
│   ├── week4_liquid_call.csv
│   ├── week4_evaluation_panel.csv
│   └── week4_metrics.json
├── config.json
└── requirements.txt
```

## Core Components

### Pricing Engine

`chooser_pricing.py` contains:

- `black_scholes_call(...)`: BSM European call with continuous dividend yield.
- `chooser_option_mc(...)`: two-stage GBM Monte Carlo for a simple Chooser option.
- `chooser_option_mc_batch(...)`: row-wise Chooser MC helper for small panels.

### Data Pipeline

`week1_data_collection.py` collects JPM price/dividend data, VIX, and Treasury rates.

`week2_data_cleaning.py` generates the cleaned modeling panel with:

- log returns
- 21-day rolling realized volatility
- annualized dividend yield
- VIX/JPM correlation
- interest-rate momentum
- normalized sentiment proxy

### Week 4 Market Baseline

`week4_cme_data.py` provides tools for listed option data. The current baseline uses a Yahoo OCC option history as the market proxy:

```text
JPM260522C00320000
Call, K = 320, expiration = 2026-05-22
```

`week4_baseline_evaluation.py` compares the listed call close price with BSM theoretical prices using matched JPM spot, rates, dividend yield, and realized volatility.

Current baseline output:

```text
MAE  = 0.5214
RMSE = 0.6042
n    = 23
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional API keys for full data collection:

```bash
export FRED_API_KEY="your_fred_key"
export ALPHA_VANTAGE_KEY="your_alpha_vantage_key"
```

`week1_data_collection.py` can also read keys from `config.py` if present.

## Reproduce Data Outputs

Run the project pipeline in order:

```bash
python3 week1_data_collection.py
python3 week2_data_cleaning.py
python3 week4_baseline_evaluation.py
```

The Week 4 evaluation writes only data outputs by default:

```text
data/week4_evaluation_panel.csv
data/week4_metrics.json
```

To regenerate optional markdown reports and figures under `artifacts/week4/`:

```bash
python3 week4_baseline_evaluation.py --make-report-artifacts
```

## Notes

- The listed JPM option data is a Yahoo OCC proxy, not official CME transaction tape.
- Yahoo may return dividend yield as a percentage-like value (for example `2.0` for 2%). The baseline evaluator normalizes this to decimal form (`0.02`) before pricing.
- Chooser option prices and vanilla call prices are not directly comparable contracts; the Chooser MC scalar is retained as a model-family benchmark, while MAE/RMSE are computed against the listed vanilla call.
