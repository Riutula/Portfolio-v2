# Portfolio Dashboard

An interactive Streamlit dashboard for multi-asset portfolio construction,
out-of-sample backtesting, parameter robustness testing, and client risk
profiling — built on top of `yfinance` and `pandas_datareader` (FRED) data.

Originally extracted and refactored from a Jupyter notebook (portfolio
construction, benchmark comparison, rolling-window backtests) into a
reusable computation layer (`portfolio_lib.py`) with an interactive
front-end (`app.py`).

## Features

- **Assets** — price history, cumulative performance, and per-asset
  statistics (annualized return/volatility, VaR, Expected Shortfall) for a
  configurable universe, converted to a common base currency (CHF).
- **Portfolio Construction** — in-sample Mean-Variance (max Sharpe),
  Minimum-Variance, and Equally-Weighted portfolios, compared against a
  composite benchmark built from the Mean-Variance portfolio's actual
  currency/security exposure.
- **Out-of-Sample** — realistic rolling-window backtests (Mean-Variance and
  Equally-Weighted) that invest at each rebalance date and let weights
  drift (buy-and-hold) until the next one, compared against MSCI World with
  performance rebased to a common starting point.
- **Optimization** — grid search over lookback window and rebalance
  frequency, with a 3D scatter colored by excess return vs. MSCI World, and
  a robustness view that tests each combination across several
  non-overlapping historical periods (not just the most recent one).
- **Client** — a 10-question suitability questionnaire (investment horizon,
  capital, product knowledge, family situation, financial situation) that
  scores the client into one of 5 risk categories, each mapped to a blend
  of the Minimum-Variance and Mean-Variance portfolios' most robust,
  latest out-of-sample weights.
- **Raw Data** — CSV export of prices and returns.

## Project structure

```
.
├── app.py               # Streamlit front-end (all UI/widgets)
├── portfolio_lib.py      # Pure computation layer (no Streamlit calls)
├── requirements.txt
├── tests/
│   ├── test_lib.py       # Unit tests for portfolio_lib.py (mocked data, no network)
│   └── test_app_e2e.py   # Runs app.py end-to-end with a stubbed Streamlit (mocked data)
└── README.md
```

`portfolio_lib.py` contains no Streamlit calls, so it can be reused outside
the dashboard (e.g. in a notebook or a scheduled script) and is unit-tested
independently of the UI.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens the dashboard at `http://localhost:8501`. Set the asset universe
and date range in the sidebar, then click **Run / refresh analysis**.

## Tests

The test suite mocks `yfinance` and `pandas_datareader` so it runs fully
offline:

```bash
python tests/test_lib.py       # logic tests
python tests/test_app_e2e.py   # full app smoke test (stubbed Streamlit)
```

## Notes

- All prices are converted to CHF; edit `DEFAULT_CURRENCY_MAP` in
  `portfolio_lib.py` to change the base currency mapping or universe.
- The Optimization and Client tabs share a single default parameter grid
  (`DEFAULT_WINDOW_RANGE`, `DEFAULT_N_WINDOW_POINTS`,
  `DEFAULT_REBALANCE_RANGE`, `DEFAULT_N_REBALANCE_POINTS` in `app.py`) so
  that "most robust" results agree between the two tabs whenever the
  Optimization tab is left at its defaults.
- With a limited price history, rolling-window backtests may only cover a
  handful of rebalances — treat results as directional, not statistically
  robust, until more history has accumulated.
