"""
portfolio_lib.py

Core computation layer for the portfolio dashboard, extracted from the
original analysis notebook (New_portfolio_2026_EN.ipynb). Every function
here is pure (data in -> data out): no plotting, no printing, no Streamlit
calls. This is what makes it reusable both by the Streamlit app and by the
original PDF report generator.

All prices/returns are expressed in CHF unless noted otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as web
from scipy.optimize import minimize

DEFAULT_TICKERS = [
    'CRM', 'HOLN.SW', 'LLY', 'NESN.SW', 'CTPNV.AS', '5JS.SI',
    'AMRZ', 'ASCN.SW', 'TPXE.PA', 'ECL', 'HYG', '0P00000BKL',
]

DEFAULT_CURRENCY_MAP = {
    'CRM': 'USD', 'HOLN.SW': 'CHF', 'LLY': 'USD', 'NESN.SW': 'CHF',
    'CTPNV.AS': 'EUR', '5JS.SI': 'SGD', 'AMRZ': 'USD', 'ASCN.SW': 'CHF',
    'TPXE.PA': 'EUR', 'ECL': 'USD', 'HYG': 'USD', '0P00000BKL': 'USD',
}

BENCHMARK_BY_CURRENCY = {
    'USD': 'SPY', 'CHF': 'CSSMI.SW', 'EUR': 'EXS1.DE', 'SGD': 'ES3.SI',
}
# TPXE.PA is an EUR-listed ETF tracking the Japanese TOPIX - mapped to itself
# rather than bucketed with other EUR names (see notebook Part 3 for the
# full rationale).
BENCHMARK_OVERRIDES = {
    'TPXE.PA': ('TPXE.PA', 'EUR'),
}

TYPE_LABELS = {
    'EQUITY': 'Equity', 'ETF': 'ETF', 'MUTUALFUND': 'Fund',
    'INDEX': 'Index', 'CURRENCY': 'Currency', 'CRYPTOCURRENCY': 'Crypto',
}
SECTOR_OVERRIDES = {
    'TPXE.PA': 'TOPIX index',
    '0P00000BKL': 'Short Term Money Market',
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def _to_series(x):
    """yf.download(...)['Close'] can return a Series or a 1-column
    DataFrame depending on context - normalize to a Series."""
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def detect_currency(ticker):
    """Best-effort currency detection for a ticker not in DEFAULT_CURRENCY_MAP
    (i.e. one the user typed in manually), via yfinance metadata. Falls back
    to 'USD' if detection fails - the price will still load, just make sure
    to sanity-check the FX conversion for that ticker if it's wrong."""
    try:
        info = yf.Ticker(ticker).info
        ccy = info.get('currency')
        if ccy:
            return ccy.upper()
    except Exception:
        pass
    return 'USD'


def load_fx_data(currencies, start_date, end_date):
    """Download FX rates vs CHF for every currency in `currencies` (CHF
    itself is skipped, no conversion needed)."""
    needed = sorted(set(currencies) - {'CHF'})
    if not needed:
        return pd.DataFrame()
    pairs = [c + 'CHF=X' for c in needed]
    data = yf.download(pairs, start=start_date, end=end_date, auto_adjust=True)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=pairs[0])
    return data


def load_prices_chf(tickers, currency_map, start_date, end_date, fx_data=None):
    """Download adjusted (dividend-inclusive) prices for `tickers` and
    convert them to CHF using `currency_map`."""
    prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    if fx_data is None:
        fx_data = load_fx_data(currency_map.values(), start_date, end_date)

    prices_chf = prices.copy()
    for t in tickers:
        ccy = currency_map.get(t, 'CHF')
        if ccy == 'CHF':
            continue
        fx_pair = ccy + 'CHF=X'
        if fx_pair in fx_data.columns:
            prices_chf[t] = prices[t] * fx_data[fx_pair].reindex(prices.index).ffill()

    return prices_chf.ffill()


def load_risk_free_rate(start_date, end_date, price_index):
    """US 3-month T-bill rate (FRED DTB3), aligned to `price_index` and
    annualized on a 252-trading-day basis."""
    raw = web.DataReader('DTB3', 'fred', start=start_date, end=end_date)
    daily = raw / (100 * 252)
    daily.columns = ['RF']
    daily = daily.reindex(price_index, method='ffill').bfill()
    annual_rate = float((1 + daily['RF'].mean()) ** 252 - 1)
    return daily, annual_rate


def get_asset_info(tickers):
    """Real security name, asset type and sector for each ticker."""
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception:
            info = {}

        name = info.get('longName') or info.get('shortName') or t

        if t in SECTOR_OVERRIDES:
            quote_type = 'ETF' if t == 'TPXE.PA' else 'MUTUALFUND'
            asset_type = TYPE_LABELS.get(quote_type, quote_type)
            sector = SECTOR_OVERRIDES[t]
        else:
            quote_type = info.get('quoteType', '-')
            asset_type = TYPE_LABELS.get(quote_type, quote_type if quote_type else '-')
            if quote_type == 'EQUITY':
                sector = info.get('sector', '-') or '-'
            else:
                sector = info.get('category') or info.get('fundFamily') or '-'

        rows.append({'Ticker': t, 'Name': name, 'Type': asset_type, 'Sector': sector})

    return pd.DataFrame(rows).set_index('Ticker')


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def compute_returns(prices_chf):
    return prices_chf.pct_change()


def compute_summary_stats(returns):
    """Annualized return/vol + daily VaR/ES 95%, one row per column."""
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    var95 = returns.quantile(0.05)
    es95 = returns[returns < var95].mean()
    out = pd.concat([ann_return, ann_vol, var95, es95], axis=1)
    out.columns = ['Annualized Return', 'Annualized Volatility', 'Daily VaR 95%', 'Daily Expected Shortfall 95%']
    return out


# ---------------------------------------------------------------------
# Portfolio construction (in-sample)
# ---------------------------------------------------------------------

def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, risk


def optimize_max_sharpe(returns, risk_free_daily):
    """Excess-return Sharpe maximization (daily), same method as the notebook."""
    excess_returns = returns.iloc[1:].sub(risk_free_daily['RF'].iloc[1:], axis=0)
    mean_excess = excess_returns.mean()
    cov = excess_returns.cov()
    n = len(mean_excess)

    def neg_sharpe(w):
        r, s = portfolio_performance(w, mean_excess.values, cov.values)
        return -r / s

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.array([1. / n] * n)
    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=returns.columns)


def optimize_min_variance(returns):
    n = returns.shape[1]
    cov = returns.cov()

    def vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov, w)) * 252)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    x0 = np.array([1. / n] * n)
    result = minimize(vol, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=returns.columns)


def equal_weights(returns):
    n = returns.shape[1]
    return pd.Series([1. / n] * n, index=returns.columns)


def build_benchmark_composite(weights_series, currency_map, start_date, end_date):
    """Composite benchmark: liquid ETFs weighted by the Mean-Variance
    portfolio's actual per-security exposure (see notebook Part 3)."""
    bench_weights, bench_currency = {}, {}
    for ticker, w in weights_series.items():
        if ticker in BENCHMARK_OVERRIDES:
            bt, ccy = BENCHMARK_OVERRIDES[ticker]
        else:
            ccy = currency_map.get(ticker)
            bt = BENCHMARK_BY_CURRENCY.get(ccy)
        if bt is None:
            continue
        bench_weights[bt] = bench_weights.get(bt, 0.0) + w
        bench_currency[bt] = ccy

    bench_weights = pd.Series(bench_weights).sort_values(ascending=False)
    bench_tickers = list(bench_weights.index)

    bench_prices = yf.download(bench_tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    if isinstance(bench_prices, pd.Series):
        bench_prices = bench_prices.to_frame(name=bench_tickers[0])

    fx_data = load_fx_data(bench_currency.values(), start_date, end_date)
    bench_prices_chf = bench_prices.copy()
    for t in bench_tickers:
        ccy = bench_currency[t]
        if ccy == 'CHF':
            continue
        fx_pair = ccy + 'CHF=X'
        if fx_pair in fx_data.columns:
            bench_prices_chf[t] = bench_prices[t] * fx_data[fx_pair].reindex(bench_prices.index)

    bench_returns = bench_prices_chf.pct_change()
    composite = pd.Series(0.0, index=bench_returns.index)
    total_w = bench_weights.sum()
    for t, w in bench_weights.items():
        if t in bench_returns.columns:
            composite = composite.add(bench_returns[t].fillna(0) * w, fill_value=0)
    if total_w > 0:
        composite = composite / total_w
    composite.name = 'Benchmark (composite)'
    return composite, bench_weights


def load_msci_world(start_date, end_date, fx_data):
    prices = _to_series(yf.download('URTH', start=start_date, end=end_date, auto_adjust=True)['Close'])
    prices_chf = prices * fx_data['USDCHF=X'].reindex(prices.index)
    return prices_chf.pct_change()


# ---------------------------------------------------------------------
# Out-of-sample rolling window portfolios (invest at rebalance, then drift)
# ---------------------------------------------------------------------

def _drift_walk(returns, risk_free_daily, window_size, rebalance_freq, mode):
    """Shared engine for the three rolling-window strategies.
    mode='mean_variance' re-optimizes (max Sharpe) at each rebalance date;
    mode='min_variance' re-optimizes (min variance) at each rebalance date;
    mode='equal_weight' simply resets to 1/N at each rebalance date.
    Between rebalances, weights drift with each asset's daily return
    (buy-and-hold, self-financed) - no daily rebalancing.
    risk_free_daily is only used (and required) when mode='mean_variance'.
    """
    n_obs = len(returns)
    tickers = returns.columns
    rolling_weights = pd.DataFrame(index=returns.index, columns=tickers, dtype=float)
    oos_returns = pd.Series(index=returns.index, dtype=float)

    if n_obs <= window_size:
        return rolling_weights, pd.Series(dtype=float), []

    n_assets = len(tickers)
    current_weights = np.array([1. / n_assets] * n_assets)
    rebalance_dates = list(range(window_size, n_obs, rebalance_freq))

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))

    for k, i in enumerate(rebalance_dates):
        if mode == 'mean_variance':
            window_data = returns.iloc[i - window_size:i]
            window_rf = risk_free_daily.iloc[i - window_size:i]
            excess = window_data.sub(window_rf['RF'], axis=0)
            mean_excess = excess.mean()
            cov = excess.cov()

            def neg_sharpe(w, mean_excess=mean_excess, cov=cov):
                r, s = portfolio_performance(w, mean_excess.values, cov.values)
                return -r / s

            result = minimize(neg_sharpe, current_weights, method='SLSQP',
                               bounds=bounds, constraints=constraints)
            if result.success:
                current_weights = result.x
            drifted = current_weights.copy()
        elif mode == 'min_variance':
            # No expected-return estimation needed here (unlike mean_variance
            # above) - only the covariance matrix, which is generally more
            # stable to estimate out-of-sample than expected returns.
            window_data = returns.iloc[i - window_size:i]
            cov = window_data.cov()

            def port_vol(w, cov=cov):
                return np.sqrt(np.dot(w.T, np.dot(cov, w)))

            result = minimize(port_vol, current_weights, method='SLSQP',
                               bounds=bounds, constraints=constraints)
            if result.success:
                current_weights = result.x
            drifted = current_weights.copy()
        else:  # equal_weight
            drifted = np.array([1. / n_assets] * n_assets)

        next_i = rebalance_dates[k + 1] if k + 1 < len(rebalance_dates) else n_obs
        for day in range(i, next_i):
            day_returns = returns.iloc[day].values
            oos_returns.iloc[day] = np.dot(drifted, day_returns)
            rolling_weights.iloc[day] = drifted
            grown = drifted * (1 + day_returns)
            total_grown = grown.sum()
            if total_grown > 0:
                drifted = grown / total_grown

    return rolling_weights, oos_returns.dropna(), rebalance_dates


def rolling_mean_variance(returns, risk_free_daily, window_size=252, rebalance_freq=21):
    return _drift_walk(returns, risk_free_daily, window_size, rebalance_freq, mode='mean_variance')


def rolling_equal_weight(returns, risk_free_daily, window_size=252, rebalance_freq=21):
    return _drift_walk(returns, risk_free_daily, window_size, rebalance_freq, mode='equal_weight')


def rolling_min_variance(returns, window_size=252, rebalance_freq=21):
    """Minimum-variance rolling window: at each rebalance date, re-optimize
    to minimize portfolio variance over the trailing window (no expected-
    return estimation needed), then let weights drift until the next
    rebalance - same drift mechanism as the other two rolling strategies."""
    return _drift_walk(returns, None, window_size, rebalance_freq, mode='min_variance')


def rebase_to_zero(returns_df):
    """Prepend a synthetic day-zero baseline (=1.0, i.e. 0%) one calendar day
    before the first observation, so every column starts at the same point
    when plotted as a cumulative-performance chart."""
    cumulative = (1 + returns_df).cumprod()
    if cumulative.empty:
        return cumulative
    baseline_date = cumulative.index[0] - pd.Timedelta(days=1)
    baseline_row = pd.DataFrame([[1.0] * cumulative.shape[1]], columns=cumulative.columns, index=[baseline_date])
    return pd.concat([baseline_row, cumulative]).ffill()


# ---------------------------------------------------------------------
# Parameter optimization: window_size x rebalance_freq x time_frame grid
# ---------------------------------------------------------------------

def run_rolling_strategy(returns, risk_free_daily, strategy, window_size, rebalance_freq):
    """Dispatch to the matching rolling-window strategy and return only the
    realized out-of-sample daily return series."""
    if strategy == "Mean-Variance":
        _, oos, _ = rolling_mean_variance(returns, risk_free_daily, window_size, rebalance_freq)
    elif strategy == "Min-Variance":
        _, oos, _ = rolling_min_variance(returns, window_size, rebalance_freq)
    elif strategy == "Equally-Weighted":
        _, oos, _ = rolling_equal_weight(returns, risk_free_daily, window_size, rebalance_freq)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return oos


def evaluate_oos(oos_returns, risk_free_rate, benchmark_returns=None):
    """Annualized return/vol/Sharpe of a realized out-of-sample return series.
    If `benchmark_returns` is given, also computes the benchmark's annualized
    return over the EXACT SAME dates as `oos_returns` (not the full time
    frame slice, since the strategy has no realized return before its first
    rebalance), and the resulting excess return / outperformance flag."""
    if oos_returns is None or oos_returns.empty:
        metrics = {"Annualized Return": np.nan, "Annualized Volatility": np.nan,
                   "Sharpe Ratio": np.nan, "n_obs": 0}
        if benchmark_returns is not None:
            metrics.update({"Benchmark Annualized Return": np.nan,
                             "Excess Return vs Benchmark": np.nan, "Outperforms": False})
        return metrics

    ann_return = oos_returns.mean() * 252
    ann_vol = oos_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    metrics = {"Annualized Return": ann_return, "Annualized Volatility": ann_vol,
               "Sharpe Ratio": sharpe, "n_obs": len(oos_returns)}

    if benchmark_returns is not None:
        common_idx = oos_returns.index.intersection(benchmark_returns.dropna().index)
        if len(common_idx) > 5:
            bench_ann_return = benchmark_returns.loc[common_idx].mean() * 252
            excess = ann_return - bench_ann_return
            metrics["Benchmark Annualized Return"] = bench_ann_return
            metrics["Excess Return vs Benchmark"] = excess
            metrics["Outperforms"] = bool(excess > 0)
        else:
            metrics["Benchmark Annualized Return"] = np.nan
            metrics["Excess Return vs Benchmark"] = np.nan
            metrics["Outperforms"] = False

    return metrics


def generate_period_slices(returns, risk_free_daily, time_frame_days_list):
    """For each requested time-frame LENGTH (in trading days), chop the full
    `returns` history into several consecutive, NON-OVERLAPPING periods of
    that length, covering the whole sample from its start forward - instead
    of just the single most recent period. This is what lets "3 months", for
    example, be tested across every distinct 3-month period available rather
    than only the latest one, for a genuine robustness check.

    A time frame equal to (or longer than) the full sample naturally yields
    a single period covering everything.

    Returns a list of dicts, one per period, each with: time_frame_days,
    period_index (0-based, in chronological order), n_periods (how many
    periods this time-frame length produced in total), returns_slice,
    risk_free_slice, period_start, period_end.
    """
    n_total = len(returns)
    slices = []
    for tf_days in sorted(set(int(min(d, n_total)) for d in time_frame_days_list if d > 0)):
        n_periods = max(n_total // tf_days, 1)
        for k in range(n_periods):
            start, end = k * tf_days, k * tf_days + tf_days
            r_slice = returns.iloc[start:end]
            rf_slice = risk_free_daily.iloc[start:end]
            slices.append({
                "time_frame_days": tf_days,
                "period_index": k,
                "n_periods": n_periods,
                "returns_slice": r_slice,
                "risk_free_slice": rf_slice,
                "period_start": r_slice.index[0],
                "period_end": r_slice.index[-1],
            })
    return slices


def optimize_grid(returns, risk_free_daily, risk_free_rate, strategy,
                   window_sizes, rebalance_freqs, time_frame_days_list,
                   benchmark_returns=None, progress_callback=None):
    """Grid search over (window_size, rebalance_freq) for a given
    rolling-window strategy, tested across every non-overlapping period that
    fits within the already-loaded `returns` / `risk_free_daily` history for
    each requested time-frame length (see `generate_period_slices`) - this
    does not fetch any additional price history beyond what the caller
    already has.

    If `benchmark_returns` is provided (e.g. MSCI World daily returns), each
    row also gets the benchmark's annualized return over the same realized
    out-of-sample dates, the resulting excess return, and an "Outperforms"
    boolean flag.

    Returns a long-format DataFrame, one row per (window_size, rebalance_freq,
    period) combination, with columns: window_size, rebalance_freq,
    time_frame_days, period_index, n_periods, period_start, period_end,
    Annualized Return, Annualized Volatility, Sharpe Ratio, n_obs[,
    Benchmark Annualized Return, Excess Return vs Benchmark, Outperforms].
    """
    period_slices = generate_period_slices(returns, risk_free_daily, time_frame_days_list)

    rows = []
    total = max(len(window_sizes) * len(rebalance_freqs) * len(period_slices), 1)
    done = 0

    for period in period_slices:
        returns_slice = period["returns_slice"]
        rf_slice = period["risk_free_slice"]

        for ws in window_sizes:
            for rb in rebalance_freqs:
                if ws >= len(returns_slice):
                    metrics = {"Annualized Return": np.nan, "Annualized Volatility": np.nan,
                               "Sharpe Ratio": np.nan, "n_obs": 0}
                    if benchmark_returns is not None:
                        metrics.update({"Benchmark Annualized Return": np.nan,
                                         "Excess Return vs Benchmark": np.nan, "Outperforms": False})
                else:
                    oos = run_rolling_strategy(returns_slice, rf_slice, strategy, ws, rb)
                    metrics = evaluate_oos(oos, risk_free_rate, benchmark_returns=benchmark_returns)
                rows.append({
                    "window_size": ws, "rebalance_freq": rb,
                    "time_frame_days": period["time_frame_days"],
                    "period_index": period["period_index"],
                    "n_periods": period["n_periods"],
                    "period_start": period["period_start"],
                    "period_end": period["period_end"],
                    **metrics,
                })
                done += 1
                if progress_callback:
                    progress_callback(done / total)

    return pd.DataFrame(rows)


def compute_robustness(grid):
    """Aggregate a long-format optimize_grid() result across time frames, for
    each (window_size, rebalance_freq) combination, to assess how
    consistently it beats the benchmark across the different time frames
    tested - i.e. a robustness check rather than a single-period result.

    Requires `grid` to have been produced with a `benchmark_returns` argument
    (so it has 'Outperforms' / 'Excess Return vs Benchmark' columns).

    Returns one row per (window_size, rebalance_freq) with:
    - n_timeframes_tested / n_timeframes_outperform / pct_outperform
    - mean / worst / best excess return across the tested time frames
    - mean_sharpe
    - fully_robust: True if it outperforms the benchmark in EVERY time frame
      tested (equivalent to worst_excess_return > 0)
    """
    cols = ["window_size", "rebalance_freq", "n_timeframes_tested", "n_timeframes_outperform",
            "pct_outperform", "mean_excess_return", "worst_excess_return", "best_excess_return",
            "mean_sharpe", "fully_robust"]
    if "Outperforms" not in grid.columns:
        raise ValueError("grid must be produced with optimize_grid(..., benchmark_returns=...) "
                          "to compute robustness (missing 'Outperforms' column).")

    valid = grid.dropna(subset=["Excess Return vs Benchmark"])
    if valid.empty:
        return pd.DataFrame(columns=cols)

    grouped = valid.groupby(["window_size", "rebalance_freq"])
    out = grouped.agg(
        n_timeframes_tested=("Outperforms", "size"),
        n_timeframes_outperform=("Outperforms", "sum"),
        mean_excess_return=("Excess Return vs Benchmark", "mean"),
        worst_excess_return=("Excess Return vs Benchmark", "min"),
        best_excess_return=("Excess Return vs Benchmark", "max"),
        mean_sharpe=("Sharpe Ratio", "mean"),
    ).reset_index()
    out["pct_outperform"] = out["n_timeframes_outperform"] / out["n_timeframes_tested"]
    out["fully_robust"] = out["n_timeframes_outperform"] == out["n_timeframes_tested"]
    return out.sort_values(["pct_outperform", "mean_excess_return"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# Client risk profiling (suitability questionnaire)
# ---------------------------------------------------------------------

# Each question is worth 0-3 points. 10 questions -> max score 30.
# Themes, in order: Investment Horizon, Capital to Invest, Product Knowledge,
# Family Situation, Financial Situation.
QUESTIONNAIRE = [
    {
        "theme": "Investment Horizon",
        "question": "What is your investment horizon for this capital?",
        "options": [
            ("Less than 2 years", 0),
            ("2 to 5 years", 1),
            ("5 to 10 years", 2),
            ("More than 10 years", 3),
        ],
    },
    {
        "theme": "Investment Horizon",
        "question": "How likely is it that you would need to access this capital earlier than planned?",
        "options": [
            ("Very likely", 0),
            ("Possible but unlikely", 1),
            ("Unlikely", 2),
            ("Very unlikely", 3),
        ],
    },
    {
        "theme": "Capital to Invest",
        "question": "What share of your total net worth does this investment amount represent?",
        "options": [
            ("More than 75%", 0),
            ("50% to 75%", 1),
            ("25% to 50%", 2),
            ("Less than 25%", 3),
        ],
    },
    {
        "theme": "Capital to Invest",
        "question": "Do you expect to make significant additional contributions or withdrawals in the "
                    "next few years?",
        "options": [
            ("Large withdrawals expected", 0),
            ("Occasional withdrawals possible", 1),
            ("Neither expected", 2),
            ("Regular additional contributions expected", 3),
        ],
    },
    {
        "theme": "Product Knowledge",
        "question": "How would you describe your knowledge of financial products?",
        "options": [
            ("Beginner (never invested before)", 0),
            ("Basic (savings accounts, bonds)", 1),
            ("Intermediate (equities, mutual funds, ETFs)", 2),
            ("Advanced (derivatives, structured products, alternatives)", 3),
        ],
    },
    {
        "theme": "Product Knowledge",
        "question": "Have you previously invested in more complex products (derivatives, structured "
                    "products, private equity)?",
        "options": [
            ("Never", 0),
            ("Once or twice", 1),
            ("Occasionally", 2),
            ("Regularly", 3),
        ],
    },
    {
        "theme": "Family Situation",
        "question": "Do you have dependents whose financial needs you must provide for (children, "
                    "parents, etc.)?",
        "options": [
            ("Yes, several", 0),
            ("Yes, one", 1),
            ("No, but expected in the medium term", 2),
            ("No", 3),
        ],
    },
    {
        "theme": "Family Situation",
        "question": "Is a major change in your family situation expected in the coming years (marriage, "
                    "divorce, inheritance, etc.)?",
        "options": [
            ("Major change expected", 0),
            ("Possible change", 1),
            ("Stable situation", 3),
        ],
    },
    {
        "theme": "Financial Situation",
        "question": "How would you describe the stability of your income relative to your expenses?",
        "options": [
            ("Irregular income, close to or below expenses", 0),
            ("Stable income, slightly above expenses", 1),
            ("Stable and comfortable income", 2),
            ("High income, well above expenses", 3),
        ],
    },
    {
        "theme": "Financial Situation",
        "question": "If your portfolio lost 20% of its value in one year, what would you do?",
        "options": [
            ("Sell everything immediately to avoid further losses", 0),
            ("Sell part of the portfolio", 1),
            ("Do nothing and wait for a recovery", 2),
            ("Invest more to take advantage of lower prices", 3),
        ],
    },
]

QUESTIONNAIRE_MAX_SCORE = sum(max(pts for _, pts in q["options"]) for q in QUESTIONNAIRE)

RISK_CATEGORIES = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"]

# (weight on Min-Variance portfolio, weight on Mean-Variance portfolio)
RISK_BLEND = {
    "Very Low Risk": (1.00, 0.00),
    "Low Risk": (0.80, 0.20),
    "Medium Risk": (0.50, 0.50),
    "High Risk": (0.20, 0.80),
    "Very High Risk": (0.00, 1.00),
}


def score_to_risk_category(score, max_score=QUESTIONNAIRE_MAX_SCORE):
    """Map a total questionnaire score to one of the 5 risk categories,
    splitting the [0, max_score] range into 5 equal buckets."""
    fraction = score / max_score if max_score else 0
    if fraction <= 0.2:
        return "Very Low Risk"
    elif fraction <= 0.4:
        return "Low Risk"
    elif fraction <= 0.6:
        return "Medium Risk"
    elif fraction <= 0.8:
        return "High Risk"
    else:
        return "Very High Risk"


def blend_weights(min_var_weights, mean_var_weights, risk_category):
    """Combine the two strategies' latest out-of-sample weights according to
    the target risk category's blend ratio. Both inputs must be indexed by
    the same tickers."""
    w_minvar, w_meanvar = RISK_BLEND[risk_category]
    combined = w_minvar * min_var_weights + w_meanvar * mean_var_weights.reindex(min_var_weights.index).fillna(0)
    return combined
