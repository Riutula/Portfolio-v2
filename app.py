"""
app.py - Portfolio dashboard

Interactive Streamlit front-end for the portfolio analysis originally built
in New_portfolio_2026_EN.ipynb. All computation lives in portfolio_lib.py;
this file is purely presentation (widgets, charts, layout).

Run locally with:
    streamlit run app.py
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import portfolio_lib as lib

st.set_page_config(page_title="Portfolio Dashboard", layout="wide", page_icon="📊")

# ---------------------------------------------------------------------
# Single source of truth for the "default" robustness search grid, used both
# as the Optimization tab's default slider values AND by the Client tab's
# automatic robust-parameter search (cached_best_params below). Keeping
# these in one place guarantees the two tabs agree on what "most robust"
# means whenever the Optimization tab is left at its defaults - previously
# they used two different hardcoded grids and could silently disagree.
DEFAULT_WINDOW_RANGE = (60, 252)
DEFAULT_N_WINDOW_POINTS = 10
DEFAULT_REBALANCE_RANGE = (5, 126)
DEFAULT_N_REBALANCE_POINTS = 5

# ---------------------------------------------------------------------
# Cached data-loading wrappers
# ---------------------------------------------------------------------
# ttl=3600: refetch at most once per hour per unique parameter combination,
# so navigating between tabs / widgets doesn't re-hit Yahoo Finance / FRED
# every time.

@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def cached_prices(tickers, currency_map_items, start_date, end_date):
    # currency_map_items is a tuple of (ticker, currency) pairs - Streamlit's
    # cache needs hashable arguments, so the dict is rebuilt here.
    currency_map = dict(currency_map_items)
    fx = lib.load_fx_data(tuple(currency_map.values()), start_date, end_date)
    prices = lib.load_prices_chf(list(tickers), currency_map, start_date, end_date, fx_data=fx)
    return prices, fx


@st.cache_data(ttl=3600, show_spinner="Fetching the risk-free rate...")
def cached_risk_free(start_date, end_date, index):
    return lib.load_risk_free_rate(start_date, end_date, index)


@st.cache_data(ttl=3600, show_spinner="Fetching security names and sectors...")
def cached_asset_info(tickers):
    return lib.get_asset_info(tickers)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_detect_currencies(tickers):
    """Best-effort currency detection for tickers the user typed in manually
    (not part of DEFAULT_CURRENCY_MAP)."""
    return {t: lib.detect_currency(t) for t in tickers}


@st.cache_data(ttl=3600, show_spinner="Building the composite benchmark...")
def cached_benchmark(weights_series, currency_map, start_date, end_date):
    return lib.build_benchmark_composite(weights_series, currency_map, start_date, end_date)


@st.cache_data(ttl=3600, show_spinner="Fetching MSCI World...")
def cached_msci(start_date, end_date, fx_data):
    return lib.load_msci_world(start_date, end_date, fx_data)


@st.cache_data(ttl=3600, show_spinner="Running the rolling-window backtests...")
def cached_rolling(returns, risk_free_daily, window_size, rebalance_freq):
    rw_mv, oos_mv, _ = lib.rolling_mean_variance(returns, risk_free_daily, window_size, rebalance_freq)
    rw_eq, oos_eq, _ = lib.rolling_equal_weight(returns, risk_free_daily, window_size, rebalance_freq)
    rw_minvar, oos_minvar, _ = lib.rolling_min_variance(returns, window_size, rebalance_freq)
    return rw_mv, oos_mv, rw_eq, oos_eq, rw_minvar, oos_minvar


@st.cache_data(ttl=3600, show_spinner=False)
def cached_best_params(returns, risk_free_daily, risk_free_rate, msci_returns, strategy):
    """Find the most robust (window_size, rebalance_freq) for `strategy` by
    running the same grid-search + robustness methodology as the
    Optimization tab, using the SAME default grid as that tab's sliders
    (see DEFAULT_* constants above) - so this always agrees with what the
    Optimization tab would show for this strategy at its default settings.
    Cached so this only runs once per data pull, not on every questionnaire
    interaction."""
    max_days = len(returns)
    window_sizes = sorted(set(np.linspace(*DEFAULT_WINDOW_RANGE, DEFAULT_N_WINDOW_POINTS).astype(int)))
    rebalance_freqs = sorted(set(np.linspace(*DEFAULT_REBALANCE_RANGE, DEFAULT_N_REBALANCE_POINTS).astype(int)))
    time_frame_days_list = sorted(set(d for d in [126, 252, max_days] if d <= max_days))
    if not time_frame_days_list:
        time_frame_days_list = [max_days]

    grid = lib.optimize_grid(returns, risk_free_daily, risk_free_rate, strategy,
                              window_sizes, rebalance_freqs, time_frame_days_list,
                              benchmark_returns=msci_returns)
    robustness = lib.compute_robustness(grid)
    if robustness.empty:
        return None
    best = robustness.iloc[0]  # already sorted: highest pct_outperform, then mean_excess_return
    return int(best["window_size"]), int(best["rebalance_freq"])


def pct(x, d=2):
    try:
        return f"{x * 100:.{d}f}%"
    except (TypeError, ValueError):
        return "-"


# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
st.sidebar.title("Settings")

tickers = st.sidebar.multiselect(
    "Universe", options=lib.DEFAULT_TICKERS, default=lib.DEFAULT_TICKERS,
    accept_new_options=True,
    help="Remove a ticker to exclude it, or type any other Yahoo Finance ticker to add it "
         "(e.g. AAPL, VOD.L, 7203.T).",
)

_custom_tickers = tuple(sorted(t for t in tickers if t not in lib.DEFAULT_CURRENCY_MAP))
if _custom_tickers:
    with st.sidebar.spinner(f"Detecting currency for {', '.join(_custom_tickers)}..."):
        _detected_currencies = cached_detect_currencies(_custom_tickers)
else:
    _detected_currencies = {}

currency_map = {
    t: lib.DEFAULT_CURRENCY_MAP.get(t, _detected_currencies.get(t, "USD"))
    for t in tickers
}

today = datetime.today()
MIN_DATE = datetime(2025, 6, 27)
default_start = max(MIN_DATE, today - timedelta(days=420))
date_range = st.sidebar.date_input(
    "Price history", value=(default_start, today),
    min_value=MIN_DATE, max_value=today,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = default_start, today

st.sidebar.markdown("---")
run_button = st.sidebar.button("Run / refresh analysis", type="primary", use_container_width=True)

if not tickers:
    st.warning("Select at least one ticker in the sidebar to begin.")
    st.stop()

if "has_run" not in st.session_state:
    st.session_state.has_run = False
if run_button:
    st.session_state.has_run = True
if not st.session_state.has_run:
    st.title("📊 Portfolio Dashboard")
    st.info("Set your universe and date range in the sidebar, then click **Run / refresh analysis**.")
    st.stop()

# ---------------------------------------------------------------------
# Core pipeline (cached)
# ---------------------------------------------------------------------
try:
    prices, fx_data = cached_prices(tuple(tickers), tuple(sorted(currency_map.items())), start_date, end_date)
except Exception as e:
    st.error(f"Could not download price data: {e}")
    st.stop()

_empty_tickers = [t for t in prices.columns if prices[t].dropna().empty]
if _empty_tickers:
    st.warning(
        f"No price data found for: {', '.join(_empty_tickers)}. Check the ticker spelling on "
        f"Yahoo Finance (finance.yahoo.com) - they were kept in the universe but will show as "
        f"blank/zero everywhere below."
    )

returns = lib.compute_returns(prices)
summary_stats = lib.compute_summary_stats(returns)
risk_free_daily, risk_free_rate = cached_risk_free(start_date, end_date, tuple(prices.index))
asset_info = cached_asset_info(tuple(tickers))

st.title("📊 Portfolio Dashboard")
st.caption(f"{len(tickers)} assets • base currency CHF • {start_date} → {end_date}")

tab_assets, tab_construction, tab_oos, tab_optim, tab_client, tab_data = st.tabs(
    ["Assets", "Portfolio Construction", "Out-of-Sample", "Optimization", "Client", "Raw Data"]
)

# ---------------------------------------------------------------------
# Tab 1: Assets
# ---------------------------------------------------------------------
with tab_assets:
    st.subheader("Selected assets")
    display_info = asset_info.copy()
    display_info.insert(1, "Currency", [currency_map.get(t, "-") for t in display_info.index])
    st.dataframe(display_info, use_container_width=True)

    st.subheader("Price evolution (CHF)")
    fig = px.line(prices, title=None)
    fig.update_layout(yaxis_title="Price (CHF)", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative performance")
    cum = (1 + returns.fillna(0)).cumprod()
    fig = px.line(cum, title=None)
    fig.update_layout(yaxis_tickformat=",.0%", yaxis_title="Cumulative performance", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistics")
    fmt_stats = summary_stats.copy()
    for c in fmt_stats.columns:
        fmt_stats[c] = fmt_stats[c].map(pct)
    st.dataframe(fmt_stats, use_container_width=True)

# ---------------------------------------------------------------------
# Tab 2: Portfolio Construction (in-sample)
# ---------------------------------------------------------------------
with tab_construction:
    st.subheader("Optimized portfolios (in-sample)")

    w_mv = lib.optimize_max_sharpe(returns, risk_free_daily)
    w_minvar = lib.optimize_min_variance(returns)
    w_eq = lib.equal_weights(returns)

    weights_df = pd.DataFrame({
        "Mean-Variance": w_mv, "Min-Variance": w_minvar, "Equally-Weighted": w_eq,
    })
    weights_df.index = [asset_info.loc[t, "Name"] if t in asset_info.index else t for t in weights_df.index]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Portfolio weights**")
        st.dataframe(weights_df.style.format("{:.1%}"), use_container_width=True)
    with col2:
        fig = px.bar(weights_df, barmode="group")
        fig.update_layout(yaxis_tickformat=",.0%", yaxis_title="Weight", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    port_returns = pd.DataFrame({
        "Mean-Variance": (returns * w_mv.values).sum(axis=1),
        "Min-Variance": (returns * w_minvar.values).sum(axis=1),
        "Equally-Weighted": (returns * w_eq.values).sum(axis=1),
    })

    with st.spinner("Building the composite benchmark..."):
        benchmark_composite, bench_weights = cached_benchmark(w_mv, currency_map, start_date, end_date)
    port_returns["Benchmark (composite)"] = benchmark_composite.reindex(port_returns.index).fillna(0)

    st.subheader("Performance vs. composite benchmark")
    cum_port = (1 + port_returns.fillna(0)).cumprod()
    fig = px.line(cum_port)
    fig.update_layout(yaxis_tickformat=",.0%", yaxis_title="Cumulative performance", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    port_stats = lib.compute_summary_stats(port_returns)
    sharpe = (port_stats["Annualized Return"] - risk_free_rate) / port_stats["Annualized Volatility"]
    port_stats["Sharpe Ratio"] = sharpe

    fmt_port_stats = port_stats.copy()
    for c in ["Annualized Return", "Annualized Volatility", "Daily VaR 95%", "Daily Expected Shortfall 95%"]:
        fmt_port_stats[c] = fmt_port_stats[c].map(pct)
    fmt_port_stats["Sharpe Ratio"] = fmt_port_stats["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    st.dataframe(fmt_port_stats, use_container_width=True)

    with st.expander("Composite benchmark composition"):
        st.dataframe(bench_weights.map(lambda x: f"{x:.1%}").rename("Weight"), use_container_width=True)

# ---------------------------------------------------------------------
# Tab 3: Out-of-sample rolling window
# ---------------------------------------------------------------------
with tab_oos:
    st.subheader("Out-of-sample rolling window")

    c1, c2 = st.columns(2)
    with c1:
        window_size = st.slider("Lookback window (trading days)", 60, 252, 252, step=1,
                                 help="History used to re-optimize the Mean-Variance portfolio at each rebalance.")
    with c2:
        rebalance_freq = st.slider("Rebalance frequency (trading days)", 5, 252, 21, step=1,
                                    help="21 trading days ≈ 1 month. 252 ≈ 1 year (i.e. no rebalancing within the period).")
    st.caption(f"Current setting: lookback = {window_size}d, rebalance every {rebalance_freq}d.")

    rw_mv, oos_mv, rw_eq, oos_eq, rw_minvar, oos_minvar = cached_rolling(returns, risk_free_daily, window_size, rebalance_freq)

    if oos_mv.empty and oos_eq.empty and oos_minvar.empty:
        st.warning(
            f"Not enough history yet: {len(returns)} observations available, "
            f"but the lookback window requires at least {window_size}. "
            f"Widen the date range or shorten the lookback window in the sidebar."
        )
    else:
        st.caption(
            "⚠️ Data caveat: with a limited price history, these rolling-window portfolios have only "
            "gone through a handful of rebalances so far. Treat this as an early, directional check of "
            "the mechanism rather than a statistically meaningful track record."
        )

        with st.spinner("Fetching MSCI World..."):
            msci_returns = cached_msci(start_date, end_date, fx_data)

        comp = pd.DataFrame({
            "Mean-Variance (rolling)": oos_mv,
            "Equally-Weighted (rolling)": oos_eq,
            "Min-Variance (rolling)": oos_minvar,
            "MSCI World": msci_returns.reindex(oos_mv.index if not oos_mv.empty else oos_eq.index),
        })
        cum_oos = lib.rebase_to_zero(comp)

        st.markdown("**Out-of-sample portfolios vs MSCI World** (rebased to 0% at inception)")
        fig = px.line(cum_oos)
        fig.update_layout(yaxis_tickformat=",.0%", yaxis_title="Cumulative performance", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Evolution of out-of-sample weights**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("Mean-Variance (rolling)")
            w = rw_mv.dropna().rename(columns=lambda t: asset_info.loc[t, "Name"] if t in asset_info.index else t)
            if not w.empty:
                fig = px.line(w)
                fig.update_layout(yaxis_tickformat=",.0%", yaxis_title="Weight", legend_title="",
                                   height=350, legend=dict(font=dict(size=8)))
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.caption("Equally-Weighted (rolling)")
            w = rw_eq.dropna().rename(columns=lambda t: asset_info.loc[t, "Name"] if t in asset_info.index else t)
            if not w.empty:
                fig = px.line(w)
                fig.update_layout(yaxis_tickformat=",.0%", yaxis_title="Weight", legend_title="",
                                   height=350, legend=dict(font=dict(size=8)))
                st.plotly_chart(fig, use_container_width=True)
        with col3:
            st.caption("Min-Variance (rolling)")
            w = rw_minvar.dropna().rename(columns=lambda t: asset_info.loc[t, "Name"] if t in asset_info.index else t)
            if not w.empty:
                fig = px.line(w)
                fig.update_layout(yaxis_tickformat=",.0%", yaxis_title="Weight", legend_title="",
                                   height=350, legend=dict(font=dict(size=8)))
                st.plotly_chart(fig, use_container_width=True)

        if not rw_mv.dropna().empty:
            st.markdown("**Latest weights breakdown - Mean-Variance (rolling)**")
            latest = rw_mv.dropna().iloc[-1].sort_values(ascending=False)
            latest.index = [asset_info.loc[t, "Name"] if t in asset_info.index else t for t in latest.index]

            c1, c2 = st.columns(2)
            with c1:
                sector_w = {}
                for t, w in rw_mv.dropna().iloc[-1].items():
                    s = asset_info.loc[t, "Sector"] if t in asset_info.index else "Not classified"
                    s = s if s and s != "-" else "Not classified"
                    sector_w[s] = sector_w.get(s, 0.0) + w
                fig = px.pie(values=list(sector_w.values()), names=list(sector_w.keys()), title="By sector")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                ccy_w = {}
                for t, w in rw_mv.dropna().iloc[-1].items():
                    c = currency_map.get(t, "Not classified")
                    ccy_w[c] = ccy_w.get(c, 0.0) + w
                fig = px.pie(values=list(ccy_w.values()), names=list(ccy_w.keys()), title="By currency")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Tab 4: Parameter optimization (window size x rebalance freq x time frame)
# ---------------------------------------------------------------------
with tab_optim:
    st.subheader("Optimize lookback window & rebalance frequency")
    st.caption(
        "Grid search over the rolling-window strategy's two parameters, repeated across several "
        "time frames sliced from the price history already loaded in the sidebar (no additional "
        "download). Each point in the 3D chart below is one backtest, colored by how much it "
        "outperformed (green) or underperformed (red) MSCI World over the exact same realized dates. "
        "Below the chart, a robustness view checks which (window, rebalance) combinations beat MSCI "
        "World consistently across time frames, rather than in just one."
    )

    strategy = st.selectbox("Strategy to optimize", ["Mean-Variance", "Min-Variance", "Equally-Weighted"])

    c1, c2 = st.columns(2)
    with c1:
        window_range = st.slider("Window size range (trading days)", 60, 252, DEFAULT_WINDOW_RANGE, step=1)
        n_window_points = st.slider("Number of window sizes to test", 2, 20, DEFAULT_N_WINDOW_POINTS)
    with c2:
        rebalance_range = st.slider("Rebalance frequency range (trading days)", 5, 252, DEFAULT_REBALANCE_RANGE, step=1)
        n_rebalance_points = st.slider("Number of rebalance frequencies to test", 2, 10, DEFAULT_N_REBALANCE_POINTS)

    max_days_available = len(returns)
    time_frame_options = {
        "3 months": 63, "6 months": 126, "9 months": 189,
        "12 months": 252, "Full period loaded": max_days_available,
    }
    time_frame_options = {k: v for k, v in time_frame_options.items() if v <= max_days_available or k == "Full period loaded"}
    selected_time_frames = st.multiselect(
        "Time frames to test",
        options=list(time_frame_options.keys()),
        default=[k for k in ["6 months", "12 months", "Full period loaded"] if k in time_frame_options],
        help="Each length is tested across every non-overlapping period of that length available in the "
             "loaded price history (e.g. '3 months' tests every distinct 3-month period, not just the "
             "latest one) - this is what makes the robustness check below meaningful.",
    )

    window_sizes = sorted(set(np.linspace(window_range[0], window_range[1], n_window_points).astype(int)))
    rebalance_freqs = sorted(set(np.linspace(rebalance_range[0], rebalance_range[1], n_rebalance_points).astype(int)))
    time_frame_days = sorted(set(time_frame_options[k] for k in selected_time_frames))

    if time_frame_days:
        period_preview = lib.generate_period_slices(returns, risk_free_daily, time_frame_days)
        periods_per_tf = {}
        for p in period_preview:
            periods_per_tf[p["time_frame_days"]] = p["n_periods"]
        label_by_days = {v: k for k, v in time_frame_options.items()}
        preview_lines = ", ".join(
            f"{label_by_days.get(d, f'{d}d')}: {n} non-overlapping period(s)"
            for d, n in sorted(periods_per_tf.items())
        )
        st.caption(f"Periods generated per time frame — {preview_lines}.")
        n_total_periods = len(period_preview)
    else:
        n_total_periods = 0

    n_combos = len(window_sizes) * len(rebalance_freqs) * n_total_periods
    st.caption(f"This will run {n_combos} backtests ({n_total_periods} periods × "
               f"{len(window_sizes)} window sizes × {len(rebalance_freqs)} rebalance frequencies). "
               f"Larger grids take longer.")
    if n_combos > 300:
        st.warning(
            "This is a large grid and may take a while to run. Consider reducing the number of "
            "window sizes / rebalance frequencies tested, or the number of time frames selected."
        )

    if st.button("Run optimization", type="primary"):
        if n_combos == 0:
            st.warning("Select at least one time frame above.")
        else:
            with st.spinner("Fetching MSCI World for the benchmark comparison..."):
                msci_returns_optim = cached_msci(start_date, end_date, fx_data)
            progress_bar = st.progress(0.0)
            grid = lib.optimize_grid(
                returns, risk_free_daily, risk_free_rate, strategy,
                window_sizes, rebalance_freqs, time_frame_days,
                benchmark_returns=msci_returns_optim,
                progress_callback=progress_bar.progress,
            )
            progress_bar.empty()
            st.session_state["optim_grid"] = grid
            st.session_state["optim_strategy"] = strategy

    if "optim_grid" in st.session_state:
        grid = st.session_state["optim_grid"]
        valid = grid.dropna(subset=["Excess Return vs Benchmark"])

        if valid.empty:
            st.warning(
                "No combination produced a valid result - every window size was longer than the "
                "time frame tested, or there weren't enough overlapping dates with MSCI World. "
                "Reduce the window size range or add a longer time frame."
            )
        else:
            time_frame_labels = {v: k for k, v in time_frame_options.items()}
            valid = valid.copy()
            valid["Time frame"] = valid["time_frame_days"].map(lambda d: time_frame_labels.get(d, f"{d}d"))
            valid["Period"] = valid.apply(
                lambda r: f"{r['Time frame']} ({r['period_index'] + 1}/{r['n_periods']}): "
                          f"{r['period_start'].strftime('%d.%m.%Y')} → {r['period_end'].strftime('%d.%m.%Y')}",
                axis=1,
            )

            only_outperform = st.checkbox("Show only combinations that outperform MSCI World", value=False)
            plot_data = valid[valid["Outperforms"]] if only_outperform else valid

            n_out = int(valid["Outperforms"].sum())
            st.caption(f"{n_out} / {len(valid)} (window, rebalance, period) combinations outperform MSCI World.")

            if plot_data.empty:
                st.warning("No combination outperforms MSCI World in this grid - try a wider parameter range.")
            else:
                st.markdown(f"**3D parameter surface — {st.session_state['optim_strategy']} vs MSCI World** "
                            "(rebalance frequency × lookback window × time frame, colored by excess annualized "
                            "return; each time-frame length shows one point per non-overlapping period tested)")
                max_abs_excess = max(abs(plot_data["Excess Return vs Benchmark"].min()),
                                      abs(plot_data["Excess Return vs Benchmark"].max()), 1e-6)
                fig3d = go.Figure(data=[go.Scatter3d(
                    x=plot_data["rebalance_freq"], y=plot_data["window_size"], z=plot_data["time_frame_days"],
                    mode="markers",
                    marker=dict(
                        size=6, color=plot_data["Excess Return vs Benchmark"], colorscale="RdYlGn",
                        cmin=-max_abs_excess, cmax=max_abs_excess,
                        colorbar=dict(title="Excess return<br>vs MSCI World"), showscale=True,
                    ),
                    text=[f"Rebalance: {r}d<br>Window: {w}d<br>{period}<br>"
                          f"Excess vs MSCI World: {ex:+.2%}<br>Strategy: {ar:.2%}<br>MSCI World: {br:.2%}<br>"
                          f"Sharpe: {s:.2f}"
                          for r, w, period, ex, ar, br, s in zip(
                              plot_data["rebalance_freq"], plot_data["window_size"], plot_data["Period"],
                              plot_data["Excess Return vs Benchmark"], plot_data["Annualized Return"],
                              plot_data["Benchmark Annualized Return"], plot_data["Sharpe Ratio"])],
                    hoverinfo="text",
                )])
                fig3d.update_layout(
                    scene=dict(
                        xaxis_title="Rebalance frequency (days)",
                        yaxis_title="Lookback window (days)",
                        zaxis_title="Time frame (trading days)",
                    ),
                    height=650, margin=dict(l=0, r=0, b=0, t=30),
                )
                st.plotly_chart(fig3d, use_container_width=True)

                best = valid.loc[valid["Excess Return vs Benchmark"].idxmax()]
                st.success(
                    f"Best single combination vs MSCI World: window = {int(best['window_size'])}d, "
                    f"rebalance = {int(best['rebalance_freq'])}d, period = {best['Period']} "
                    f"→ excess return {best['Excess Return vs Benchmark']:+.2%} "
                    f"(strategy {best['Annualized Return']:.2%} vs MSCI World "
                    f"{best['Benchmark Annualized Return']:.2%}, Sharpe {best['Sharpe Ratio']:.2f}). "
                    f"See the robustness view below for combinations that perform well consistently, "
                    f"not just in this one period."
                )

                with st.expander("Full grid results (one row per window, rebalance, and individual period)"):
                    display_grid = valid[["Period", "window_size", "rebalance_freq",
                                           "Annualized Return", "Benchmark Annualized Return",
                                           "Excess Return vs Benchmark", "Outperforms",
                                           "Annualized Volatility", "Sharpe Ratio", "n_obs"]]
                    display_grid = display_grid.rename(columns={"window_size": "Window (days)",
                                                                  "rebalance_freq": "Rebalance (days)"})
                    display_grid = display_grid.sort_values("Excess Return vs Benchmark", ascending=False)
                    st.dataframe(
                        display_grid.style.format({
                            "Annualized Return": "{:.2%}", "Benchmark Annualized Return": "{:.2%}",
                            "Excess Return vs Benchmark": "{:+.2%}", "Annualized Volatility": "{:.2%}",
                            "Sharpe Ratio": "{:.2f}",
                        }),
                        use_container_width=True,
                    )
                st.caption(
                    "⚠️ Reminder: with a limited price history, many of these backtests only cover a "
                    "handful of rebalances. Treat outperformance here as a directional signal, not a "
                    "robust result - re-running this once more price history has accumulated is "
                    "recommended before acting on it."
                )

                st.markdown("---")
                st.markdown("### Robustness across time frames")
                st.caption(
                    "For each (rolling window, rebalance frequency) pair, this collapses the time-frame "
                    "axis of the grid above into a single robustness score: in how many of the tested "
                    "time frames did this combination beat MSCI World? A combination that only wins in "
                    "one time frame is more likely to be a lucky fit than a genuinely good rule."
                )

                robustness = lib.compute_robustness(grid)
                if robustness.empty or robustness["n_timeframes_tested"].max() < 2:
                    st.info(
                        "Only one period was tested in total (e.g. you selected only 'Full period loaded'). "
                        "Select at least one time frame shorter than the full period above and re-run the "
                        "optimization to get multiple non-overlapping periods to assess robustness across."
                    )
                else:
                    heat = robustness.pivot(index="window_size", columns="rebalance_freq", values="pct_outperform")
                    fig_heat = px.imshow(
                        heat, color_continuous_scale="RdYlGn", zmin=0, zmax=1, aspect="auto",
                        labels=dict(x="Rebalance frequency (days)", y="Lookback window (days)",
                                    color="% time frames<br>outperformed"),
                        text_auto=".0%",
                    )
                    fig_heat.update_layout(height=450)
                    st.plotly_chart(fig_heat, use_container_width=True)

                    n_fully_robust = int(robustness["fully_robust"].sum())
                    st.caption(
                        f"{n_fully_robust} / {len(robustness)} combinations beat MSCI World in "
                        f"**every** time frame tested (dark green cells above, 100%)."
                    )

                    if n_fully_robust > 0:
                        best_robust = robustness[robustness["fully_robust"]].sort_values(
                            "mean_excess_return", ascending=False
                        ).iloc[0]
                        st.success(
                            f"Most robust combination: window = {int(best_robust['window_size'])}d, "
                            f"rebalance = {int(best_robust['rebalance_freq'])}d → beats MSCI World in "
                            f"all {int(best_robust['n_timeframes_tested'])} time frames tested "
                            f"(excess return: worst {best_robust['worst_excess_return']:+.2%}, "
                            f"mean {best_robust['mean_excess_return']:+.2%}, "
                            f"best {best_robust['best_excess_return']:+.2%})."
                        )
                    else:
                        st.warning(
                            "No combination beats MSCI World in every time frame tested with the current "
                            "grid - widen the parameter ranges or reduce the number of time frames."
                        )

                    with st.expander("Full robustness ranking"):
                        display_robust = robustness.rename(columns={
                            "window_size": "Window (days)", "rebalance_freq": "Rebalance (days)",
                            "n_timeframes_tested": "Time frames tested",
                            "n_timeframes_outperform": "Time frames outperformed",
                            "pct_outperform": "% outperformed",
                            "mean_excess_return": "Mean excess return",
                            "worst_excess_return": "Worst-case excess return",
                            "best_excess_return": "Best-case excess return",
                            "mean_sharpe": "Mean Sharpe",
                            "fully_robust": "Fully robust",
                        })
                        st.dataframe(
                            display_robust.style.format({
                                "% outperformed": "{:.0%}", "Mean excess return": "{:+.2%}",
                                "Worst-case excess return": "{:+.2%}", "Best-case excess return": "{:+.2%}",
                                "Mean Sharpe": "{:.2f}",
                            }),
                            use_container_width=True,
                        )

# ---------------------------------------------------------------------
# Tab 4: Raw data / export
# ---------------------------------------------------------------------
with tab_client:
    st.subheader("Client risk profiling")
    st.caption(
        "Complete the suitability questionnaire below to determine the client's risk category and "
        "the resulting blend between the Minimum-Variance and Mean-Variance portfolios. Both "
        "portfolios use their most robust rolling-window / rebalancing-frequency combination and "
        "their LATEST out-of-sample weights - i.e. what each strategy would hold today if it had "
        "been run live, not a backward-looking in-sample fit. The robustness search always uses the "
        "Optimization tab's *default* grid (window 60-252d, rebalance 5-126d, 6/12 months + full "
        "period) - if you change the sliders on that tab, results shown there may differ from this "
        "tab, which always uses the same reproducible default."
    )

    with st.form("client_questionnaire"):
        answers = []
        current_theme = None
        for i, q in enumerate(lib.QUESTIONNAIRE):
            if q["theme"] != current_theme:
                st.markdown(f"**{q['theme']}**")
                current_theme = q["theme"]
            labels = [opt[0] for opt in q["options"]]
            choice = st.radio(q["question"], labels, index=None, key=f"client_q_{i}")
            answers.append((q, choice))

        st.markdown("**Investment amount**")
        capital_amount = st.number_input(
            "Amount to invest (CHF, for reference only - does not affect the risk score)",
            min_value=0, value=500000, step=10000,
        )

        submitted = st.form_submit_button("Calculate risk profile & build portfolio", type="primary")

    if submitted:
        if any(choice is None for _, choice in answers):
            st.warning("Please answer every question before submitting.")
        else:
            score = sum(dict(q["options"])[choice] for q, choice in answers)
            st.session_state["client_score"] = score
            st.session_state["client_category"] = lib.score_to_risk_category(score)
            st.session_state["client_capital"] = capital_amount

    if "client_category" in st.session_state:
        category = st.session_state["client_category"]
        score = st.session_state["client_score"]
        capital_amount = st.session_state["client_capital"]
        w_minvar_ratio, w_meanvar_ratio = lib.RISK_BLEND[category]

        st.markdown("---")
        st.markdown(f"### Result: **{category}**")
        st.caption(f"Questionnaire score: {score} / {lib.QUESTIONNAIRE_MAX_SCORE}")
        st.info(
            f"Recommended allocation: **{w_minvar_ratio:.0%} Minimum-Variance** portfolio + "
            f"**{w_meanvar_ratio:.0%} Mean-Variance** portfolio."
        )

        with st.spinner("Finding the most robust parameters for each strategy..."):
            msci_returns_client = cached_msci(start_date, end_date, fx_data)
            best_mv = cached_best_params(returns, risk_free_daily, risk_free_rate, msci_returns_client, "Mean-Variance")
            best_minvar = cached_best_params(returns, risk_free_daily, risk_free_rate, msci_returns_client, "Min-Variance")

        if best_mv is None or best_minvar is None:
            st.error(
                "Could not determine robust parameters with the current price history - try widening "
                "the date range in the sidebar."
            )
        else:
            mv_window, mv_rebal = best_mv
            minvar_window, minvar_rebal = best_minvar

            with st.expander("Parameters used (most robust per strategy)"):
                st.write(f"Mean-Variance: lookback window = {mv_window} days, rebalance every {mv_rebal} days")
                st.write(f"Min-Variance: lookback window = {minvar_window} days, rebalance every {minvar_rebal} days")

            rw_mv_final, oos_mv_final, _ = lib.rolling_mean_variance(returns, risk_free_daily, mv_window, mv_rebal)
            rw_minvar_final, oos_minvar_final, _ = lib.rolling_min_variance(returns, minvar_window, minvar_rebal)

            if rw_mv_final.dropna().empty or rw_minvar_final.dropna().empty:
                st.error(
                    "Not enough out-of-sample history yet to build a live portfolio with these "
                    "parameters - try widening the date range in the sidebar."
                )
            else:
                latest_mv = rw_mv_final.dropna().iloc[-1]
                latest_minvar = rw_minvar_final.dropna().iloc[-1]
                combined_weights = lib.blend_weights(latest_minvar, latest_mv, category)

                display_weights = combined_weights.rename(
                    index=lambda t: asset_info.loc[t, "Name"] if t in asset_info.index else t
                ).sort_values(ascending=False)

                st.markdown("### Recommended portfolio")
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.dataframe(display_weights.map(lambda x: f"{x:.1%}").rename("Weight"),
                                 use_container_width=True)
                    if capital_amount:
                        st.caption("Indicative allocation in CHF:")
                        st.dataframe(
                            (display_weights * capital_amount).map(lambda x: f"{x:,.0f} CHF").rename("Amount"),
                            use_container_width=True,
                        )
                with col2:
                    fig = px.pie(values=display_weights.values, names=display_weights.index,
                                 title="Combined portfolio weights")
                    st.plotly_chart(fig, use_container_width=True)

                common_idx = oos_mv_final.index.intersection(oos_minvar_final.index)
                if len(common_idx) > 5:
                    combined_oos = (w_minvar_ratio * oos_minvar_final.loc[common_idx]
                                     + w_meanvar_ratio * oos_mv_final.loc[common_idx])
                    comp = pd.DataFrame({
                        "Your portfolio": combined_oos,
                        "MSCI World": msci_returns_client.reindex(common_idx),
                    })
                    cum = lib.rebase_to_zero(comp)
                    st.markdown("### Recent out-of-sample performance (illustrative)")
                    # ADD: make the actual start date explicit, rather than leaving the user to
                    # infer it from the chart. The combined series can only start once BOTH
                    # strategies have enough lookback history - i.e. at the longer of the two
                    # windows - which is exactly what the index intersection above enforces.
                    longer_window = max(mv_window, minvar_window)
                    longer_strategy = "Min-Variance" if minvar_window >= mv_window else "Mean-Variance"
                    st.caption(
                        f"Combined performance starts on **{common_idx[0].strftime('%d.%m.%Y')}** "
                        f"({longer_window} trading days into the loaded history) - the point where "
                        f"BOTH strategies have enough lookback data, driven by {longer_strategy}'s "
                        f"{longer_window}-day window (the longer of the two)."
                    )
                    fig = px.line(cum)
                    fig.update_layout(yaxis_tickformat=",.0%", yaxis_title="Cumulative performance",
                                       legend_title="")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "⚠️ Based on a limited out-of-sample history - illustrative only, not a "
                        "guarantee of future performance."
                    )
                else:
                    longer_window = max(mv_window, minvar_window)
                    st.info(
                        f"Not enough overlapping out-of-sample history yet to chart combined "
                        f"performance: the longer of the two lookback windows "
                        f"({longer_window} days) leaves too few trading days after it within the "
                        f"loaded price history. Widen the date range in the sidebar to see this chart."
                    )

with tab_data:
    st.subheader("Daily returns (CHF)")
    st.dataframe(returns, use_container_width=True)
    st.download_button(
        "Download returns as CSV", returns.to_csv().encode("utf-8"),
        file_name="returns.csv", mime="text/csv",
    )

    st.subheader("Prices (CHF)")
    st.dataframe(prices, use_container_width=True)
    st.download_button(
        "Download prices as CSV", prices.to_csv().encode("utf-8"),
        file_name="prices_chf.csv", mime="text/csv",
    )
