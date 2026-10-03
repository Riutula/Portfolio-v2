"""
tests/test_app_e2e.py

Executes app.py top-to-bottom with a stubbed `streamlit` module and mocked
`yfinance` / `pandas_datareader`, so the full script - including every tab -
runs without a real browser session or network access. This is a smoke test
for the Streamlit-specific wiring (widget calls, session_state, caching),
complementary to tests/test_lib.py which tests the underlying logic.

Run with: python tests/test_app_e2e.py
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, REPO_ROOT)

np.random.seed(123)

n_obs = 320
tickers_all = ['CRM', 'HOLN.SW', 'LLY', 'NESN.SW', 'CTPNV.AS', '5JS.SI',
               'AMRZ', 'ASCN.SW', 'TPXE.PA', 'ECL', 'HYG', '0P00000BKL']
dates = pd.date_range("2025-06-23", periods=n_obs, freq="B")


# ---------------------------------------------------------------------
# Minimal Streamlit stub: enough surface area for app.py to run top-to-bottom
# without a real browser/session, while still exercising every code path.
# ---------------------------------------------------------------------
class DummyColumn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DummyExpander:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DummySidebar:
    def title(self, *a, **k): pass
    def subheader(self, *a, **k): pass
    def markdown(self, *a, **k): pass

    def multiselect(self, label, options, default=None, **k):
        return list(default) if default is not None else list(options)

    def date_input(self, label, value=None, **k):
        return value

    def slider(self, label, min_value, max_value, value, **k):
        return value

    def button(self, *a, **k):
        return True  # simulate the "Run" button being clicked

    def spinner(self, *a, **k):
        return DummyExpander()


class StStub(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.sidebar = DummySidebar()
        self.session_state = SessionState()

    # ---- layout / display no-ops ----
    def set_page_config(self, *a, **k): pass
    def title(self, *a, **k): pass
    def caption(self, *a, **k): pass
    def subheader(self, *a, **k): pass
    def markdown(self, *a, **k): pass
    def write(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, msg, *a, **k): raise RuntimeError(f"st.error called: {msg}")

    def stop(self):
        raise SystemExit("st.stop() called")

    def spinner(self, *a, **k):
        return DummyExpander()

    def expander(self, *a, **k):
        return DummyExpander()

    def columns(self, n, **k):
        count = n if isinstance(n, int) else len(n)
        return [DummyColumn() for _ in range(count)]

    def tabs(self, labels):
        return [DummyColumn() for _ in labels]

    def dataframe(self, df, *a, **k):
        assert df is not None

    def download_button(self, *a, **k): pass

    def plotly_chart(self, fig, *a, **k):
        assert fig is not None

    def pyplot(self, fig, *a, **k):
        assert fig is not None

    # ---- top-level widgets used outside the sidebar ----
    def selectbox(self, label, options, **k):
        options = list(options)
        return options[0] if options else None

    def multiselect(self, label, options, default=None, **k):
        return list(default) if default is not None else list(options)

    def slider(self, label, min_value, max_value, value=None, **k):
        return value if value is not None else min_value

    def button(self, *a, **k):
        return True

    def success(self, *a, **k): pass

    def checkbox(self, label, value=False, **k):
        return value

    def radio(self, label, options, index=0, **k):
        options = list(options)
        if not options:
            return None
        if index is None:
            return options[0]  # simulate an answered question (happy-path)
        return options[index]

    def number_input(self, label, min_value=None, value=0, **k):
        return value

    def form(self, key, **k):
        return DummyColumn()

    def form_submit_button(self, *a, **k):
        return True

    def progress(self, *a, **k):
        class DummyProgress:
            def progress(self, *a, **k): pass
            def empty(self): pass
        return DummyProgress()

    # ---- caching: simple passthrough cache keyed by args ----
    def cache_data(self, *dargs, **dkwargs):
        def decorator(func):
            store = {}

            def wrapper(*args, **kwargs):
                key = (args, tuple(sorted(kwargs.items())))
                try:
                    hash(key)
                    cache_key = key
                except TypeError:
                    cache_key = None
                if cache_key is not None and cache_key in store:
                    return store[cache_key]
                result = func(*args, **kwargs)
                if cache_key is not None:
                    store[cache_key] = result
                return result
            return wrapper
        return decorator


class SessionState(dict):
    """Supports both st.session_state['x'] and st.session_state.x access."""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


st_stub = StStub()
sys.modules["streamlit"] = st_stub


# ---------------------------------------------------------------------
# Mock yfinance / pandas_datareader BEFORE importing portfolio_lib / app
# ---------------------------------------------------------------------
class FakeYF:
    @staticmethod
    def download(t, start=None, end=None, auto_adjust=True):
        tl = [t] if isinstance(t, str) else list(t)
        data = {x: 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_obs)) for x in tl}
        return {"Close": pd.DataFrame(data, index=dates)}

    class Ticker:
        def __init__(self, t):
            fake_names = {
                'CRM': 'Salesforce, Inc.', 'HOLN.SW': 'Holcim AG', 'LLY': 'Eli Lilly and Company',
                'NESN.SW': 'Nestle S.A.', 'CTPNV.AS': 'CTP N.V.', '5JS.SI': 'Some Singapore Co Ltd',
                'AMRZ': 'Amrize AG', 'ASCN.SW': 'Some Swiss Co AG', 'TPXE.PA': 'Amundi Prime Japan UCITS ETF',
                'ECL': 'Ecolab Inc.', 'HYG': 'iShares iBoxx High Yield Corporate Bond ETF',
                '0P00000BKL': 'Some Money Market Fund',
            }
            fake_currencies = {'AAPL': 'USD', 'VOD.L': 'GBP'}
            self.info = {
                "dividendYield": float(np.random.uniform(0.015, 0.06)),
                "longName": fake_names.get(t, t),
                "currency": fake_currencies.get(t, "USD"),
            }


class FakeWeb:
    @staticmethod
    def DataReader(code, source, start=None, end=None):
        if code == 'DTB3':
            vals = np.random.uniform(4.0, 5.5, n_obs)
            return pd.DataFrame({'DTB3': vals}, index=dates)
        if code == 'DGS10':
            vals = 4.0 + np.cumsum(np.random.normal(0, 0.02, n_obs))
            return pd.DataFrame({'DGS10': vals}, index=dates)
        if code == 'T10YIE':
            vals = 2.3 + np.cumsum(np.random.normal(0, 0.01, n_obs))
            return pd.DataFrame({'T10YIE': vals}, index=dates)
        raise ValueError(code)


sys.modules["yfinance"] = FakeYF()
fake_pdr_module = types.ModuleType("pandas_datareader")
fake_pdr_module.DataReader = FakeWeb.DataReader
sys.modules["pandas_datareader"] = fake_pdr_module

import plotly.express as px  # noqa: E402,F401 (must be importable; real plotly is used)

# ---------------------------------------------------------------------
# Run app.py end to end
# ---------------------------------------------------------------------
app_path = os.path.join(REPO_ROOT, "app.py")
app_globals = {"__name__": "__main__", "__file__": app_path}
with open(app_path) as f:
    app_code = f.read()

exec(compile(app_code, "app.py", "exec"), app_globals)

# ---- Optimization tab ----
assert "optim_grid" in st_stub.session_state, "optimization grid was not computed/stored"
grid = st_stub.session_state["optim_grid"]
assert grid.shape[0] > 0, "optimization grid is empty"
assert "Excess Return vs Benchmark" in grid.columns, "benchmark comparison columns missing"
assert "Outperforms" in grid.columns
assert grid["Excess Return vs Benchmark"].notna().any(), "no valid excess return in the optimization grid"
print(f"CONFIRMED: optimization grid computed with {grid.shape[0]} combinations, "
      f"strategy={st_stub.session_state['optim_strategy']}, "
      f"{int(grid['Outperforms'].sum())} outperforming MSCI World.")

# ---- Client tab ----
assert "client_category" in st_stub.session_state, "client risk category was not computed"
category = st_stub.session_state["client_category"]
score = st_stub.session_state["client_score"]
assert category in ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
print(f"CONFIRMED: client questionnaire scored {score}/30 -> category = {category}")

print("\nAPP.PY EXECUTED END TO END WITHOUT ERROR (all 6 tabs rendered: Assets, Portfolio "
      "Construction, Out-of-Sample, Optimization, Client, Raw Data).")

# ---------------------------------------------------------------------
# Dedicated run: user types a custom ticker not in DEFAULT_TICKERS
# ---------------------------------------------------------------------
import portfolio_lib as lib  # noqa: E402

original_sidebar_multiselect = st_stub.sidebar.multiselect


def sidebar_multiselect_with_custom_ticker(label, options, default=None, **k):
    if label == "Universe":
        return list(default) + ["AAPL"]
    return original_sidebar_multiselect(label, options, default=default, **k)


st_stub.sidebar.multiselect = sidebar_multiselect_with_custom_ticker
st_stub.session_state = SessionState()  # fresh session

app_globals2 = {"__name__": "__main__", "__file__": app_path}
exec(compile(app_code, "app.py", "exec"), app_globals2)

assert "AAPL" in app_globals2["currency_map"], "custom ticker missing from currency_map"
assert app_globals2["currency_map"]["AAPL"] == "USD", "custom ticker currency not detected correctly"
print("CONFIRMED: custom ticker 'AAPL' (not in DEFAULT_TICKERS) accepted, currency auto-detected, "
      "full pipeline re-ran without error.")
