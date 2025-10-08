# ajinkya_predictor_final_release.py
# Ajinkya's Predictor - Final Release (ready-to-paste)
# - UI title: Ajinkya's Predictor
# - Ensemble: Holt-Winters, Ridge, RandomForest, AR(1), FFT spectral, Trend extrapolate
# - Risk controls: Historical VaR, Parametric VaR (Gaussian approx via sampling), CVaR (Expected Shortfall)
# - UI displays only the fields you asked for; all other computation runs internally.
#
# Requirements (suggested):
# streamlit, pandas, numpy, scikit-learn, statsmodels, yfinance, requests
#
# Usage:
# pip install -r requirements.txt
# streamlit run ajinkya_predictor_final_release.py

import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
import warnings
from datetime import datetime, timedelta
from functools import lru_cache

warnings.filterwarnings("ignore")

# Optional libraries (wrapped in try/except)
try:
    import yfinance as yf
    HAVE_YFINANCE = True
except Exception:
    HAVE_YFINANCE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.decomposition import PCA
    HAVE_SK = True
except Exception:
    HAVE_SK = False

st.set_page_config(page_title="Ajinkya's Predictor", layout="wide")

# -------------------------
# Stylish Header (only app name visible)
# -------------------------
st.markdown("""
<div style="background:linear-gradient(90deg,#072b17,#0b6e4f); padding:18px; border-radius:12px; color: white; display:flex; align-items:center; gap:16px;">
  <div style="width:64px;height:64px;background:linear-gradient(135deg,#66cdaa,#0b6e4f);border-radius:12px;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:22px;color:#042f1b;">
    AJ
  </div>
  <div>
    <div style="font-size:34px; font-weight:800; color:white; letter-spacing:0.6px;">Ajinkya's Predictor</div>
    <div style="font-size:13px; color:#e6fff3; margin-top:4px;">Math-first, RF+Ridge ensemble with advanced risk controls</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Inputs
# -------------------------
with st.form("inputs", clear_on_submit=False):
    left, right = st.columns([3, 1])
    with left:
        fmp_key = st.text_input("FMP API Key (optional)", type="password")
        ticker_raw = st.text_input("Ticker (Indian, e.g., TCS, RELIANCE)", value="TCS")
    with right:
        interval = st.selectbox("Interval", ["3d", "15d", "1m", "3m", "6m", "1y"])
        submitted = st.form_submit_button("Run Prediction")

# map intervals to trading days approx.
MAP = {"3d": 3, "15d": 15, "1m": 22, "3m": 66, "6m": 132, "1y": 252}

# -------------------------
# Helpers & Data Fetch
# -------------------------
def append_nse_if_needed(ticker: str):
    t = ticker.strip().upper()
    if "." in t:
        return t
    return t + ".NS"

def safe_api_get(url, params=None, timeout=8):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

@lru_cache(maxsize=64)
def fetch_prices_yf(ticker: str, days=1500):
    """Fetch daily historical OHLCV via yfinance (if available)."""
    if not HAVE_YFINANCE:
        return None
    try:
        t = yf.Ticker(ticker)
        end = datetime.now()
        start = end - timedelta(days=int(days * 1.1))
        df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1d", actions=False)
        if df.empty:
            return None
        df = df.reset_index()
        df = df.rename(columns={"Date": "date", "Close": "close", "High": "high", "Low": "low", "Volume": "volume"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df[["date", "close", "high", "low", "volume"]]
    except Exception:
        return None

# -------------------------
# Core signal math & indicators
# -------------------------
def log_return(series):
    return np.log(series / series.shift(1)).replace([np.inf, -np.inf], 0).fillna(0)

def ema(series, span): 
    return series.ewm(span=span, adjust=False).mean()

def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).fillna(0)
    down = -1 * delta.clip(upper=0).fillna(0)
    ma_up = up.ewm(alpha=1.0/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1.0/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def zscore(series, window=20):
    m = series.rolling(window=window, min_periods=1).mean()
    s = series.rolling(window=window, min_periods=1).std().replace(0, np.nan)
    return ((series - m) / s).fillna(0)

def hurst_exponent(ts, lags_range=range(2, 50)):
    ts = np.array(ts.dropna())
    if len(ts) < 100:
        return 0.5
    rs = []
    for lag in lags_range:
        pp = np.subtract(ts[lag:], ts[:-lag])
        rs.append(np.std(pp))
    rs = np.array(rs)
    with np.errstate(divide='ignore', invalid='ignore'):
        poly = np.polyfit(np.log(lags_range), np.log(rs + 1e-9), 1)
    return max(0.0, min(1.0, poly[0]))

def rolling_ols_slope(series, window=30):
    return series.rolling(window).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if x.size >= 3 else 0, raw=False).fillna(0)

def kalman_smooth(series):
    n = len(series)
    xhat = np.zeros(n)
    P = np.zeros(n)
    Q = 1e-5
    R = np.var(series) * 0.01 + 1e-6
    xhat[0] = series.iloc[0]
    P[0] = 1.0
    for k in range(1, n):
        xhatminus = xhat[k-1]
        Pminus = P[k-1] + Q
        K = Pminus / (Pminus + R)
        xhat[k] = xhatminus + K * (series.iloc[k] - xhatminus)
        P[k] = (1 - K) * Pminus
    return pd.Series(xhat, index=series.index)

def fft_dominant_forecast(series, days_ahead=3, top_k=4):
    x = series.values
    n = len(x)
    if n < 12:
        return float(series.iloc[-1])
    t = np.arange(n)
    p = np.polyfit(t, x, 1)
    trend = np.polyval(p, t)
    resid = x - trend
    fft = np.fft.rfft(resid)
    freqs = np.fft.rfftfreq(n)
    amps = np.abs(fft)
    idx = np.argsort(amps)[-top_k:]
    future_t = np.arange(n, n + days_ahead)
    recon = np.zeros(days_ahead)
    for i in idx:
        a = fft[i]
        freq = freqs[i]
        phase = np.angle(a)
        amplitude = np.abs(a) / n * 2
        recon += amplitude * np.cos(2 * np.pi * freq * future_t + phase)
    last_trend = np.polyval(p, n - 1)
    slope = p[0]
    trend_fore = last_trend + slope * np.arange(1, days_ahead + 1)
    forecast = trend_fore + recon
    return float(forecast[-1])

def prepare_features(df):
    df = df.copy()
    df["return1"] = df["close"].pct_change().fillna(0)
    df["logret"] = log_return(df["close"])
    df["ema8"] = ema(df["close"], 8)
    df["ema21"] = ema(df["close"], 21)
    df["sma50"] = sma(df["close"], 50)
    df["vol20"] = df["return1"].rolling(20).std().fillna(0)
    df["ewma_vol"] = df["return1"].ewm(span=20, adjust=False).std().fillna(0)
    df["rsi14"] = rsi(df["close"])
    df["ret_z"] = zscore(df["return1"])
    for lag in [1, 2, 3, 5, 8, 13, 21]:
        df[f"lag_{lag}"] = df["close"].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

# -------------------------
# Forecast building blocks
# -------------------------
def hw_predict(df, days_ahead=3):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        hw = ExponentialSmoothing(df["close"], trend="add", seasonal=None, damped_trend=True)
        hwf = hw.fit(optimized=True)
        return float(hwf.forecast(days_ahead).iloc[-1])
    except Exception:
        return float(df["close"].iloc[-1])

def ridge_predict(df, days_ahead=3):
    try:
        from sklearn.linear_model import Ridge
    except Exception:
        return float(df["close"].iloc[-1])
    df2 = df.copy().reset_index(drop=True)
    cols = [c for c in df2.columns if c.startswith("lag_")]
    if len(df2) < 40 or len(cols) == 0:
        return float(df["close"].iloc[-1])
    X = df2[cols]
    y = df2["close"]
    split = max(5, int(0.8 * len(X)))
    try:
        model = Ridge(alpha=1.0)
        model.fit(X.iloc[:split], y.iloc[:split])
        return float(model.predict(X.iloc[-1:].values.reshape(1, -1))[0])
    except Exception:
        return float(df["close"].iloc[-1])

def rf_predict(df, days_ahead=3):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return float(df["close"].iloc[-1])
    df2 = df.copy().reset_index(drop=True)
    cols = [c for c in df2.columns if c.startswith("lag_")] + ["vol20", "ewma_vol", "rsi14", "ret_z"]
    if len(df2) < 80:
        return float(df["close"].iloc[-1])
    X = df2[cols]
    y = df2["close"]
    split = max(10, int(0.8 * len(X)))
    model = RandomForestRegressor(n_estimators=80, max_depth=7, random_state=1, n_jobs=1)
    model.fit(X.iloc[:split], y.iloc[:split])
    return float(model.predict(X.iloc[-1:].values.reshape(1, -1))[0])

def ar1_predict(df, days_ahead=3):
    try:
        r = df["logret"].dropna()
        if len(r) < 10:
            return float(df["close"].iloc[-1])
        phi = np.corrcoef(r[:-1], r[1:])[0, 1] if len(r) > 1 else 0
        mu = r.mean()
        last = r.iloc[-1]
        forecast_logret = mu + (phi * (last - mu))
        price_fore = float(df["close"].iloc[-1] * math.exp(forecast_logret * days_ahead))
        return price_fore
    except Exception:
        return float(df["close"].iloc[-1])

def trend_extrapolate(df, days_ahead=3):
    slope = rolling_ols_slope(df["close"], window=30).iloc[-1]
    return float(df["close"].iloc[-1] + slope * days_ahead)

def ensemble_predict(df, days_ahead=3):
    p_hw = hw_predict(df, days_ahead)
    p_ridge = ridge_predict(df, days_ahead)
    p_rf = rf_predict(df, days_ahead)
    p_ar1 = ar1_predict(df, days_ahead)
    p_spec = fft_dominant_forecast(df["close"], days_ahead=days_ahead)
    p_trend = trend_extrapolate(df, days_ahead)
    # adaptive weights
    hurst = hurst_exponent(df["close"])
    vol = df["ewma_vol"].iloc[-1] if "ewma_vol" in df else df["vol20"].iloc[-1]
    weights = {"hw": 0.18, "ridge": 0.22, "rf": 0.22, "ar1": 0.10, "spec": 0.14, "trend": 0.14}
    if hurst > 0.55:
        weights["trend"] += 0.10
        weights["spec"] -= 0.05
        weights["ar1"] -= 0.05
    if vol > 0.06:
        weights["spec"] += 0.10
        weights["rf"] += 0.05
        weights["trend"] -= 0.05
    arr = np.array(list(weights.values()))
    arr = np.clip(arr, 0.01, 0.9)
    arr = arr / arr.sum()
    wkeys = list(weights.keys())
    weights = dict(zip(wkeys, [float(x) for x in arr]))
    preds = {"hw": p_hw, "ridge": p_ridge, "rf": p_rf, "ar1": p_ar1, "spec": p_spec, "trend": p_trend}
    pred = sum(preds[k] * weights[k] for k in preds)
    sigma = vol * math.sqrt(days_ahead)
    lower = max(0.0, pred * (1 - sigma))
    upper = pred * (1 + sigma)
    return float(pred), float(lower), float(upper), weights

# -------------------------
# Risk controls: VaR & CVaR
# -------------------------
def historical_var(returns, alpha=0.05):
    """Historical VaR: positive number representing expected loss quantile."""
    if len(returns) < 10:
        return 0.0
    q = np.percentile(returns, alpha * 100)  # e.g., 5th percentile (often negative)
    return float(max(0.0, -q))

def parametric_var(returns, alpha=0.05):
    """Parametric Gaussian VaR using simple sampling for z quantile (no scipy dependency)."""
    if len(returns) < 10:
        return 0.0
    mu = np.mean(returns)
    sigma = np.std(returns)
    # approximate z_alpha using sampling from standard normal
    z = np.percentile(np.random.normal(size=200000), alpha * 100)
    var = -(mu + sigma * z)
    return float(max(0.0, var))

def cvar_expected_shortfall(returns, alpha=0.05):
    """Expected shortfall (CVaR) at level alpha."""
    if len(returns) < 10:
        return 0.0
    threshold = np.quantile(returns, alpha)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return 0.0
    return float(max(0.0, -np.mean(tail)))

# -------------------------
# Scoring, sentiment, description, recommendation
# -------------------------
def compute_momentum_score(df):
    ret = df["close"].pct_change(5).iloc[-1]
    ema_slope = (df["ema8"].iloc[-1] - df["ema8"].iloc[-12]) if len(df) >= 12 else 0
    r = df["rsi14"].iloc[-1] if "rsi14" in df else 50
    score = 0.4 * (np.tanh(ret * 10)) + 0.35 * (np.tanh(ema_slope / (df["close"].iloc[-1] + 1e-9))) + 0.25 * ((r - 50) / 50)
    return float(max(0, min(100, (score + 1) / 2 * 100)))

def compute_fundamentals_score(profile_fmp, profile_yf):
    score = 50.0
    try:
        if profile_fmp and isinstance(profile_fmp, dict):
            mcap = float(profile_fmp.get("mktCap", 0) or 0)
            pe = float(profile_fmp.get("pe", 0) or 0)
            roe = float(profile_fmp.get("returnOnEquity", 0) or 0)
            if mcap > 0:
                score += 5
            if 0 < pe < 30:
                score += 5
            if roe > 0.05:
                score += 5
    except Exception:
        pass
    try:
        if profile_yf and isinstance(profile_yf, dict):
            pe = profile_yf.get("trailingPE") or profile_yf.get("forwardPE")
            if pe and 0 < pe < 30:
                score += 5
            roe = profile_yf.get("returnOnEquity") or profile_yf.get("roe")
            if roe and roe > 0.05:
                score += 5
    except Exception:
        pass
    return float(max(0, min(100, score)))

def detect_trend(df):
    ema8 = df["ema8"].iloc[-1]
    ema21 = df["ema21"].iloc[-1]
    sma50 = df["sma50"].iloc[-1]
    price = df["close"].iloc[-1]
    if price > ema8 > ema21 and price > sma50:
        return "Strong Uptrend"
    if ema8 > ema21 and price >= sma50:
        return "Uptrend"
    if ema8 < ema21 and price < sma50:
        return "Downtrend"
    return "Sideways / Uncertain"

POS = set(["good", "beat", "beats", "growth", "upgrade", "strong", "positive", "outperform", "gain", "rise", "record"])
NEG = set(["loss", "miss", "misses", "downgrade", "weak", "negative", "underperform", "drop", "fall", "concern", "decline", "cut"])

@lru_cache(maxsize=32)
def fetch_news_yf(ticker: str):
    if not HAVE_YFINANCE:
        return []
    try:
        t = yf.Ticker(ticker)
        news = getattr(t, "news", []) or []
        items = []
        for n in news[:60]:
            items.append({"title": n.get("title", ""), "text": n.get("summary", "")})
        return items
    except Exception:
        return []

def simple_sentiment_desc(news_items):
    score = 0.0
    count = 0
    for n in news_items[:60]:
        txt = " ".join([str(n.get(k, "")) for k in ("title", "text")]).lower()
        s = 0
        for w in POS:
            if w in txt:
                s += 1
        for w in NEG:
            if w in txt:
                s -= 1
        score += s
        count += 1
    if count == 0:
        return 0.0, "No recent news available."
    val = float(max(-1.0, min(1.0, score / (8 * count))))
    if val > 0.2:
        desc = "News tone is generally positive."
    elif val < -0.2:
        desc = "News tone is generally negative — check headlines."
    else:
        desc = "News tone is mixed/neutral."
    return val, desc

@lru_cache(maxsize=32)
def fetch_company_description(fmp_key: str, ticker: str):
    desc = ""
    if fmp_key:
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}"
            data = safe_api_get(url, params={"apikey": fmp_key})
            if data and isinstance(data, list) and len(data) > 0:
                d = data[0]
                desc = d.get("description") or d.get("companyName") or ""
                if desc:
                    return desc
        except Exception:
            pass
    if HAVE_YFINANCE:
        try:
            info = yf.Ticker(ticker).info
            desc = info.get("longBusinessSummary") or info.get("shortBusinessSummary") or ""
            if desc:
                return desc
        except Exception:
            pass
    return desc or "No company description available."

def recommendation_text(pred, cur, lower, upper, implied_return, conf, sentiment_score, sentiment_desc, momentum_score, fundamentals_score, trend):
    change = (pred / cur - 1) * 100
    lines = []
    lines.append(f"The model predicts ~{change:.2f}% change over your chosen horizon. Confidence: {conf:.2f}.")
    lines.append(f"Trend: {trend}. Momentum score: {momentum_score:.1f}/100. Fundamentals score: {fundamentals_score:.1f}/100.")
    lines.append(f"News: {sentiment_desc}")
    lines.append("")
    buy_price = cur * (1 - 0.03)
    strong_buy_price = cur * (1 - 0.08)
    take_profit = cur * (1 + max(0.06, change / 2 / 100))
    stop_loss = cur * (1 - 0.07)
    if change > 6 and conf > 0.55 and momentum_score > 55:
        lines.append("Recommendation: BUY (Reason: expected upside and supporting momentum).")
        lines.append(f"Suggested entry: consider buying near {buy_price:.2f} or in tranches. Strong entry if dips to {strong_buy_price:.2f}.")
        lines.append(f"Target / take-profit: {take_profit:.2f}. Stop-loss suggestion: {stop_loss:.2f}.")
    elif change > 2 and conf > 0.45:
        lines.append("Recommendation: CONSIDER BUY (small position).")
        lines.append(f"Suggested entry: small buy near {buy_price:.2f}. Target: {take_profit:.2f}. Stop-loss: {stop_loss:.2f}.")
    elif change < -4 and conf > 0.5:
        lines.append("Recommendation: SELL / AVOID NEW BUY (Reason: downside expected).")
        lines.append(f"If holding, consider trimming or set stop-loss near {stop_loss:.2f}.")
    else:
        lines.append("Recommendation: HOLD / WAIT (No clear edge).")
        lines.append("If you want to enter, prefer small positions and tight stop-losses; wait for clearer trend confirmation.")
    lines.append("")
    lines.append(f"Practical predicted range: {lower:.2f} — {upper:.2f}. Implied return: {implied_return:.2f}%.")
    lines.append("Risk note: Use position sizing; VaR and CVaR below estimate potential shortfall levels for one day.")
    return "\n".join(lines)

# -------------------------
# Main app execution
# -------------------------
if submitted:
    ticker_input = ticker_raw.strip().upper()
    if "." not in ticker_input:
        ticker = append_nse_if_needed(ticker_input)
    else:
        ticker = ticker_input
    hist = None
    if HAVE_YFINANCE:
        hist = fetch_prices_yf(ticker, days=1500)
    if hist is None or len(hist) < 80:
        st.error("Not enough historical data. Ensure yfinance is installed and ticker is valid.")
    else:
        df = hist.rename(columns={"date": "date", "close": "close", "high": "high", "low": "low", "volume": "volume"}).sort_values("date").reset_index(drop=True)
        dfp = prepare_features(df)
        # smoothing
        dfp["kf_close"] = kalman_smooth(dfp["close"])
        # PCA (optional)
        if HAVE_SK:
            try:
                cols = [c for c in dfp.columns if c not in ["date", "close", "high", "low", "volume"]]
                if len(cols) > 6:
                    pca = PCA(n_components=min(6, len(cols)))
                    comps = pca.fit_transform(dfp[cols].fillna(0))
                    for i in range(comps.shape[1]):
                        dfp[f"pca_{i}"] = comps[:, i]
            except Exception:
                pass
        days = MAP.get(interval, 3)
        pred, lower, upper, weights = ensemble_predict(dfp, days)
        cur = float(dfp["close"].iloc[-1])
        implied_return = (pred / cur - 1) * 100.0
        # sentiment & description
        news_items = fetch_news_yf(ticker) if HAVE_YFINANCE else []
        sentiment_score, sentiment_desc = simple_sentiment_desc(news_items)
        momentum_score = compute_momentum_score(dfp)
        desc = fetch_company_description(fmp_key, ticker)
        profile_yf = None
        if HAVE_YFINANCE:
            try:
                profile_yf = yf.Ticker(ticker).info
            except Exception:
                profile_yf = None
        fundamentals_score = compute_fundamentals_score(None, profile_yf)
        trend = detect_trend(dfp)
        vol = dfp["ewma_vol"].iloc[-1] if "ewma_vol" in dfp else dfp["vol20"].iloc[-1]
        conf = 0.5 + 0.28 * (1 - min(1, vol * 10))
        conf = max(0.12, min(0.98, conf))
        final_score = 0.45 * (momentum_score / 100) + 0.35 * (fundamentals_score / 100) + 0.20 * conf
        final_score_pct = float(max(0, min(100, final_score * 100)))
        # risk metrics (daily returns)
        daily_returns = df["close"].pct_change().dropna().values
        hist_var_95 = historical_var(daily_returns, alpha=0.05)
        param_var_95 = parametric_var(daily_returns, alpha=0.05)
        cvar_95 = cvar_expected_shortfall(daily_returns, alpha=0.05)
        # recommendation
        rec_text = recommendation_text(pred, cur, lower, upper, implied_return, conf, sentiment_score, sentiment_desc, momentum_score, fundamentals_score, trend)
        # -------------------------
        # Display: only the fields you requested
        # -------------------------
        st.markdown("<div style='max-width:980px;margin:18px auto;padding:20px;border-radius:14px;box-shadow:0 12px 36px rgba(6,45,32,0.08);background:linear-gradient(180deg,#ffffff,#f7fff8);'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Current Price</h3><p style='font-size:22px;margin-top:0;margin-bottom:8px;font-weight:600;'>{cur:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Predicted Share Price ({interval})</h3><p style='font-size:22px;margin-top:0;margin-bottom:8px;font-weight:600;'>{pred:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Practical Predicted Range</h3><p style='font-size:18px;margin-top:0;margin-bottom:8px;'>{lower:.2f} — {upper:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-bottom:6px;'>Implied Return</h3><p style='font-size:18px;margin-top:0;margin-bottom:8px;'>{implied_return:.2f}%</p>", unsafe_allow_html=True)
        cols = st.columns(3)
        cols[0].metric("Momentum (0-100)", f"{momentum_score:.1f}")
        cols[1].metric("Fundamentals (0-100)", f"{fundamentals_score:.1f}")
        cols[2].metric("Trend", f"{trend}")
        st.markdown(f"<h3 style='color:#0B6E4F;margin-top:8px;margin-bottom:6px;'>Company Description</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:14px;color:#333;margin-top:0;margin-bottom:8px;line-height:1.4'>{desc}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-top:8px;margin-bottom:6px;'>Sentiment</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:14px;color:#333;margin-top:0;margin-bottom:8px;line-height:1.4'>{sentiment_desc}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-top:8px;margin-bottom:6px;'>Confidence Level</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:16px;color:#333;margin-top:0;margin-bottom:8px; font-weight:600'>{conf:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-top:8px;margin-bottom:6px;'>Recommendation (Beginner-friendly)</h3>", unsafe_allow_html=True)
        st.markdown(f"<pre style='white-space:pre-wrap;font-size:14px;color:#222;background:transparent;border:none;padding:0;margin:0'>{rec_text}</pre>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#0B6E4F;margin-top:8px;margin-bottom:6px;'>Risk Controls (1-day)</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:14px;color:#333;margin-top:0;margin-bottom:8px;'>Historical VaR(95%): {hist_var_95:.4f} | Parametric VaR(95%): {param_var_95:.4f} | CVaR(95%): {cvar_95:.4f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

