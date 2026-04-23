# indicator_engine.py
import numpy as np
import pandas as pd


def ema(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).mean().values


def wma(series, period):
    weights = np.arange(1, period + 1)
    return pd.Series(series).rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).values


def sma(series, period):
    return pd.Series(series).rolling(period).mean().values


def rsi(close, period=14):
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return (100 - 100 / (1 + rs)).values


def macd(close, fast=12, slow=26, signal=9):
    e_fast = pd.Series(close).ewm(span=fast, adjust=False).mean()
    e_slow = pd.Series(close).ewm(span=slow, adjust=False).mean()
    macd_line = (e_fast - e_slow).values
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def hull_ma(close, period=16):
    half = round(period / 2)
    sqn = round(np.sqrt(period))
    n2 = 2 * pd.Series(wma(close, half)) - pd.Series(wma(close, period))
    hull = pd.Series(wma(n2.values, sqn))
    return hull.values


def bollinger(close, period=20, mult=2.0):
    mid = sma(close, period)
    std = pd.Series(close).rolling(period).std().values
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower


def atr(high, low, close, period=14):
    tr = np.maximum(
        np.maximum(
            np.array(high) - np.array(low),
            np.abs(np.array(high) - np.concatenate([[close[0]], close[:-1]])),
        ),
        np.abs(np.array(low) - np.concatenate([[close[0]], close[:-1]])),
    )
    return pd.Series(tr).ewm(com=period - 1, adjust=False).mean().values


def supertrend(high, low, close, factor=3.0, period=10):
    atr_vals = atr(high, low, close, period)
    hl2 = (np.array(high) + np.array(low)) / 2
    upper_band = hl2 + factor * atr_vals
    lower_band = hl2 - factor * atr_vals
    direction = np.ones(len(close))  # 1 = uptrend
    for i in range(1, len(close)):
        if close[i] > upper_band[i - 1]:
            direction[i] = 1
        elif close[i] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return direction


def stochastic(high, low, close, k_period=14, smooth=3, d_period=3):
    h = pd.Series(high).rolling(k_period).max()
    l = pd.Series(low).rolling(k_period).min()
    k_raw = 100 * (pd.Series(close) - l) / (h - l + 1e-10)
    k = k_raw.rolling(smooth).mean()
    d = k.rolling(d_period).mean()
    return k.values, d.values


def obv(close, volume):
    direction = np.sign(np.diff(close, prepend=close[0]))
    return np.cumsum(direction * volume)


def dema(series, period):
    e1 = ema(series, period)
    e2 = ema(e1, period)
    return 2 * np.array(e1) - np.array(e2)


def dema_macd(close, fast=12, slow=26, signal=9):
    d_fast = dema(close, fast)
    d_slow = dema(close, slow)
    line = d_fast - d_slow
    sig = dema(line, signal)
    hist = line - sig
    return line, sig, hist


def knn_score(close, rsi_vals, macd_hist_vals, bb_pct_b_vals, atr_vals, k=5, lookback=100):
    scores = []
    n = len(close)
    for i in range(n):
        if i < lookback + 10:
            scores.append(0.0)
            continue
        f1_now = rsi_vals[i] / 100.0
        f2_now = macd_hist_vals[i] / (atr_vals[i] + 1e-10)
        f3_now = bb_pct_b_vals[i]
        distances, votes = [], []
        for j in range(i - lookback, i - 10):
            f1 = rsi_vals[j] / 100.0
            f2 = macd_hist_vals[j] / (atr_vals[j] + 1e-10)
            f3 = bb_pct_b_vals[j]
            dist = np.sqrt((f1_now - f1) ** 2 + (f2_now - f2) ** 2 + (f3_now - f3) ** 2)
            future_ret = close[j + 5] - close[j] if j + 5 < n else 0
            distances.append(dist)
            votes.append(1 if future_ret > 0 else -1)
        idx_sorted = np.argsort(distances)[:k]
        knn_vote = sum(votes[m] for m in idx_sorted) / k * 100
        scores.append(knn_vote)
    return np.array(scores)


def compute_score(df: pd.DataFrame, params: dict = None) -> dict:
    """
    df: OHLCV DataFrame (columns: open, high, low, close, volume)
    params: indikatör parametreleri (None ise default)
    Döner: {"score": float, "signal": str, "components": dict}
    """
    if params is None:
        params = {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # Parametreler
    ema_fast_p = params.get("ema_fast", 9)
    ema_mid_p = params.get("ema_mid", 21)
    ema_slow_p = params.get("ema_slow", 50)
    ema_trend_p = params.get("ema_trend", 200)
    rsi_len_p = params.get("rsi_len", 14)
    rsi_ob = params.get("rsi_ob", 70)
    rsi_os = params.get("rsi_os", 30)
    macd_fast_p = params.get("macd_fast", 12)
    macd_slow_p = params.get("macd_slow", 26)
    macd_sig_p = params.get("macd_signal", 9)
    stoch_k_p = params.get("stoch_k", 14)
    stoch_d_p = params.get("stoch_d", 3)
    stoch_sm_p = params.get("stoch_smooth", 3)
    bb_len_p = params.get("bb_len", 20)
    bb_mult_p = params.get("bb_mult", 2.0)
    atr_len_p = params.get("atr_len", 14)
    vol_ma_len_p = params.get("vol_ma_len", 20)
    w_trend = params.get("w_trend", 25)
    w_momentum = params.get("w_momentum", 20)
    w_volatility = params.get("w_volatility", 15)
    w_volume = params.get("w_volume", 15)
    w_sr = params.get("w_sr", 10)
    w_ml = params.get("w_ml", 10)
    w_mtf = params.get("w_mtf", 5)
    strong_thresh = params.get("strong_thresh", 50)

    n = len(close)
    if n < 210:
        return {"score": 0, "signal": "BEKLE", "components": {}}

    # ── 1. TREND ──────────────────────────────────────────────────────────────
    e9 = ema(close, ema_fast_p)
    e21 = ema(close, ema_mid_p)
    e50 = ema(close, ema_slow_p)
    e200 = ema(close, ema_trend_p)
    st_dir = supertrend(high, low, close)

    ema_bull = 1.0 if (e9[-1] > e21[-1] > e50[-1] > e200[-1]) else 0.0
    ema_bear = -1.0 if (e9[-1] < e21[-1] < e50[-1] < e200[-1]) else 0.0
    ema_align = ema_bull + ema_bear
    pve200 = 1.0 if close[-1] > e200[-1] else -1.0
    pve50 = 1.0 if close[-1] > e50[-1] else -1.0
    golden = 1.0 if (e50[-1] > e200[-1] and e50[-2] <= e200[-2]) else 0.0
    death = -1.0 if (e50[-1] < e200[-1] and e50[-2] >= e200[-2]) else 0.0
    st_sig = 1.0 if st_dir[-1] == 1 else -1.0
    trend_score = ema_align * 30 + pve200 * 20 + pve50 * 15 + (golden + death) * 15 + st_sig * 20

    # ── 2. MOMENTUM ───────────────────────────────────────────────────────────
    rsi_vals = rsi(close, rsi_len_p)
    rv = rsi_vals[-1]
    if rv > rsi_ob:
        rsi_score = -((rv - rsi_ob) / (100 - rsi_ob)) * 100
    elif rv < rsi_os:
        rsi_score = ((rsi_os - rv) / rsi_os) * 100
    else:
        rsi_score = ((rv - 50) / 50) * 50

    macd_l, macd_s, macd_h = macd(close, macd_fast_p, macd_slow_p, macd_sig_p)
    bull_cross = macd_l[-1] > macd_s[-1] and macd_l[-2] <= macd_s[-2]
    bear_cross = macd_l[-1] < macd_s[-1] and macd_l[-2] >= macd_s[-2]
    if bull_cross:
        macd_score = 100.0
    elif bear_cross:
        macd_score = -100.0
    else:
        macd_score = 50.0 if macd_l[-1] > macd_s[-1] else -50.0
        macd_score += 25.0 if macd_h[-1] > macd_h[-2] else -25.0

    stk, std = stochastic(high, low, close, stoch_k_p, stoch_sm_p, stoch_d_p)
    sv = stk[-1]
    stoch_score = -(sv - 80) / 20 * 100 if sv > 80 else (20 - sv) / 20 * 100 if sv < 20 else (sv - 50) / 50 * 60
    if stk[-1] < 30 and stk[-1] > stk[-2] and stk[-2] <= std[-2]:
        stoch_score = 100.0
    elif stk[-1] > 70 and stk[-1] < stk[-2] and stk[-2] >= std[-2]:
        stoch_score = -100.0

    momentum_score = rsi_score * 0.35 + macd_score * 0.40 + stoch_score * 0.25

    # ── 3. VOLATİLİTE ─────────────────────────────────────────────────────────
    bb_mid_v, bb_up, bb_lo = bollinger(close, bb_len_p, bb_mult_p)
    bb_pct_b = (close[-1] - bb_lo[-1]) / (bb_up[-1] - bb_lo[-1] + 1e-10)
    bb_score = -80.0 if bb_pct_b > 1.0 else 80.0 if bb_pct_b < 0.0 else (bb_pct_b - 0.5) * 160
    volatility_score = bb_score

    atr_vals = atr(high, low, close, atr_len_p)

    # ── 4. HACİM ──────────────────────────────────────────────────────────────
    vol_ma_v = sma(volume, vol_ma_len_p)
    vol_ratio = volume[-1] / (vol_ma_v[-1] + 1e-10)
    obv_vals = obv(close, volume)
    obv_ma_v = sma(obv_vals, vol_ma_len_p)
    vwap = np.sum(close[-vol_ma_len_p:] * volume[-vol_ma_len_p:]) / (np.sum(volume[-vol_ma_len_p:]) + 1e-10)

    vol_confirm_bull = 40.0 if (close[-1] > close[-2] and vol_ratio > 1.2) else 0.0
    vol_confirm_bear = -40.0 if (close[-1] < close[-2] and vol_ratio > 1.2) else 0.0
    obv_trend = 30.0 if obv_vals[-1] > obv_ma_v[-1] else -30.0
    vwap_sig = 30.0 if close[-1] > vwap else -30.0
    volume_score = vol_confirm_bull + vol_confirm_bear + obv_trend + vwap_sig

    # ── 5. DESTEK/DİRENÇ ──────────────────────────────────────────────────────
    window = 10
    pivot_highs = [i for i in range(window, n - window) if high[i] == max(high[i - window:i + window + 1])]
    pivot_lows = [i for i in range(window, n - window) if low[i] == min(low[i - window:i + window + 1])]
    last_res = high[pivot_highs[-1]] if pivot_highs else None
    last_sup = low[pivot_lows[-1]] if pivot_lows else None

    if last_res and last_sup:
        d_res = (last_res - close[-1]) / close[-1] * 100
        d_sup = (close[-1] - last_sup) / close[-1] * 100
        if d_res < 1.0:
            sr_score = -60.0
        elif d_sup < 1.0:
            sr_score = 60.0
        else:
            sr_score = (d_res - d_sup) / (d_res + d_sup + 0.001) * 100
    else:
        sr_score = 0.0

    # ── 6. kNN ML ─────────────────────────────────────────────────────────────
    bb_pct_b_arr = (close - bb_lo) / (bb_up - bb_lo + 1e-10)
    knn_scores = knn_score(close, rsi_vals, macd_h, bb_pct_b_arr, atr_vals)
    ml_score = float(knn_scores[-1])

    # ── 7. MTF (günlük üzerine haftalık EMA bakışı) ────────────────────────────
    mtf_score = 0.0  # yfinance ile aynı sembolden haftalık veri çekilecek (bot.py'de)

    # ── BİRLEŞİK SKOR ─────────────────────────────────────────────────────────
    total_w = w_trend + w_momentum + w_volatility + w_volume + w_sr + w_ml + w_mtf
    total_w = total_w if total_w > 0 else 100
    mc_raw = (
        trend_score * w_trend
        + momentum_score * w_momentum
        + volatility_score * w_volatility
        + volume_score * w_volume
        + sr_score * w_sr
        + ml_score * w_ml
        + mtf_score * w_mtf
    ) / total_w

    # DEMA MACD
    dema_l, dema_s, dema_h = dema_macd(close)
    dema_bull_x = dema_l[-1] > dema_s[-1] and dema_l[-2] <= dema_s[-2]
    dema_bear_x = dema_l[-1] < dema_s[-1] and dema_l[-2] >= dema_s[-2]
    if dema_bull_x:
        dema_score = 100.0
    elif dema_bear_x:
        dema_score = -100.0
    else:
        dema_score = 50.0 if dema_l[-1] > dema_s[-1] else -50.0
        dema_score += 25.0 if dema_h[-1] > dema_h[-2] else -25.0

    htf_dema_bull = dema_l[-1] > dema_s[-1]  # simplification; bot.py injects weekly

    prediction_score = mc_raw * 0.55 + dema_score * 0.25 + 0.0 * 0.10 + (40.0 if htf_dema_bull else -40.0) * 0.10
    prediction_score = max(-100.0, min(100.0, prediction_score))

    # Smooth (3-bar EMA emülasyonu — tek bar; yeterli)
    smooth = prediction_score

    if smooth >= strong_thresh:
        signal = "AL"
    elif smooth <= -strong_thresh:
        signal = "SAT"
    else:
        signal = "BEKLE"

    return {
        "score": round(smooth, 2),
        "signal": signal,
        "components": {
            "trend": round(trend_score, 1),
            "momentum": round(momentum_score, 1),
            "volatility": round(volatility_score, 1),
            "volume": round(volume_score, 1),
            "sr": round(sr_score, 1),
            "ml": round(ml_score, 1),
            "dema": round(dema_score, 1),
            "rsi": round(rv, 1),
            "atr": round(float(atr_vals[-1]), 4),
        },
    }
