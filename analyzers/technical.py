# ============================================================
# indicators.py -- Technical Indicators
# ============================================================
# 9 indicators grouped by category:
#   Momentum   : RSI, Stochastic Oscillator, Rate of Change
#   Trend      : MACD, Moving Averages, Golden/Death Cross
#   Volatility : Bollinger Bands, ATR
#   Volume     : Volume Trend
# ============================================================

import pandas as pd

from core.config import (
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT,
    STOCH_PERIOD, STOCH_OVERSOLD, STOCH_OVERBOUGHT,
    ROC_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    MA_SHORT, MA_MID, MA_LONG,
    BB_PERIOD, BB_STD,
    ATR_PERIOD,
    VOLUME_RISING, VOLUME_FALLING, VOLUME_LOOKBACK,
)


# ============================================================
# Momentum Indicators
# ============================================================

def compute_rsi(closes: pd.Series) -> dict:
    """
    Relative Strength Index
    Measures speed and magnitude of price movements.
    Scale: 0 to 100
    Below 30 = oversold (bullish), above 70 = overbought (bearish)
    """
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs    = gain / loss.replace(0, float("inf"))
    rsi   = round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)

    if rsi < RSI_OVERSOLD:
        signal, icon = "oversold -- strong buy", "✅"
    elif rsi < 45:
        signal, icon = "mildly oversold -- buy", "✅"
    elif rsi > RSI_OVERBOUGHT:
        signal, icon = "overbought -- caution", "⚠️"
    elif rsi > 55:
        signal, icon = "mildly overbought -- watch", "⚠️"
    else:
        signal, icon = "neutral", "➡️"

    return {"value": rsi, "signal": signal, "icon": icon}


def compute_stochastic(df: pd.DataFrame) -> dict:
    """
    Stochastic Oscillator
    Compares closing price to high-low range over a period.
    Scale: 0 to 100
    Below 20 = oversold (bullish), above 80 = overbought (bearish)
    """
    low_min  = df["Low"].rolling(STOCH_PERIOD).min()
    high_max = df["High"].rolling(STOCH_PERIOD).max()
    k_range  = high_max - low_min
    k        = round(float(
        (((df["Close"] - low_min) / k_range.replace(0, float("inf")))
         * 100).iloc[-1]
    ), 2)

    if k < STOCH_OVERSOLD:
        signal, icon = "oversold -- strong buy", "✅"
    elif k < 35:
        signal, icon = "mildly oversold -- buy", "✅"
    elif k > STOCH_OVERBOUGHT:
        signal, icon = "overbought -- caution", "⚠️"
    elif k > 65:
        signal, icon = "mildly overbought -- watch", "⚠️"
    else:
        signal, icon = "neutral", "➡️"

    return {"value": k, "signal": signal, "icon": icon}


def compute_roc(closes: pd.Series) -> dict:
    """
    Rate of Change
    Measures percentage price change over a set period.
    Positive = bullish momentum, Negative = bearish momentum
    """
    roc = round(float(
        ((closes.iloc[-1] - closes.iloc[-ROC_PERIOD])
         / closes.iloc[-ROC_PERIOD]) * 100
    ), 2)

    if roc > 5:
        signal, icon = "strong positive momentum", "✅"
    elif roc > 0:
        signal, icon = "mild positive momentum", "✅"
    elif roc < -5:
        signal, icon = "strong negative momentum", "⚠️"
    else:
        signal, icon = "mild negative momentum", "⚠️"

    return {"value": roc, "signal": signal, "icon": icon}


# ============================================================
# Trend Indicators
# ============================================================

def compute_macd(closes: pd.Series) -> dict:
    """
    Moving Average Convergence Divergence
    Identifies trend shifts via EMA crossovers.
    MACD above signal = bullish, below = bearish
    """
    ema_fast  = closes.ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow  = closes.ewm(span=MACD_SLOW,   adjust=False).mean()
    macd      = ema_fast - ema_slow
    signal    = macd.ewm(span=MACD_SIGNAL,   adjust=False).mean()
    histogram = macd - signal

    macd_val   = round(float(macd.iloc[-1]),      4)
    signal_val = round(float(signal.iloc[-1]),    4)
    hist_val   = round(float(histogram.iloc[-1]), 4)

    if hist_val > 0 and macd_val > signal_val:
        sig, icon = "bullish crossover", "✅"
    elif hist_val < 0 and macd_val < signal_val:
        sig, icon = "bearish crossover", "⚠️"
    else:
        sig, icon = "neutral", "➡️"

    return {
        "macd":         macd_val,
        "signal":       signal_val,
        "histogram":    hist_val,
        "signal_label": sig,
        "icon":         icon,
    }


def compute_moving_averages(closes: pd.Series,
                             current_price: float) -> dict:
    """
    Simple Moving Averages -- MA20, MA50, MA200
    Price above MA = bullish, below = bearish
    """
    results = {}
    for period, label in [
        (MA_SHORT, "ma20"),
        (MA_MID,   "ma50"),
        (MA_LONG,  "ma200")
    ]:
        if len(closes) >= period:
            ma_val = round(float(
                closes.rolling(period).mean().iloc[-1]
            ), 2)
            above  = current_price > ma_val
            results[label] = {
                "value":  ma_val,
                "above":  above,
                "signal": "price above -- bullish" if above
                          else "price below -- bearish",
                "icon":   "✅" if above else "⚠️",
            }
        else:
            results[label] = {
                "value":  None,
                "above":  None,
                "signal": "insufficient data",
                "icon":   "➡️",
            }
    return results


def compute_golden_cross(closes: pd.Series) -> dict:
    """
    Golden Cross and Death Cross
    MA50 crosses above MA200 = Golden Cross (bullish)
    MA50 crosses below MA200 = Death Cross (bearish)
    One of the most watched signals by institutional traders.
    """
    if len(closes) < MA_LONG:
        return {
            "value":  None,
            "signal": "insufficient data",
            "icon":   "➡️",
            "golden": None,
        }

    ma50  = closes.rolling(MA_MID).mean()
    ma200 = closes.rolling(MA_LONG).mean()

    ma50_now,  ma200_now  = ma50.iloc[-1],  ma200.iloc[-1]
    ma50_prev, ma200_prev = ma50.iloc[-2],  ma200.iloc[-2]

    golden_cross = (ma50_prev <= ma200_prev) and (ma50_now > ma200_now)
    death_cross  = (ma50_prev >= ma200_prev) and (ma50_now < ma200_now)

    if ma50_now > ma200_now:
        if golden_cross:
            signal, icon = "fresh golden cross -- strong buy", "✅"
        else:
            signal, icon = "golden cross active -- bullish", "✅"
    else:
        if death_cross:
            signal, icon = "fresh death cross -- strong sell", "⚠️"
        else:
            signal, icon = "death cross active -- bearish", "⚠️"

    return {
        "value":  round(float(ma50_now), 2),
        "ma200":  round(float(ma200_now), 2),
        "signal": signal,
        "icon":   icon,
        "golden": ma50_now > ma200_now,
    }


# ============================================================
# Volatility Indicators
# ============================================================

def compute_bollinger_bands(closes: pd.Series,
                             current_price: float) -> dict:
    """
    Bollinger Bands
    Price channel based on standard deviation from MA20.
    Price near lower band = potential buy
    Price near upper band = potential sell
    """
    ma    = closes.rolling(BB_PERIOD).mean()
    std   = closes.rolling(BB_PERIOD).std()
    upper = round(float((ma + BB_STD * std).iloc[-1]), 2)
    mid   = round(float(ma.iloc[-1]), 2)
    lower = round(float((ma - BB_STD * std).iloc[-1]), 2)

    bb_range = upper - lower
    bb_pct   = round(
        float((current_price - lower) / bb_range * 100)
        if bb_range > 0 else 50, 2
    )

    # Clamp to 0-100 range to prevent overflow
    bb_pct = max(0, min(100, bb_pct))

    if bb_pct < 20:
        signal, icon = "near lower band -- potential bounce", "✅"
    elif bb_pct > 80:
        signal, icon = "near upper band -- potential pullback", "⚠️"
    else:
        signal, icon = "mid band -- neutral", "➡️"

    return {
        "upper":  upper,
        "middle": mid,
        "lower":  lower,
        "pct":    bb_pct,
        "signal": signal,
        "icon":   icon,
    }


def compute_atr(df: pd.DataFrame) -> dict:
    """
    Average True Range
    Measures market volatility -- how much price moves per day.
    Higher ATR = more volatile stock.
    """
    high       = df["High"]
    low        = df["Low"]
    prev_close = df["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)

    atr           = round(float(tr.rolling(ATR_PERIOD).mean().iloc[-1]), 4)
    current_price = round(float(df["Close"].iloc[-1]), 2)
    atr_pct       = round((atr / current_price) * 100, 2)

    if atr_pct > 3:
        signal, icon = "high volatility", "⚠️"
    elif atr_pct > 1.5:
        signal, icon = "moderate volatility", "➡️"
    else:
        signal, icon = "low volatility", "✅"

    return {
        "value":  atr,
        "pct":    atr_pct,
        "signal": signal,
        "icon":   icon,
    }


# ============================================================
# Volume Indicator
# ============================================================

def compute_volume_trend(df: pd.DataFrame) -> dict:
    """
    Volume Trend
    Compares recent volume to historical average.
    Rising volume confirms price moves.
    Falling volume suggests weak conviction.
    """
    avg_vol    = float(df["Volume"].mean())
    recent_vol = float(df["Volume"].tail(VOLUME_LOOKBACK).mean())
    ratio      = round(recent_vol / avg_vol, 2) if avg_vol > 0 else 1.0

    if ratio > VOLUME_RISING:
        signal, icon = "rising -- confirms price move", "✅"
    elif ratio < VOLUME_FALLING:
        signal, icon = "falling -- weak conviction", "⚠️"
    else:
        signal, icon = "stable", "➡️"

    return {
        "avg_volume":    round(avg_vol, 0),
        "recent_volume": round(recent_vol, 0),
        "ratio":         ratio,
        "signal":        signal,
        "icon":          icon,
    }
