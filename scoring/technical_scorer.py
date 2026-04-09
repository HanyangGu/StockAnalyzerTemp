# ============================================================
# technical_scorer.py -- Technical Analysis Scoring Engine
# ============================================================
# Scores all 9 technical indicators across 3 time horizons:
#   Short term  : RSI + Stochastic + ROC + Bollinger Bands
#   Mid term    : MACD + MA20 + MA50 + ATR + Volume
#   Long term   : MA200 + Golden Cross + Volume Trend
#   Overall     : Weighted average (30% / 35% / 35%)
# ============================================================

from core.config import (
    BASELINE_SCORE,
    RSI_OVERSOLD, RSI_OVERBOUGHT,
    STOCH_OVERSOLD, STOCH_OVERBOUGHT,
    VOLUME_RISING, VOLUME_FALLING,
)


# ============================================================
# Utility Functions
# ============================================================

def get_technical_verdict(score: int) -> tuple:
    """
    Converts a numerical score into a momentum verdict label and emoji.
    Labels describe what price is doing, not a direct buy/sell instruction.
    Returns: (verdict_label, emoji)
    """
    if score >= 75:
        return "Strong Uptrend",   "🟢"
    elif score >= 60:
        return "Uptrend",          "🟩"
    elif score >= 40:
        return "Neutral",          "⬜"
    elif score >= 25:
        return "Downtrend",        "🟥"
    else:
        return "Strong Downtrend", "🔴"


def check_downtrend(mas: dict) -> tuple:
    """
    Checks structural downtrend based on Moving Average positioning.
    MA200 is the key dividing line between structural breakdown
    and temporary pullback.

    Returns: (is_downtrend, penalty, signal_label)
    """
    ma20_below  = mas["ma20"]["above"]  is False
    ma50_below  = mas["ma50"]["above"]  is False
    ma200_below = mas["ma200"]["above"] is False

    # All 3 MAs below -- severe structural downtrend
    if ma20_below and ma50_below and ma200_below:
        return True, -30, \
            "price below MA20+MA50+MA200 -- structural downtrend 🔴"

    # Below MA200 -- long term structural breakdown
    elif ma200_below:
        return True, -20, \
            "price below MA200 -- long term breakdown ⚠️"

    # Below MA20+MA50 but ABOVE MA200 -- temporary pullback
    elif ma20_below and ma50_below and not ma200_below:
        return False, 0, \
            "short term pullback -- long term uptrend intact ✅"

    # Below MA20 only -- minor dip
    elif ma20_below and not ma50_below and not ma200_below:
        return False, 0, \
            "minor pullback -- no structural concern ✅"

    # All above -- confirmed uptrend
    else:
        return False, 0, \
            "price above all MAs -- uptrend confirmed ✅"


# ============================================================
# Time Horizon Scorers
# ============================================================

def score_short_term(rsi: dict, stoch: dict,
                     roc: dict, bb: dict,
                     mas: dict) -> dict:
    """
    Short term score (0-100)
    RSI (25pts) + Stochastic (20pts) + ROC (20pts)
    + Bollinger Bands (15pts) + Downtrend penalty
    """
    score   = BASELINE_SCORE
    signals = []

    # Relative Strength Index (max 25pts)
    if rsi["value"] < RSI_OVERSOLD:
        score += 25
    elif rsi["value"] < 45:
        score += 18
    elif rsi["value"] > RSI_OVERBOUGHT:
        score += 5
    elif rsi["value"] > 55:
        score += 8
    else:
        score += 12
    signals.append(
        f"RSI ({rsi['value']}) -- {rsi['signal']} {rsi['icon']}"
    )

    # Stochastic Oscillator (max 20pts)
    if stoch["value"] < STOCH_OVERSOLD:
        score += 20
    elif stoch["value"] < 35:
        score += 14
    elif stoch["value"] > STOCH_OVERBOUGHT:
        score += 4
    elif stoch["value"] > 65:
        score += 7
    else:
        score += 10
    signals.append(
        f"Stochastic ({stoch['value']}) -- "
        f"{stoch['signal']} {stoch['icon']}"
    )

    # Rate of Change (max 20pts)
    if roc["value"] > 5:
        score += 20
    elif roc["value"] > 0:
        score += 12
    elif roc["value"] < -5:
        score += 0
    else:
        score += 4
    signals.append(
        f"ROC ({roc['value']}%) -- {roc['signal']} {roc['icon']}"
    )

    # Bollinger Bands (max 15pts)
    if bb["pct"] < 20:
        score += 15
    elif bb["pct"] > 80:
        score += 3
    else:
        score += 8
    signals.append(
        f"Bollinger Bands ({bb['pct']}%) -- "
        f"{bb['signal']} {bb['icon']}"
    )

    # Structural downtrend penalty
    is_downtrend, penalty, dt_signal = check_downtrend(mas)
    if penalty != 0:
        score += penalty
        signals.append(f"Trend structure -- {dt_signal}")

    score         = max(0, min(100, score))
    verdict, icon = get_technical_verdict(score)

    return {
        "score":   score,
        "verdict": verdict,
        "icon":    icon,
        "signals": signals,
    }


def score_mid_term(macd: dict, mas: dict,
                   atr: dict, vol: dict) -> dict:
    """
    Mid term score (0-100)
    MACD (30pts) + MA20 (15pts) + MA50 (20pts)
    + ATR (10pts) + Volume (15pts) + Downtrend penalty
    """
    score   = BASELINE_SCORE
    signals = []

    # Moving Average Convergence Divergence (max 30pts)
    if macd["histogram"] > 0 and macd["macd"] > macd["signal"]:
        score += 30
    elif macd["histogram"] < 0 and macd["macd"] < macd["signal"]:
        score += 0
    else:
        score += 12
    signals.append(
        f"MACD ({macd['histogram']}) -- "
        f"{macd['signal_label']} {macd['icon']}"
    )

    # 20-Day Moving Average (max 15pts)
    if mas["ma20"]["above"] is True:
        score += 15
    elif mas["ma20"]["above"] is False:
        # Partial credit if above MA200 (temporary pullback)
        if mas["ma200"]["above"] is True:
            score += 5
        else:
            score += 0
    signals.append(
        f"MA20 (${mas['ma20']['value']}) -- "
        f"{mas['ma20']['signal']} {mas['ma20']['icon']}"
    )

    # 50-Day Moving Average (max 20pts)
    if mas["ma50"]["above"] is True:
        score += 20
    elif mas["ma50"]["above"] is False:
        # Partial credit if above MA200 (temporary pullback)
        if mas["ma200"]["above"] is True:
            score += 7
        else:
            score += 0
    signals.append(
        f"MA50 (${mas['ma50']['value']}) -- "
        f"{mas['ma50']['signal']} {mas['ma50']['icon']}"
    )

    # Average True Range (max 10pts)
    if atr["pct"] < 1.5:
        score += 10
    elif atr["pct"] < 3:
        score += 6
    else:
        score += 2
    signals.append(
        f"ATR (${atr['value']} / {atr['pct']}%) -- "
        f"{atr['signal']} {atr['icon']}"
    )

    # Volume Trend (max 15pts)
    if vol["ratio"] > VOLUME_RISING:
        score += 15
    elif vol["ratio"] < VOLUME_FALLING:
        score += 0
    else:
        score += 7
    signals.append(
        f"Volume ({vol['ratio']}x avg) -- "
        f"{vol['signal']} {vol['icon']}"
    )

    # Structural downtrend penalty
    is_downtrend, penalty, dt_signal = check_downtrend(mas)
    if penalty != 0:
        score += penalty
        signals.append(f"Trend structure -- {dt_signal}")

    score         = max(0, min(100, score))
    verdict, icon = get_technical_verdict(score)

    return {
        "score":   score,
        "verdict": verdict,
        "icon":    icon,
        "signals": signals,
    }


def score_long_term(mas: dict, golden: dict, vol: dict) -> dict:
    """
    Long term score (0-100)
    MA200 (35pts) + Golden/Death Cross (40pts)
    + Volume Trend (15pts) + Downtrend penalty
    """
    score   = BASELINE_SCORE
    signals = []

    # 200-Day Moving Average (max 35pts)
    if mas["ma200"]["above"] is True:
        score += 35
    elif mas["ma200"]["above"] is False:
        score += 0
    signals.append(
        f"MA200 (${mas['ma200']['value']}) -- "
        f"{mas['ma200']['signal']} {mas['ma200']['icon']}"
    )

    # Golden Cross and Death Cross (max 40pts)
    if golden["value"] is not None:
        if golden["golden"] and "fresh" in golden["signal"]:
            score += 40
        elif golden["golden"]:
            score += 28
        elif not golden["golden"] and "fresh" in golden["signal"]:
            score += 0
        else:
            score += 5
    signals.append(
        f"Golden/Death Cross -- "
        f"{golden['signal']} {golden['icon']}"
    )

    # Volume Trend (max 15pts)
    if vol["ratio"] > VOLUME_RISING:
        score += 15
    elif vol["ratio"] < VOLUME_FALLING:
        score += 0
    else:
        score += 7
    signals.append(
        f"Volume ({vol['ratio']}x avg) -- "
        f"{vol['signal']} {vol['icon']}"
    )

    # Structural downtrend penalty
    is_downtrend, penalty, dt_signal = check_downtrend(mas)
    if penalty != 0:
        score += penalty
        signals.append(f"Trend structure -- {dt_signal}")

    score         = max(0, min(100, score))
    verdict, icon = get_technical_verdict(score)

    return {
        "score":   score,
        "verdict": verdict,
        "icon":    icon,
        "signals": signals,
    }


# ============================================================
# Technical Overall Score
# ============================================================

def score_technical_overall(short: dict, mid: dict,
                             long: dict) -> dict:
    """
    Combines short, mid, long term scores into one technical score.
    Weights: Short 30% | Mid 35% | Long 35%
    """
    overall_score = round(
        (short["score"] * 0.30) +
        (mid["score"]   * 0.35) +
        (long["score"]  * 0.35)
    )
    overall_score         = max(0, min(100, overall_score))
    overall_verdict, icon = get_technical_verdict(overall_score)

    return {
        "score":   overall_score,
        "verdict": overall_verdict,
        "icon":    icon,
    }
