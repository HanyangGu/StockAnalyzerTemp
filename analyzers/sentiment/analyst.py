# ============================================================
# analyzers/sentiment/analyst.py -- Analyst Ratings Analyzer
# ============================================================
# Fetches analyst ratings and evaluates across 8 dimensions:
#
#   1. Rating Distribution  : weighted Strong Buy→Strong Sell ratio
#   2. Analyst Authority    : GPT judges firm tier (top/major/general)
#   3. Time Decay           : newer ratings weighted higher
#   4. Rating Momentum      : upgrades vs downgrades trend
#   5. Price Target Gap     : analyst mean target vs current price
#   6. Price Target Dispersion : high/low target spread (uncertainty)
#   7. Initiation Coverage  : first-time coverage signals
#   8. Rating Consensus     : agreement level across analysts
#
# Data sources (all via yfinance):
#   ticker.recommendations        -- historical rating records
#   ticker.recommendations_summary -- current distribution counts
#   ticker.analyst_price_targets  -- mean/high/low/median targets
#
# Data strategy:
#   Primary window : last 6 months
#   Fallback       : extend to 12 months if < 5 ratings found
#   Minimum        : if still < 3, return low-confidence neutral
# ============================================================

import json
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf
from openai import OpenAI


# ============================================================
# Constants
# ============================================================

PRIMARY_MONTHS    = 6
FALLBACK_MONTHS   = 12
MIN_RATINGS       = 3
MIN_FOR_CONSENSUS = 5

RATING_SCORES = {
    "strong buy":     2.0,
    "buy":            1.0,
    "hold":           0.0,
    "sell":          -1.0,
    "strong sell":   -2.0,
    "outperform":     1.0,
    "overweight":     1.0,
    "underperform":  -1.0,
    "underweight":   -1.0,
    "neutral":        0.0,
    "market perform": 0.0,
    "equal weight":   0.0,
    "sector perform": 0.0,
    "sector weight":  0.0,
}

TIME_DECAY_ANALYST = [
    (1,  1.0),
    (3,  0.8),
    (6,  0.6),
    (12, 0.3),
]
TIME_DECAY_DEFAULT = 0.1

AUTHORITY_WEIGHTS = {
    "top":     2.0,
    "major":   1.2,
    "general": 0.7,
}

TARGET_GAP_THRESHOLDS = [
    (0.30, 20),
    (0.20, 16),
    (0.10, 10),
    (0.05,  5),
    (0.00,  2),
]

CONSENSUS_BOOST_A   = 1.15
CONSENSUS_PENALTY_A = 0.85


# ============================================================
# GPT Prompt
# ============================================================

AUTHORITY_PROMPT = """
You are a financial analyst authority classifier.

Given a list of analyst firm names, classify each one and return
ONLY a JSON array. No preamble, no markdown, no explanation.

For each firm return:
{
  "firm": "<firm name exactly as given>",
  "tier": "top" | "major" | "general"
}

Tier definitions:
- top     : Bulge bracket banks and top-tier research firms
             (Goldman Sachs, Morgan Stanley, JPMorgan, Bank of America,
              Citigroup, Wells Fargo, UBS, Barclays, Deutsche Bank,
              HSBC, Jefferies, Piper Sandler, Needham, Raymond James,
              Oppenheimer, Wedbush, Mizuho, Evercore, Bernstein)
- major   : Well-known regional or mid-tier firms with established track records
- general : Smaller, boutique, or less well-known research firms

Return ONLY the JSON array. No other text.
"""


# ============================================================
# Helpers
# ============================================================

def _analyst_time_weight(date: pd.Timestamp) -> float:
    """Time decay weight for an analyst rating by age in months."""
    try:
        now = datetime.now(timezone.utc)
        if date.tzinfo is None:
            date = date.tz_localize("UTC")
        age_months = (now - date.to_pydatetime()).days / 30.0
        for max_months, weight in TIME_DECAY_ANALYST:
            if age_months <= max_months:
                return weight
        return TIME_DECAY_DEFAULT
    except Exception:
        return TIME_DECAY_DEFAULT


def _grade_to_score(grade: str) -> float:
    """Converts rating string to numeric score."""
    normalised = grade.lower().strip()
    if normalised in RATING_SCORES:
        return RATING_SCORES[normalised]
    for key, val in RATING_SCORES.items():
        if key in normalised or normalised in key:
            return val
    return 0.0


# ============================================================
# Data Fetchers
# ============================================================

def _fetch_recommendations(ticker_symbol: str) -> pd.DataFrame:
    """Fetches historical recommendations with primary/fallback windows."""
    try:
        time.sleep(1)
        recs = yf.Ticker(ticker_symbol).recommendations
        if recs is None or recs.empty:
            return pd.DataFrame()

        recs.index = pd.to_datetime(recs.index, utc=True)
        recs = recs.sort_index(ascending=False)

        cutoff  = datetime.now(timezone.utc) - timedelta(days=PRIMARY_MONTHS * 30)
        primary = recs[recs.index >= cutoff]
        if len(primary) >= MIN_RATINGS:
            return primary

        cutoff = datetime.now(timezone.utc) - timedelta(days=FALLBACK_MONTHS * 30)
        return recs[recs.index >= cutoff]

    except Exception as e:
        print(f"  Recommendations fetch error: {e}")
        return pd.DataFrame()


def _fetch_summary(ticker_symbol: str) -> dict:
    """Fetches current analyst rating distribution."""
    try:
        time.sleep(0.5)
        summary = yf.Ticker(ticker_symbol).recommendations_summary
        if summary is None or summary.empty:
            return {}
        row = summary.iloc[0]
        return {
            "strong_buy":  int(row.get("strongBuy",  0)),
            "buy":         int(row.get("buy",         0)),
            "hold":        int(row.get("hold",        0)),
            "sell":        int(row.get("sell",        0)),
            "strong_sell": int(row.get("strongSell",  0)),
        }
    except Exception as e:
        print(f"  Summary fetch error: {e}")
        return {}


def _fetch_price_targets(ticker_symbol: str) -> dict:
    """Fetches analyst price targets."""
    try:
        time.sleep(0.5)
        targets = yf.Ticker(ticker_symbol).analyst_price_targets
        if targets is None:
            return {}
        if hasattr(targets, "to_dict"):
            targets = targets.to_dict()
        return {
            "mean":   targets.get("mean"),
            "high":   targets.get("high"),
            "low":    targets.get("low"),
            "median": targets.get("median"),
        }
    except Exception as e:
        print(f"  Price targets fetch error: {e}")
        return {}


def _fetch_current_price(ticker_symbol: str) -> float | None:
    """Fetches current stock price."""
    try:
        info = yf.Ticker(ticker_symbol).info
        return info.get("currentPrice") or info.get("regularMarketPrice")
    except Exception:
        return None


# ============================================================
# GPT Authority Classifier
# ============================================================

def _classify_authority(firms: list) -> dict:
    """Classifies analyst firms by authority tier using GPT."""
    if not firms:
        return {}
    unique = list(set(firms))
    try:
        client   = OpenAI()
        response = client.chat.completions.create(
            model       = "gpt-4.1-mini",
            max_tokens  = 800,
            temperature = 0.0,
            messages    = [
                {"role": "system", "content": AUTHORITY_PROMPT},
                {"role": "user",   "content": f"Classify: {json.dumps(unique)}"},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        results = json.loads(raw)
        return {r["firm"]: r["tier"] for r in results if "firm" in r}
    except Exception as e:
        print(f"  Authority classification error: {e}")
        return {}


# ============================================================
# Score Components
# ============================================================

def _score_distribution(summary: dict) -> tuple:
    """Rating distribution score (0-30 pts)."""
    if not summary:
        return 0, "Rating distribution: no data ➡️"

    sb = summary.get("strong_buy",  0)
    b  = summary.get("buy",         0)
    h  = summary.get("hold",        0)
    s  = summary.get("sell",        0)
    ss = summary.get("strong_sell", 0)
    total = sb + b + h + s + ss

    if total == 0:
        return 0, "Rating distribution: no ratings ➡️"

    bull = (sb * 2.0 + b * 1.0) / total
    bear = (ss * 2.0 + s * 1.0) / total
    net  = bull - bear

    if net >= 1.5:
        score, icon = 30, "✅"
        label = f"overwhelmingly bullish ({sb} Strong Buy, {b} Buy of {total})"
    elif net >= 0.8:
        score, icon = 22, "✅"
        label = f"mostly bullish ({sb} Strong Buy, {b} Buy of {total})"
    elif net >= 0.3:
        score, icon = 15, "✅"
        label = f"mildly bullish ({sb+b} buy-side of {total})"
    elif net >= -0.3:
        score, icon = 10, "➡️"
        label = f"neutral ({h} Hold dominant of {total})"
    elif net >= -0.8:
        score, icon = 4, "⚠️"
        label = f"mildly bearish ({s+ss} sell-side of {total})"
    else:
        score, icon = 0, "⚠️"
        label = f"overwhelmingly bearish ({ss} Strong Sell of {total})"

    return score, f"Rating distribution: {label} {icon}"


def _score_momentum(recs: pd.DataFrame) -> tuple:
    """Rating momentum score (0-20 pts)."""
    if recs.empty or "Action" not in recs.columns:
        return 10, "Rating momentum: no action data ➡️"

    cutoff  = datetime.now(timezone.utc) - timedelta(days=90)
    recent  = recs[recs.index >= cutoff]

    ups   = len(recent[recent["Action"].str.lower() == "up"])
    downs = len(recent[recent["Action"].str.lower() == "down"])
    inits = len(recent[recent["Action"].str.lower() == "init"])
    net   = ups - downs + (inits * 0.5)

    if net >= 3:
        score, icon = 20, "✅"
        label = f"strong upgrade momentum ({ups} upgrades, {downs} downgrades)"
    elif net >= 1:
        score, icon = 15, "✅"
        label = f"mild upgrade momentum ({ups} upgrades, {downs} downgrades)"
    elif net == 0:
        score, icon = 10, "➡️"
        label = f"neutral ({ups} upgrades, {downs} downgrades)"
    elif net >= -2:
        score, icon = 5, "⚠️"
        label = f"mild downgrade pressure ({downs} downgrades, {ups} upgrades)"
    else:
        score, icon = 0, "⚠️"
        label = f"strong downgrade pressure ({downs} downgrades, {ups} upgrades)"

    return score, f"Rating momentum: {label} {icon}"


def _score_price_target(targets: dict,
                         current_price: float | None) -> tuple:
    """Price target gap + dispersion score (0-30 pts total)."""
    signals = []
    score   = 0

    mean_t = targets.get("mean")
    high_t = targets.get("high")
    low_t  = targets.get("low")

    # Gap (0-20 pts)
    if mean_t and current_price:
        gap_pct   = (mean_t - current_price) / current_price
        gap_score = 0
        for threshold, pts in TARGET_GAP_THRESHOLDS:
            if gap_pct >= threshold:
                gap_score = pts
                break
        score += gap_score
        icon   = "✅" if gap_pct > 0.05 else ("⚠️" if gap_pct < -0.05 else "➡️")
        sign   = "+" if gap_pct >= 0 else ""
        signals.append(
            f"Price target: mean ${round(mean_t, 2)} "
            f"({sign}{round(gap_pct * 100, 1)}% vs current ${round(current_price, 2)}) {icon}"
        )
    else:
        signals.append("Price target: no data ➡️")

    # Dispersion (0-10 pts)
    if high_t and low_t and low_t > 0:
        disp = (high_t - low_t) / low_t
        if disp < 0.15:
            d_score, d_icon = 10, "✅"
            d_label = f"low dispersion (${round(low_t,0)}–${round(high_t,0)}) -- high agreement"
        elif disp < 0.35:
            d_score, d_icon = 6, "➡️"
            d_label = f"moderate dispersion (${round(low_t,0)}–${round(high_t,0)})"
        elif disp < 0.60:
            d_score, d_icon = 3, "⚠️"
            d_label = f"high dispersion (${round(low_t,0)}–${round(high_t,0)}) -- analyst disagreement"
        else:
            d_score, d_icon = 0, "⚠️"
            d_label = f"very high dispersion (${round(low_t,0)}–${round(high_t,0)}) -- very uncertain"
        score += d_score
        signals.append(f"Target dispersion: {d_label} {d_icon}")

    return score, signals


def _score_weighted_ratings(recs: pd.DataFrame,
                              authority_map: dict) -> tuple:
    """Weighted ratings by authority + time (0-20 pts)."""
    if recs.empty:
        return 10, "Weighted ratings: no data ➡️"

    grade_col = next((c for c in ["To Grade", "toGrade", "Grade"]
                      if c in recs.columns), None)
    firm_col  = next((c for c in ["Firm", "firm"]
                      if c in recs.columns), None)

    if not grade_col:
        return 10, "Weighted ratings: grade column not found ➡️"

    weighted_sum     = 0.0
    total_weight     = 0.0
    initiation_boost = 0.0

    for date, row in recs.iterrows():
        grade  = str(row.get(grade_col, ""))
        firm   = str(row.get(firm_col,  "Unknown")) if firm_col else "Unknown"
        action = str(row.get("Action",  "")).lower()

        g_score  = _grade_to_score(grade)
        t_weight = _analyst_time_weight(date)
        a_weight = AUTHORITY_WEIGHTS.get(
                       authority_map.get(firm, "general"), 0.7
                   )

        weight        = t_weight * a_weight
        weighted_sum += g_score * weight
        total_weight += weight

        if action == "init":
            tier = authority_map.get(firm, "general")
            initiation_boost += 0.3 if tier == "top" else (0.15 if tier == "major" else 0.05)

    if total_weight == 0:
        return 10, "Weighted ratings: insufficient data ➡️"

    avg   = (weighted_sum / total_weight) + initiation_boost
    score = round(10 + avg * 5)
    score = max(0, min(20, score))

    if avg >= 1.0:
        label, icon = "strongly bullish analyst consensus", "✅"
    elif avg >= 0.3:
        label, icon = "mildly bullish analyst consensus", "✅"
    elif avg >= -0.3:
        label, icon = "neutral analyst consensus", "➡️"
    elif avg >= -1.0:
        label, icon = "mildly bearish analyst consensus", "⚠️"
    else:
        label, icon = "strongly bearish analyst consensus", "⚠️"

    if initiation_boost > 0:
        label += f" (+{round(initiation_boost,2)} initiation boost)"

    return score, f"Weighted ratings: {label} {icon}"


def _score_consensus(summary: dict) -> float:
    """Consensus multiplier based on analyst agreement."""
    if not summary:
        return 1.0

    sb = summary.get("strong_buy",  0)
    b  = summary.get("buy",         0)
    h  = summary.get("hold",        0)
    s  = summary.get("sell",        0)
    ss = summary.get("strong_sell", 0)
    total = sb + b + h + s + ss

    if total < MIN_FOR_CONSENSUS:
        return 1.0

    dominant = max(sb + b, h, s + ss)
    ratio    = dominant / total

    if ratio >= 0.70:
        return CONSENSUS_BOOST_A
    elif ratio <= 0.40:
        return CONSENSUS_PENALTY_A
    else:
        return 1.0


# ============================================================
# Master Function
# ============================================================

def fetch_analyst_sentiment(ticker_symbol: str) -> dict:
    """
    Master function -- fetches analyst data and returns scored sentiment.

    Scoring breakdown (before consensus multiplier):
      Rating Distribution : 0-30 pts
      Weighted Ratings    : 0-20 pts
      Rating Momentum     : 0-20 pts
      Price Target Gap    : 0-20 pts
      Target Dispersion   : 0-10 pts
      Total               : 0-100 pts
      Consensus           : ×0.85 to ×1.15
    """
    print(f"  Fetching analyst sentiment for {ticker_symbol}...")

    # Fetch all data
    recs          = _fetch_recommendations(ticker_symbol)
    summary       = _fetch_summary(ticker_symbol)
    targets       = _fetch_price_targets(ticker_symbol)
    current_price = _fetch_current_price(ticker_symbol)
    rating_count  = len(recs)

    # Insufficient data fallback
    if rating_count < MIN_RATINGS and not summary:
        return {
            "score":        50,
            "direction":    "neutral",
            "rating_count": rating_count,
            "summary":      {},
            "targets":      {},
            "signals": [
                f"Insufficient analyst data ({rating_count} ratings) "
                f"-- neutral score applied ➡️"
            ],
        }

    # Get authority classifications
    firm_col = next((c for c in ["Firm", "firm"]
                     if not recs.empty and c in recs.columns), None)
    firms    = recs[firm_col].dropna().unique().tolist() if firm_col else []
    authority_map = _classify_authority(firms) if firms else {}

    # Score each component
    dist_score,   dist_signal    = _score_distribution(summary)
    weight_score, weight_signal  = _score_weighted_ratings(recs, authority_map)
    mom_score,    mom_signal     = _score_momentum(recs)
    target_score, target_signals = _score_price_target(targets, current_price)

    # Consensus multiplier
    consensus = _score_consensus(summary)

    # Total
    raw_score   = dist_score + weight_score + mom_score + target_score
    raw_score   = max(0, min(100, raw_score))
    final_score = round(raw_score * consensus)
    final_score = max(0, min(100, final_score))

    # Direction
    if final_score >= 60:
        direction = "bullish"
    elif final_score <= 40:
        direction = "bearish"
    else:
        direction = "neutral"

    # Signals
    signals = [dist_signal, weight_signal, mom_signal] + target_signals
    if consensus >= CONSENSUS_BOOST_A:
        signals.append("Strong analyst consensus -- high agreement ✅")
    elif consensus <= CONSENSUS_PENALTY_A:
        signals.append("Divided analyst opinions -- reduced confidence ⚠️")
    icon = "✅" if direction == "bullish" else ("⚠️" if direction == "bearish" else "➡️")
    signals.append(f"Overall analyst tone: {direction} {icon}")

    return {
        "score":        final_score,
        "direction":    direction,
        "rating_count": rating_count,
        "summary":      summary,
        "targets":      targets,
        "signals":      signals,
    }