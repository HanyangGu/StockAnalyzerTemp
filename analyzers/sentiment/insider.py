# ============================================================
# analyzers/sentiment/insider.py -- Insider Trading Analyzer
# ============================================================
# Fetches insider transactions via yfinance insider_transactions.
#
# Confirmed columns (yfinance 1.2.x):
#   Shares, Value, URL, Text, Insider, Position,
#   Transaction, Start Date, Ownership
#
# Evaluates 5 scoring dimensions:
#   1. Net Buy/Sell Ratio   : recent buy value vs sell value
#   2. Buyer/Seller Count   : number of buyers vs sellers
#   3. Transaction Size     : larger purchases = stronger signal
#   4. Position Level       : CEO/CFO buys > general director
#   5. Time Decay           : newer transactions weighted higher
#
# Known limitations (data not available in yfinance):
#   - Cannot identify 10b5-1 scheduled trading plans
#   - Cannot identify option exercise followed by immediate sale
#   - Cannot assess buy size relative to insider net worth
#   - Cannot track post-purchase holding behavior
#   - Pump and dump patterns not reliably detectable
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

PRIMARY_MONTHS   = 6
FALLBACK_MONTHS  = 12
MIN_TRANSACTIONS = 3

TIME_DECAY_INSIDER = [
    (1,  1.0),
    (3,  0.85),
    (6,  0.65),
    (12, 0.35),
]
TIME_DECAY_DEFAULT = 0.15

POSITION_WEIGHTS = {
    "ceo":       2.0,
    "cfo":       1.8,
    "coo":       1.6,
    "president": 1.6,
    "chairman":  1.5,
    "director":  1.0,
    "vp":        0.9,
    "officer":   0.8,
    "other":     0.6,
}

SIZE_THRESHOLDS = [
    (10_000_000, 20),
    (5_000_000,  16),
    (1_000_000,  12),
    (500_000,     8),
    (100_000,     4),
    (0,           1),
]

# Keywords to detect purchase vs sale from Transaction column
PURCHASE_KEYWORDS = ["purchase", "buy", "bought", "acquisition"]
SALE_KEYWORDS     = ["sale", "sell", "sold", "disposed"]
SKIP_KEYWORDS     = ["gift", "stock gift", "option", "exercise", "award", "grant"]


# ============================================================
# GPT Prompt
# ============================================================

INSIDER_SYSTEM_PROMPT = """
You are an insider trading signal analyst.

Given a list of insider transactions for a stock, analyse each one
and return ONLY a JSON array. No preamble, no markdown, no explanation.

For each transaction return exactly this structure:
{
  "index":    <int, the transaction index>,
  "signal":   "bullish" | "bearish" | "neutral",
  "strength": "strong" | "moderate" | "weak",
  "reason":   "<one concise sentence explaining the signal quality>"
}

Signal guidelines:
- bullish  : open market purchase by insider
- bearish  : open market sale by insider
- neutral  : ambiguous, small amounts, gifts, or unclear motivation

Strength guidelines:
- strong   : CEO/CFO/President, large amount, clear open market buy
- moderate : mid-level officer or director, meaningful amount
- weak     : small amount, unclear motivation, or routine-looking

Key considerations:
- Large open market purchases by top executives are the strongest bullish signals
- Sales are inherently weaker signals -- many benign reasons exist for selling
- Stock gifts (price $0) are not meaningful trading signals
- Very small purchases may be symbolic rather than conviction buys
- Multiple insiders buying simultaneously strengthens the signal

Return ONLY the JSON array. No other text.
"""


# ============================================================
# Helpers
# ============================================================

def _insider_time_weight(date_val) -> float:
    """Calculates time decay weight for a transaction by age."""
    try:
        now = datetime.now(timezone.utc)
        ts  = pd.Timestamp(date_val, tz="UTC") if not hasattr(date_val, "tzinfo") \
              else (date_val.tz_localize("UTC") if date_val.tzinfo is None else date_val)
        age_months = (now - ts.to_pydatetime()).days / 30.0
        for max_months, weight in TIME_DECAY_INSIDER:
            if age_months <= max_months:
                return round(weight, 2)
        return TIME_DECAY_DEFAULT
    except Exception:
        return TIME_DECAY_DEFAULT


def _classify_position(title: str) -> tuple:
    """Classifies insider position → (key, weight)."""
    if not title:
        return "other", POSITION_WEIGHTS["other"]
    t = title.lower()
    if "chief executive" in t or t.strip().startswith("ceo"):
        return "ceo",       POSITION_WEIGHTS["ceo"]
    elif "chief financial" in t or t.strip().startswith("cfo"):
        return "cfo",       POSITION_WEIGHTS["cfo"]
    elif "chief operating" in t or t.strip().startswith("coo"):
        return "coo",       POSITION_WEIGHTS["coo"]
    elif "president" in t:
        return "president", POSITION_WEIGHTS["president"]
    elif "chairman" in t or "chair" in t:
        return "chairman",  POSITION_WEIGHTS["chairman"]
    elif "director" in t:
        return "director",  POSITION_WEIGHTS["director"]
    elif "vice president" in t or " vp" in t:
        return "vp",        POSITION_WEIGHTS["vp"]
    elif "officer" in t or "chief" in t:
        return "officer",   POSITION_WEIGHTS["officer"]
    else:
        return "other",     POSITION_WEIGHTS["other"]


def _classify_transaction(txn_str: str, text_str: str, value: float) -> tuple:
    """
    Determines if a transaction is a purchase, sale, or skip.
    Returns (is_buy, is_sell, should_skip).
    Uses Transaction column first, Text column as fallback.
    """
    combined = (str(txn_str) + " " + str(text_str)).lower()

    # Skip gifts, options, awards
    if any(k in combined for k in SKIP_KEYWORDS):
        return False, False, True

    # Stock gift at $0 value
    if value == 0 and "gift" in combined:
        return False, False, True

    is_buy  = any(k in combined for k in PURCHASE_KEYWORDS)
    is_sell = any(k in combined for k in SALE_KEYWORDS)

    # If both detected, Transaction column takes priority
    if is_buy and is_sell:
        txn_lower = str(txn_str).lower()
        is_buy  = any(k in txn_lower for k in PURCHASE_KEYWORDS)
        is_sell = not is_buy

    return is_buy, is_sell, False


def _transaction_size_score(value: float) -> int:
    """Maps transaction USD value to size score."""
    for threshold, score in SIZE_THRESHOLDS:
        if value >= threshold:
            return score
    return 1


# ============================================================
# Data Fetcher
# ============================================================

def _fetch_insider_transactions(ticker_symbol: str) -> pd.DataFrame:
    """
    Fetches insider_transactions from yfinance.
    Confirmed columns: Shares, Value, URL, Text, Insider,
                       Position, Transaction, Start Date, Ownership
    """
    try:
        time.sleep(1)
        stock = yf.Ticker(ticker_symbol)
        txns  = stock.insider_transactions

        if txns is None or txns.empty:
            print(f"  insider_transactions empty for {ticker_symbol}")
            return pd.DataFrame()

        print(f"  insider_transactions found: {txns.shape}, cols: {list(txns.columns)}")

        # Parse Start Date
        txns["_date"] = pd.to_datetime(txns["Start Date"], utc=True, errors="coerce")
        txns = txns.dropna(subset=["_date"])
        txns = txns.sort_values("_date", ascending=False).reset_index(drop=True)

        # Primary window
        cutoff  = datetime.now(timezone.utc) - timedelta(days=PRIMARY_MONTHS * 30)
        primary = txns[txns["_date"] >= cutoff]
        if len(primary) >= MIN_TRANSACTIONS:
            return primary.reset_index(drop=True)

        # Fallback
        cutoff = datetime.now(timezone.utc) - timedelta(days=FALLBACK_MONTHS * 30)
        return txns[txns["_date"] >= cutoff].reset_index(drop=True)

    except Exception as e:
        print(f"  Insider fetch error: {e}")
        return pd.DataFrame()


# ============================================================
# GPT Analyser
# ============================================================

def _analyse_with_gpt(transactions: list, ticker: str) -> list:
    """Sends transactions to GPT for signal analysis."""
    if not transactions:
        return []

    txn_text = ""
    for i, t in enumerate(transactions):
        txn_text += (
            f"\n[{i}] {t['insider']} ({t['title']})\n"
            f"  Action: {t['transaction_type']}\n"
            f"  Shares: {t['shares']:,}\n"
            f"  Value:  USD {t['value']:,.0f}\n"
            f"  Date:   {t['date_str']}\n"
        )

    try:
        client   = OpenAI()
        response = client.chat.completions.create(
            model       = "gpt-4.1-mini",
            max_tokens  = 1500,
            temperature = 0.1,
            messages    = [
                {"role": "system", "content": INSIDER_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Stock: {ticker}\nAnalyse:\n{txn_text}"},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        print(f"  GPT insider analysis error: {e}")
        return []


# ============================================================
# Master Function
# ============================================================

def fetch_insider_sentiment(ticker_symbol: str) -> dict:
    """
    Fetches insider transactions and scores sentiment (0-100).

    Scoring breakdown:
      Net Buy/Sell Ratio  : 0-30 pts
      Buyer/Seller Count  : 0-25 pts
      Transaction Size    : 0-20 pts
      Position Level      : 0-15 pts
      Baseline            : 10 pts
    """
    print(f"  Fetching insider sentiment for {ticker_symbol}...")

    txns_df = _fetch_insider_transactions(ticker_symbol)

    if txns_df.empty:
        return {
            "score":             50,
            "direction":         "neutral",
            "transaction_count": 0,
            "transactions":      [],
            "signals": ["No insider transaction data available -- neutral score applied ➡️"],
        }

    # Parse each row into structured transaction
    parsed = []
    for _, row in txns_df.iterrows():
        txn_str  = str(row.get("Transaction", ""))
        text_str = str(row.get("Text",        ""))
        shares   = abs(float(row.get("Shares", 0) or 0))
        value    = abs(float(row.get("Value",  0) or 0))
        date_ts  = row.get("_date")
        insider  = str(row.get("Insider",  "Unknown")).strip().title()
        title    = str(row.get("Position", "")).strip()

        is_buy, is_sell, skip = _classify_transaction(txn_str, text_str, value)
        if skip:
            continue

        pos_key, pos_weight = _classify_position(title)
        t_weight = _insider_time_weight(date_ts)
        date_str = date_ts.strftime("%Y-%m-%d") if date_ts is not None else "Unknown"

        parsed.append({
            "insider":          insider,
            "title":            title,
            "position_key":     pos_key,
            "position_weight":  pos_weight,
            "transaction_type": txn_str if txn_str else ("Purchase" if is_buy else "Sale"),
            "shares":           int(shares),
            "value":            value,
            "date_str":         date_str,
            "time_weight":      t_weight,
            "is_buy":           is_buy,
            "is_sell":          is_sell,
        })

    if not parsed:
        return {
            "score":             50,
            "direction":         "neutral",
            "transaction_count": 0,
            "transactions":      [],
            "signals": ["All insider transactions were non-market (gifts/awards) -- neutral ➡️"],
        }

    # GPT analysis
    judgments    = _analyse_with_gpt(parsed, ticker_symbol)
    judgment_map = {j["index"]: j for j in judgments if "index" in j}

    enriched = []
    for i, t in enumerate(parsed):
        j = judgment_map.get(i, {})
        enriched.append({
            **t,
            "signal":   j.get("signal",   "neutral"),
            "strength": j.get("strength", "weak"),
            "reason":   j.get("reason",   ""),
        })

    # Score calculation
    buyers  = [t for t in enriched if t["is_buy"]]
    sellers = [t for t in enriched if t["is_sell"]]

    buy_value   = sum(t["value"] * t["time_weight"] for t in buyers)
    sell_value  = sum(t["value"] * t["time_weight"] for t in sellers)
    total_value = buy_value + sell_value

    # 1. Net buy/sell ratio (0-30 pts)
    if total_value > 0:
        net_ratio   = (buy_value - sell_value) / total_value
        ratio_score = max(0, min(30, round(15 + net_ratio * 15)))
    else:
        ratio_score = 15

    # 2. Buyer/seller count (0-25 pts)
    n_buyers  = len(set(t["insider"] for t in buyers))
    n_sellers = len(set(t["insider"] for t in sellers))
    n_total   = n_buyers + n_sellers
    count_score = round((n_buyers / n_total) * 25) if n_total > 0 else 12

    # 3. Largest single purchase (0-20 pts)
    max_buy    = max((t["value"] for t in buyers), default=0)
    size_score = min(20, _transaction_size_score(max_buy)) if buyers else 0

    # 4. Highest position buyer (0-15 pts)
    if buyers:
        top_pos = max(t["position_weight"] for t in buyers)
        pos_score = max(0, min(15, round((top_pos - 0.6) / 1.4 * 15)))
    else:
        pos_score = 0

    # 5. Baseline
    baseline    = 10
    final_score = max(0, min(100, ratio_score + count_score + size_score + pos_score + baseline))
    direction   = "bullish" if final_score >= 60 else ("bearish" if final_score <= 40 else "neutral")

    # Signals
    signals = []
    signals.append(
        f"Insider activity: {n_buyers} buyer(s), {n_sellers} seller(s) "
        f"of {len(enriched)} total transactions"
    )
    if buy_value > 0 or sell_value > 0:
        net_dir = "net buying" if buy_value > sell_value else "net selling"
        icon    = "✅" if buy_value > sell_value else "⚠️"
        signals.append(
            f"Value flow: USD {buy_value:,.0f} bought vs "
            f"USD {sell_value:,.0f} sold -- {net_dir} {icon}"
        )
    if buyers:
        tb = sorted(buyers, key=lambda x: x["value"] * x["position_weight"], reverse=True)[0]
        signals.append(
            f"Top purchase: {tb['insider']} ({tb['title']}) "
            f"-- USD {tb['value']:,.0f} ✅"
        )
    icon = "✅" if direction == "bullish" else ("⚠️" if direction == "bearish" else "➡️")
    signals.append(f"Overall insider tone: {direction} {icon}")

    return {
        "score":             final_score,
        "direction":         direction,
        "transaction_count": len(enriched),
        "transactions":      enriched,
        "signals":           signals,
    }
