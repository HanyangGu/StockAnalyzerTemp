# ============================================================
# orchestrator.py -- Master Analysis Orchestrator
# ============================================================
# Coordinates the full analysis pipeline for a single stock:
#   1. Resolve ticker
#   2. Fetch price data (live or historical)
#   3. Fetch historical OHLCV data
#   4. Run all analyzer modules
#   5. Run all scorer modules
#   6. Return complete serializable result
#
# To add a new analysis dimension:
#   1. Add analyzer in analyzers/
#   2. Add scorer in scoring/
#   3. Call both here and include result in the return dict
#   4. Update composite.py weights
# ============================================================

import time
import numpy as np
from datetime import datetime

from core.data import (
    fetch_price_data,
    fetch_price_data_historical,
    fetch_historical_data,
    resolve_ticker,
)
from analyzers.technical import (
    compute_rsi,
    compute_stochastic,
    compute_roc,
    compute_macd,
    compute_moving_averages,
    compute_golden_cross,
    compute_bollinger_bands,
    compute_atr,
    compute_volume_trend,
)
from analyzers.fundamental import fetch_fundamental_data
from analyzers.sentiment.news import fetch_news_sentiment
from analyzers.sentiment.analyst import fetch_analyst_sentiment
from analyzers.sentiment.insider import fetch_insider_sentiment
from scoring.technical_scorer import (
    score_short_term,
    score_mid_term,
    score_long_term,
    score_technical_overall,
)
from scoring.fundamental_scorer import score_fundamentals
from scoring.sentiment_scorer import score_sentiment
from scoring.composite import get_composite


# ============================================================
# Utility
# ============================================================

def make_serializable(obj):
    """
    Recursively converts numpy types to native Python types
    so the result can be safely serialized to JSON.
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


# ============================================================
# Master Orchestrator
# ============================================================

def run_analysis(company: str, backtest_date: str = None) -> dict:
    """
    Master function -- runs full multi-dimension analysis on a stock.

    Steps:
      1.  Resolve ticker symbol
      2.  Fetch price data (live or historical)
      3.  Fetch historical OHLCV data
      4.  Compute all 9 technical indicators
      5.  Score across 3 technical time horizons
      6.  Calculate technical overall score
      7.  Fetch and score fundamental data
      8.  Fetch and score sentiment data
      9.  Calculate composite score across all dimensions
      10. Return complete serializable result
    """
    # Step 1: Resolve ticker
    ticker = resolve_ticker(company)
    print(f"  Running analysis for: {ticker}...")

    # Step 2: Fetch price data
    if backtest_date:
        price_data = fetch_price_data_historical(ticker, backtest_date)
    else:
        price_data = fetch_price_data(ticker)

    if "error" in price_data:
        if "too many requests" in str(price_data["error"]).lower() or \
           "rate limit"        in str(price_data["error"]).lower():
            return {
                "error": (
                    "Yahoo Finance is temporarily rate limited. "
                    "Please wait 30 seconds and try again."
                )
            }
        return {"error": price_data["error"]}

    # Step 3: Fetch historical OHLCV data
    time.sleep(1)
    hist_data = fetch_historical_data(ticker, end_date=backtest_date)
    if not hist_data["success"]:
        if "too many requests" in str(hist_data["error"]).lower() or \
           "rate limit"        in str(hist_data["error"]).lower():
            return {
                "error": (
                    "Yahoo Finance is temporarily rate limited. "
                    "Please wait 30 seconds and try again."
                )
            }
        return {"error": hist_data["error"]}

    df            = hist_data["df"]
    closes        = hist_data["closes"]
    current_price = price_data["current_price"]

    # Step 4: Compute all 9 technical indicators
    rsi    = compute_rsi(closes)
    stoch  = compute_stochastic(df)
    roc    = compute_roc(closes)
    macd   = compute_macd(closes)
    mas    = compute_moving_averages(closes, current_price)
    golden = compute_golden_cross(closes)
    bb     = compute_bollinger_bands(closes, current_price)
    atr    = compute_atr(df)
    vol    = compute_volume_trend(df)

    # Step 5: Score each technical time horizon
    short = score_short_term(rsi, stoch, roc, bb, mas)
    mid   = score_mid_term(macd, mas, atr, vol)
    long  = score_long_term(mas, golden, vol)

    # Step 6: Technical overall score
    overall = score_technical_overall(short, mid, long)

    # Step 7: Fetch and score fundamental data (non-blocking)
    fund_data = fetch_fundamental_data(ticker)
    if "error" in fund_data:
        fundamental  = {
            "score":   None,
            "verdict": "Data unavailable",
            "icon":    "➡️",
            "signals": [],
        }
        fund_details = {}
        print(f"  Fundamental analysis skipped: {fund_data['error']}")
    else:
        fund_scored  = score_fundamentals(fund_data)
        fundamental  = fund_scored
        fund_details = {
            "valuation": {
                "pe_ratio":       fund_data["pe_ratio"],
                "forward_pe":     fund_data["forward_pe"],
                "price_to_book":  fund_data["price_to_book"],
                "price_to_sales": fund_data["price_to_sales"],
            },
            "profitability": {
                "gross_margin": fund_data["gross_margin"],
                "net_margin":   fund_data["net_margin"],
                "roe":          fund_data["roe"],
                "roa":          fund_data["roa"],
            },
            "growth": {
                "revenue_growth":  fund_data["revenue_growth"],
                "earnings_growth": fund_data["earnings_growth"],
                "eps":             fund_data["eps"],
                "forward_eps":     fund_data["forward_eps"],
            },
            "health": {
                "debt_to_equity": fund_data["debt_to_equity"],
                "current_ratio":  fund_data["current_ratio"],
                "free_cash_flow": fund_data["free_cash_flow"],
                "total_revenue":  fund_data["total_revenue"],
            },
            "analyst": {
                "target_price":   fund_data["target_price"],
                "target_high":    fund_data["target_high"],
                "target_low":     fund_data["target_low"],
                "recommendation": fund_data["recommendation"],
                "analyst_count":  fund_data["analyst_count"],
                "current_price":  fund_data["current_price"],
                "upside_pct":     fund_data["upside_pct"],
            },
        }

    # Step 8: Fetch and score sentiment data (non-blocking)
    news_data     = fetch_news_sentiment(
        ticker_symbol = ticker,
        company_name  = price_data.get("name", ""),
    )
    analyst_data  = fetch_analyst_sentiment(ticker)
    insider_data  = fetch_insider_sentiment(ticker)
    sentiment     = score_sentiment(
        news    = news_data,
        analyst = analyst_data,
        insider = insider_data,
    )

    # Step 9: Composite score across all dimensions
    composite = get_composite(
        technical   = overall["score"],
        fundamental = fundamental["score"],
        sentiment   = sentiment["score"],
    )

    # Step 10: Build and return result
    result = {
        "ticker":           ticker,
        "company":          price_data["name"],
        "current_price":    current_price,
        "price_change":     price_data["price_change"],
        "price_change_pct": price_data["price_change_pct"],
        "backtest_date":    backtest_date or "live",

        # Technical scores
        "short_term":       short,
        "mid_term":         mid,
        "long_term":        long,
        "overall":          overall,

        # Raw technical indicators
        "indicators": {
            "rsi":    rsi,
            "stoch":  stoch,
            "roc":    roc,
            "macd":   macd,
            "mas":    mas,
            "golden": golden,
            "bb":     bb,
            "atr":    atr,
            "volume": vol,
        },

        # Fundamental scores and raw data
        "fundamental":  fundamental,
        "fund_details": fund_details,

        # Sentiment scores and breakdown
        "sentiment":    sentiment,

        # Composite decision
        "composite":    composite,

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return make_serializable(result)
