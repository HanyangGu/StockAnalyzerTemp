# ============================================================
# comparison.py -- Stock Comparison
# ============================================================
# Compares 2-3 stocks side by side using technical analysis.
# Ranks by overall score and identifies best picks per horizon.
# ============================================================

from datetime import datetime

from core.data import validate_companies
from scoring.orchestrator import run_analysis


def compare_stocks(companies: list) -> dict:
    """
    Compares 2 to 3 stocks side by side using technical analysis.

    Steps:
      1. Validate input (min 2, max 3 stocks)
      2. Run run_analysis() on each stock
      3. Sort by overall score descending
      4. Assign medals 🥇🥈🥉
      5. Build comparison table
      6. Identify best picks per time horizon
      7. Return complete comparison dict
    """

    # Step 1: Validate input
    validation = validate_companies(companies)
    if not validation["valid"]:
        return {"error": validation["error"]}

    # Step 2: Run analysis on each stock
    results = []
    errors  = []

    for company in validation["tickers"]:
        print(f"  Analysing {company}...")
        analysis = run_analysis(company)
        if "error" in analysis:
            errors.append(f"{company}: {analysis['error']}")
        else:
            results.append(analysis)

    # Return early if all stocks failed
    if not results:
        return {
            "error":   "Could not retrieve data for any of the requested stocks.",
            "details": errors,
        }

    partial_failure = len(errors) > 0

    # Step 3: Sort by overall score descending
    results.sort(
        key=lambda x: x["overall"]["score"],
        reverse=True
    )

    # Step 4: Assign medals
    medals  = ["🥇", "🥈", "🥉"]
    ranking = []
    for i, r in enumerate(results):
        ranking.append({
            "rank":        i + 1,
            "medal":       medals[i],
            "ticker":      r["ticker"],
            "company":     r["company"],
            "score":       r["overall"]["score"],
            "verdict":     r["overall"]["verdict"],
            "icon":        r["overall"]["icon"],
            "short_score": r["short_term"]["score"],
            "mid_score":   r["mid_term"]["score"],
            "long_score":  r["long_term"]["score"],
            "fund_score":  r["fundamental"]["score"],
            "fund_verdict":r["fundamental"]["verdict"],
        })

    # Step 5: Build comparison table
    comparison_table = {}
    for r in results:
        ticker = r["ticker"]
        comparison_table[ticker] = {
            "price":         r["current_price"],
            "change":        r["price_change"],
            "change_pct":    r["price_change_pct"],
            "short_term":    r["short_term"]["score"],
            "mid_term":      r["mid_term"]["score"],
            "long_term":     r["long_term"]["score"],
            "overall":       r["overall"]["score"],
            "verdict":       r["overall"]["verdict"],
            "icon":          r["overall"]["icon"],
            "fundamental":   r["fundamental"]["score"],
            "fund_verdict":  r["fundamental"]["verdict"],
            "sentiment":     r.get("sentiment", {}).get("score"),
            "rsi":           r["indicators"]["rsi"]["value"],
            "rsi_signal":    r["indicators"]["rsi"]["icon"],
            "stoch":         r["indicators"]["stoch"]["value"],
            "stoch_signal":  r["indicators"]["stoch"]["icon"],
            "roc":           r["indicators"]["roc"]["value"],
            "roc_signal":    r["indicators"]["roc"]["icon"],
            "macd":          r["indicators"]["macd"]["signal_label"],
            "macd_signal":   r["indicators"]["macd"]["icon"],
            "ma20":          r["indicators"]["mas"]["ma20"]["value"],
            "ma20_signal":   r["indicators"]["mas"]["ma20"]["icon"],
            "ma50":          r["indicators"]["mas"]["ma50"]["value"],
            "ma50_signal":   r["indicators"]["mas"]["ma50"]["icon"],
            "ma200":         r["indicators"]["mas"]["ma200"]["value"],
            "ma200_signal":  r["indicators"]["mas"]["ma200"]["icon"],
            "golden_cross":  r["indicators"]["golden"]["signal"],
            "golden_signal": r["indicators"]["golden"]["icon"],
            "bb_pct":        r["indicators"]["bb"]["pct"],
            "bb_signal":     r["indicators"]["bb"]["icon"],
            "atr":           r["indicators"]["atr"]["value"],
            "atr_signal":    r["indicators"]["atr"]["icon"],
            "volume":        r["indicators"]["volume"]["ratio"],
            "volume_signal": r["indicators"]["volume"]["icon"],
        }

    # Step 6: Identify best picks
    best_overall = ranking[0]["ticker"]
    best_short   = max(
        results, key=lambda x: x["short_term"]["score"]
    )["ticker"]
    best_mid     = max(
        results, key=lambda x: x["mid_term"]["score"]
    )["ticker"]
    best_long    = max(
        results, key=lambda x: x["long_term"]["score"]
    )["ticker"]
    lowest_risk  = min(
        results, key=lambda x: x["indicators"]["atr"]["pct"]
    )["ticker"]

    # Best fundamentals -- skip stocks with None score
    fund_eligible = [r for r in results
                     if r["fundamental"]["score"] is not None]
    best_fund = max(
        fund_eligible,
        key=lambda x: x["fundamental"]["score"]
    )["ticker"] if fund_eligible else "N/A"

    # Step 7: Return complete result
    return {
        "stocks_analysed":  len(results),
        "ranking":          ranking,
        "comparison_table": comparison_table,
        "best_picks": {
            "overall":      best_overall,
            "short_term":   best_short,
            "mid_term":     best_mid,
            "long_term":    best_long,
            "lowest_risk":  lowest_risk,
            "fundamentals": best_fund,
        },
        "partial_failure": partial_failure,
        "errors":          errors if partial_failure else [],
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }