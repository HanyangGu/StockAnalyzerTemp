# ============================================================
# scoring/sentiment_scorer.py -- Sentiment Scoring Engine
# ============================================================
# Aggregates all sentiment sub-dimension scores into one
# overall sentiment score (0-100).
#
# Current weights:
#   News Sentiment   : 60%
#   Analyst Ratings  : 25%
#   Insider Trading  : 15%
#
# Future:
#   Options Sentiment: TBD
# ============================================================

WEIGHT_NEWS    = 0.60
WEIGHT_ANALYST = 0.25
WEIGHT_INSIDER = 0.15

# Future:
# WEIGHT_NEWS    = 0.45
# WEIGHT_ANALYST = 0.20
# WEIGHT_INSIDER = 0.15
# WEIGHT_OPTIONS = 0.20


def get_sentiment_verdict(score: int) -> tuple:
    """Converts sentiment score to verdict label and emoji."""
    if score >= 75:
        return "Very Bullish Sentiment",  "🟢"
    elif score >= 60:
        return "Bullish Sentiment",       "🟩"
    elif score >= 40:
        return "Neutral Sentiment",       "⬜"
    elif score >= 25:
        return "Bearish Sentiment",       "🟥"
    else:
        return "Very Bearish Sentiment",  "🔴"


def score_sentiment(news: dict,
                    analyst: dict = None,
                    insider: dict = None) -> dict:
    """
    Aggregates all active sentiment sub-dimension scores.

    Args:
        news    : Result from analyzers/sentiment/news.py
        analyst : Result from analyzers/sentiment/analyst.py
        insider : Result from analyzers/sentiment/insider.py

    Returns:
        {
          "score"    : 0-100,
          "verdict"  : str,
          "icon"     : emoji,
          "signals"  : [str],
          "breakdown": { news, analyst, insider }
        }
    """
    signals   = []
    breakdown = {}

    # -- News (60%) -------------------------------------------
    news_score = news.get("score", 50) if news else 50
    breakdown["news"] = {
        "score":         news_score,
        "direction":     news.get("direction",     "neutral") if news else "neutral",
        "article_count": news.get("article_count", 0)        if news else 0,
        "articles":      news.get("articles",      [])        if news else [],
        "signals":       news.get("signals",       [])        if news else [],
    }

    # -- Analyst (25%) ----------------------------------------
    analyst_score = analyst.get("score", 50) if analyst else 50
    breakdown["analyst"] = {
        "score":        analyst_score,
        "direction":    analyst.get("direction",    "neutral") if analyst else "neutral",
        "rating_count": analyst.get("rating_count", 0)        if analyst else 0,
        "summary":      analyst.get("summary",      {})        if analyst else {},
        "targets":      analyst.get("targets",      {})        if analyst else {},
        "signals":      analyst.get("signals",      [])        if analyst else [],
    }

    # -- Insider (15%) ----------------------------------------
    insider_score = insider.get("score", 50) if insider else 50
    breakdown["insider"] = {
        "score":             insider_score,
        "direction":         insider.get("direction",         "neutral") if insider else "neutral",
        "transaction_count": insider.get("transaction_count", 0)        if insider else 0,
        "transactions":      insider.get("transactions",      [])        if insider else [],
        "signals":           insider.get("signals",           [])        if insider else [],
    }

    # -- Weighted aggregate -----------------------------------
    active_weight = 0.0
    weighted_sum  = 0.0

    weighted_sum  += news_score    * WEIGHT_NEWS
    active_weight += WEIGHT_NEWS

    if analyst:
        weighted_sum  += analyst_score * WEIGHT_ANALYST
        active_weight += WEIGHT_ANALYST

    if insider:
        weighted_sum  += insider_score * WEIGHT_INSIDER
        active_weight += WEIGHT_INSIDER

    # Redistribute if some dimensions unavailable
    overall_score = round(weighted_sum / active_weight) if active_weight > 0 else 50
    overall_score = max(0, min(100, overall_score))
    verdict, icon = get_sentiment_verdict(overall_score)

    return {
        "score":     overall_score,
        "verdict":   verdict,
        "icon":      icon,
        "signals":   signals,
        "breakdown": breakdown,
    }
