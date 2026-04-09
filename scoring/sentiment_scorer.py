# ============================================================
# scoring/sentiment_scorer.py -- Sentiment Scoring Engine
# ============================================================
# Aggregates all sentiment sub-dimension scores into one
# overall sentiment score (0-100).
#
# Current sub-dimensions and weights:
#   News Sentiment     : 60%
#   Analyst Ratings    : 40%
#
# Future sub-dimensions:
#   Insider Trading    : weight TBD
#   Options Sentiment  : weight TBD
# ============================================================

# ============================================================
# Sub-dimension Weights
# ============================================================
WEIGHT_NEWS    = 0.60
WEIGHT_ANALYST = 0.40

# Future:
# WEIGHT_NEWS    = 0.40
# WEIGHT_ANALYST = 0.30
# WEIGHT_INSIDER = 0.20
# WEIGHT_OPTIONS = 0.10


# ============================================================
# Verdict
# ============================================================

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


# ============================================================
# Aggregator
# ============================================================

def score_sentiment(news: dict, analyst: dict = None) -> dict:
    """
    Aggregates all active sentiment sub-dimension scores.

    Args:
        news    : Result from analyzers/sentiment/news.py
        analyst : Result from analyzers/sentiment/analyst.py (optional)

    Returns:
        {
          "score"    : 0-100,
          "verdict"  : str,
          "icon"     : emoji,
          "signals"  : [str],
          "breakdown": {
              "news":    { score, direction, article_count, signals },
              "analyst": { score, direction, rating_count,  signals },
          }
        }
    """
    signals   = []
    breakdown = {}

    # -- News Sentiment (60%) ---------------------------------
    news_score = news.get("score", 50) if news else 50
    breakdown["news"] = {
        "score":         news_score,
        "direction":     news.get("direction",     "neutral"),
        "article_count": news.get("article_count", 0),
        "articles":      news.get("articles",      []),
        "signals":       news.get("signals",       []),
    }
    signals.extend(news.get("signals", []))

    # -- Analyst Ratings (40%) --------------------------------
    analyst_score = analyst.get("score", 50) if analyst else 50
    breakdown["analyst"] = {
        "score":        analyst_score,
        "direction":    analyst.get("direction",    "neutral") if analyst else "neutral",
        "rating_count": analyst.get("rating_count", 0)        if analyst else 0,
        "summary":      analyst.get("summary",      {})        if analyst else {},
        "targets":      analyst.get("targets",      {})        if analyst else {},
        "signals":      analyst.get("signals",      [])        if analyst else [],
    }
    if analyst:
        signals.extend(analyst.get("signals", []))

    # -- Weighted aggregate -----------------------------------
    if analyst:
        overall_score = round(
            news_score    * WEIGHT_NEWS +
            analyst_score * WEIGHT_ANALYST
        )
    else:
        # Redistribute analyst weight to news if unavailable
        overall_score = news_score

    overall_score = max(0, min(100, overall_score))
    verdict, icon = get_sentiment_verdict(overall_score)

    return {
        "score":     overall_score,
        "verdict":   verdict,
        "icon":      icon,
        "signals":   signals,
        "breakdown": breakdown,
    }