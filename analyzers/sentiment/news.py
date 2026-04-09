# ============================================================
# analyzers/sentiment/news.py -- News Sentiment Analyzer
# ============================================================
# Fetches news from two sources and uses GPT to evaluate
# each article across multiple dimensions:
#
# Sources:
#   1. yfinance general news  -- mixed financial news
#   2. yfinance ticker search -- stock-specific news
#   (deduplicated by title similarity before analysis)
#
# Relevance (GPT judged):
#   direct   : explicitly about this stock        → full weight
#   indirect : affects industry/macro environment → × 0.4
#   unrelated: no bearing on this stock           → × 0
#
# Per-article dimensions (GPT judged):
#   sentiment   : positive / neutral / negative
#   intensity   : strong / moderate / mild
#   impact      : major / normal / minor
#   scope       : company / industry / macro
#   credibility : high / medium / low
#   novelty     : first / followup / repeat
#   surprise    : unexpected / partial / expected
#
# Time decay (auto-calculated) with impact floor:
#   0-3d  → 1.0 | 4-7d → 0.8 | 8-14d → 0.6
#   15-30d → 0.4 | 30d+ → 0.2
#   Floor: major ≥ 0.5 | normal ≥ 0.3 | minor = no floor
#
# Cross-article adjustment:
#   consensus : agreement across relevant articles only
#               (requires MIN_RELEVANT_ARTICLES to activate)
# ============================================================

import json
import time
from datetime import datetime, timezone

import yfinance as yf
from openai import OpenAI


# ============================================================
# Constants
# ============================================================

MAX_GENERAL        = 10   # Articles from yfinance general news
MAX_SPECIFIC       = 8    # Articles from yfinance ticker search
MAX_ANALYSE        = 12   # Max total articles sent to GPT
MIN_RELEVANT       = 3    # Minimum relevant articles for consensus

# Relevance multipliers
RELEVANCE_WEIGHTS  = {
    "direct":    1.0,
    "indirect":  0.4,
    "unrelated": 0.0,
}

# Time decay weights by age bucket (days)
TIME_DECAY = [
    (3,  1.0),
    (7,  0.8),
    (14, 0.6),
    (30, 0.4),
]
TIME_DECAY_DEFAULT = 0.2   # 30d+

# Impact-based time decay floor
TIME_WEIGHT_FLOOR = {
    "major":  0.5,
    "normal": 0.3,
    "minor":  0.0,
}

# Per-article dimension multipliers
INTENSITY_WEIGHTS   = {"strong": 1.0,  "moderate": 0.6,  "mild": 0.3}
IMPACT_WEIGHTS      = {"major":  2.0,  "normal":   1.0,  "minor": 0.5}
SCOPE_WEIGHTS       = {"company": 1.5, "industry": 1.0,  "macro": 0.6}
CREDIBILITY_WEIGHTS = {"high":   1.5,  "medium":   1.0,  "low":   0.5}
NOVELTY_WEIGHTS     = {"first":  1.5,  "followup": 0.7,  "repeat": 0.3}
SURPRISE_WEIGHTS    = {"unexpected": 1.5, "partial": 1.0, "expected": 0.5}

# Consensus adjustment
CONSENSUS_BOOST   = 1.2
CONSENSUS_PENALTY = 0.8


# ============================================================
# GPT System Prompt
# ============================================================

NEWS_SYSTEM_PROMPT = """
You are a financial news sentiment analyst.

Given a list of news articles about a stock, analyse each one
and return ONLY a JSON array. No preamble, no markdown, no explanation.

For each article return exactly this structure:
{
  "index":       <int, the article index>,
  "relevance":   "direct" | "indirect" | "unrelated",
  "sentiment":   "positive" | "neutral" | "negative",
  "intensity":   "strong" | "moderate" | "mild",
  "impact":      "major" | "normal" | "minor",
  "scope":       "company" | "industry" | "macro",
  "credibility": "high" | "medium" | "low",
  "novelty":     "first" | "followup" | "repeat",
  "surprise":    "unexpected" | "partial" | "expected",
  "reason":      "<one concise sentence explaining your judgment>"
}

Dimension definitions:
- relevance   : direct=explicitly about this stock's business/price/operations,
                indirect=affects the industry or macro environment that impacts this stock,
                unrelated=about other companies or topics with no bearing on this stock
- sentiment   : effect on this stock's price outlook (neutral if unrelated)
- intensity   : how strongly positive/negative (mild if neutral or unrelated)
- impact      : major=earnings/regulation/M&A/CEO change, normal=products/partnerships, minor=general commentary
- scope       : company=affects only this stock, industry=affects sector, macro=affects all stocks
- credibility : high=Reuters/Bloomberg/WSJ/FT, medium=established outlets, low=unknown/blog
- novelty     : first=new information, followup=additional coverage of same event, repeat=repost/duplicate
- surprise    : unexpected=market had no warning, partial=some rumours existed, expected=widely anticipated

IMPORTANT:
- indirect articles (macro/industry news) still count but at reduced weight
- unrelated articles are fully excluded from scoring
- Be generous with indirect: if the news could plausibly affect this stock, mark indirect not unrelated

Return ONLY the JSON array. No other text.
"""


# ============================================================
# Time Decay Calculator
# ============================================================

def _time_decay_weight(published_ts: int, impact: str = "minor") -> float:
    """
    Calculates time decay weight based on article age in days.
    Applies impact-based floor so major events retain minimum weight.
    """
    if not published_ts:
        return TIME_DECAY_DEFAULT

    try:
        now      = datetime.now(timezone.utc)
        pub      = datetime.fromtimestamp(published_ts, tz=timezone.utc)
        age_days = (now - pub).days

        raw_weight = TIME_DECAY_DEFAULT
        for max_days, weight in TIME_DECAY:
            if age_days <= max_days:
                raw_weight = weight
                break

        # Apply impact floor -- major events stay impactful longer
        floor = TIME_WEIGHT_FLOOR.get(impact, 0.0)
        return max(raw_weight, floor)

    except Exception:
        return TIME_DECAY_DEFAULT


# ============================================================
# News Fetchers
# ============================================================

def _parse_article(item: dict) -> dict | None:
    """Parses a raw yfinance news item into a clean dict."""
    content = item.get("content", item)
    title   = (
        content.get("title") or
        item.get("title", "")
    )
    if not title:
        return None

    summary = (
        content.get("summary") or
        content.get("description") or
        item.get("summary", "")
    )
    source  = (
        content.get("provider", {}).get("displayName") or
        content.get("source") or
        item.get("publisher", "Unknown")
    )
    pub_ts  = (
        content.get("pubDate") or
        item.get("providerPublishTime") or
        0
    )
    if isinstance(pub_ts, str):
        try:
            pub_ts = int(
                datetime.fromisoformat(
                    pub_ts.replace("Z", "+00:00")
                ).timestamp()
            )
        except Exception:
            pub_ts = 0

    return {
        "title":        title,
        "summary":      summary or "",
        "source":       source,
        "published_ts": int(pub_ts),
    }


def _fetch_general_news(ticker_symbol: str) -> list:
    """Fetches general financial news from yfinance ticker."""
    try:
        time.sleep(1)
        raw = yf.Ticker(ticker_symbol).news or []
        articles = []
        for item in raw[:MAX_GENERAL]:
            parsed = _parse_article(item)
            if parsed:
                articles.append(parsed)
        return articles
    except Exception as e:
        print(f"  General news fetch error: {e}")
        return []


def _fetch_specific_news(ticker_symbol: str, company_name: str) -> list:
    """
    Fetches stock-specific news via yfinance Search.
    Uses both ticker and company name for better coverage.
    """
    articles = []
    seen     = set()

    queries = [ticker_symbol, company_name] if company_name else [ticker_symbol]

    for query in queries:
        try:
            time.sleep(0.5)
            results = yf.Search(query, news_count=MAX_SPECIFIC).news or []
            for item in results:
                parsed = _parse_article(item)
                if parsed and parsed["title"] not in seen:
                    seen.add(parsed["title"])
                    articles.append(parsed)
        except Exception as e:
            print(f"  Specific news fetch error for '{query}': {e}")

    return articles[:MAX_SPECIFIC]


def _deduplicate(general: list, specific: list) -> list:
    """
    Merges two article lists and removes duplicates by title similarity.
    Specific news comes first (higher direct relevance priority).
    """
    merged = []
    seen   = set()

    for article in specific + general:
        # Normalise title for comparison
        key = article["title"].lower().strip()[:60]
        if key not in seen:
            seen.add(key)
            merged.append(article)

    return merged[:MAX_ANALYSE]


# ============================================================
# GPT Analyser
# ============================================================

def _analyse_with_gpt(articles: list, ticker: str,
                       company_name: str) -> list:
    """
    Sends articles to GPT for multi-dimension sentiment analysis.
    Returns list of GPT judgments matching article indices.
    """
    if not articles:
        return []

    article_text = ""
    for i, a in enumerate(articles):
        article_text += (
            f"\n[{i}] Source: {a['source']}\n"
            f"Title: {a['title']}\n"
            f"Summary: {a['summary'][:300]}\n"
        )

    prompt = (
        f"Stock: {ticker} ({company_name})\n"
        f"Analyse these {len(articles)} news articles.\n"
        f"{article_text}"
    )

    try:
        client   = OpenAI()
        response = client.chat.completions.create(
            model       = "gpt-4.1-mini",
            max_tokens  = 2500,
            temperature = 0.1,
            messages    = [
                {"role": "system", "content": NEWS_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )

        raw_text = response.choices[0].message.content.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        return json.loads(raw_text)

    except Exception as e:
        print(f"  GPT news analysis error: {e}")
        return []


# ============================================================
# Score Calculator
# ============================================================

def _sentiment_to_direction(sentiment: str) -> int:
    """Converts sentiment label to +1 / 0 / -1."""
    return {"positive": 1, "neutral": 0, "negative": -1}.get(sentiment, 0)


def _calc_article_score(judgment: dict, time_weight: float) -> float:
    """
    Calculates the weighted score for a single article.
    Relevance multiplier is applied here.
    """
    relevance   = RELEVANCE_WEIGHTS.get(judgment.get("relevance", "unrelated"), 0.0)
    if relevance == 0.0:
        return 0.0

    direction   = _sentiment_to_direction(judgment.get("sentiment", "neutral"))
    intensity   = INTENSITY_WEIGHTS.get(   judgment.get("intensity",   "mild"),     0.3)
    impact      = IMPACT_WEIGHTS.get(      judgment.get("impact",      "minor"),    0.5)
    scope       = SCOPE_WEIGHTS.get(       judgment.get("scope",       "macro"),    0.6)
    credibility = CREDIBILITY_WEIGHTS.get( judgment.get("credibility", "low"),      0.5)
    novelty     = NOVELTY_WEIGHTS.get(     judgment.get("novelty",     "repeat"),   0.3)
    surprise    = SURPRISE_WEIGHTS.get(    judgment.get("surprise",    "expected"), 0.5)

    multiplier = (
        relevance   *
        intensity   *
        impact      *
        scope       *
        credibility *
        novelty     *
        surprise    *
        time_weight
    )

    return direction * multiplier


def _calc_consensus(judgments: list) -> float:
    """
    Calculates consensus adjustment.
    Only activates if MIN_RELEVANT articles are available.
    Counts indirect as half-weight for consensus calculation.
    """
    # Filter to relevant articles only
    relevant = [
        j for j in judgments
        if j.get("relevance") in ("direct", "indirect")
    ]

    if len(relevant) < MIN_RELEVANT:
        return 1.0  # Not enough data -- no adjustment

    sentiments = [j.get("sentiment", "neutral") for j in relevant]
    pos   = sentiments.count("positive")
    neg   = sentiments.count("negative")
    neu   = sentiments.count("neutral")
    total = len(sentiments)

    dominant = max(pos, neg, neu)
    ratio    = dominant / total

    if ratio >= 0.75:
        return CONSENSUS_BOOST
    elif ratio <= 0.45:
        return CONSENSUS_PENALTY
    else:
        return 1.0


# ============================================================
# Master News Sentiment Function
# ============================================================

def fetch_news_sentiment(ticker_symbol: str,
                         company_name: str = "") -> dict:
    """
    Master function -- fetches and scores news sentiment.

    Fetches from two sources, deduplicates, sends to GPT,
    applies relevance + time + impact weights, normalises to 0-100.
    """
    print(f"  Fetching news sentiment for {ticker_symbol}...")

    # Step 1: Fetch from both sources
    general  = _fetch_general_news(ticker_symbol)
    specific = _fetch_specific_news(ticker_symbol, company_name)
    articles = _deduplicate(general, specific)

    if not articles:
        return {
            "score":         50,
            "direction":     "neutral",
            "article_count": 0,
            "relevant_count":0,
            "articles":      [],
            "consensus":     1.0,
            "signals":       ["No recent news found -- neutral score applied ➡️"],
        }

    # Step 2: GPT analysis
    judgments = _analyse_with_gpt(articles, ticker_symbol, company_name)
    if not judgments:
        return {
            "score":         50,
            "direction":     "neutral",
            "article_count": len(articles),
            "relevant_count":0,
            "articles":      [],
            "consensus":     1.0,
            "signals":       ["News analysis unavailable -- neutral score applied ➡️"],
        }

    # Step 3: Build index map
    judgment_map = {j["index"]: j for j in judgments if "index" in j}

    # Step 4: Calculate per-article scores
    raw_scores = []
    enriched   = []

    for i, article in enumerate(articles):
        judgment   = judgment_map.get(i, {})
        relevance  = judgment.get("relevance", "unrelated")
        impact     = judgment.get("impact", "minor")

        # Time decay with impact floor
        time_weight = _time_decay_weight(article["published_ts"], impact)
        art_score   = _calc_article_score(judgment, time_weight)
        raw_scores.append(art_score)

        enriched.append({
            "title":        article["title"],
            "source":       article["source"],
            "published_ts": article["published_ts"],
            "time_weight":  time_weight,
            "relevance":    relevance,
            "sentiment":    judgment.get("sentiment",   "neutral"),
            "intensity":    judgment.get("intensity",   "mild"),
            "impact":       impact,
            "scope":        judgment.get("scope",       "macro"),
            "credibility":  judgment.get("credibility", "medium"),
            "novelty":      judgment.get("novelty",     "followup"),
            "surprise":     judgment.get("surprise",    "expected"),
            "reason":       judgment.get("reason",      ""),
            "raw_score":    round(art_score, 4),
        })

    # Step 5: Consensus (relevant articles only, MIN_RELEVANT gate)
    consensus = _calc_consensus(judgments)

    # Step 6: Normalise to 0-100
    # Calibration: a typical "all positive, direct, normal impact" scenario
    # should produce a clearly bullish score (~70-75).
    # Typical article score = relevance(1.0) × intensity(0.6) × impact(1.0)
    #   × scope(1.0) × credibility(1.0) × novelty(1.5) × surprise(1.0) × time(1.0) = 0.9
    # With 8 relevant positive articles at typical weights: 8 × 0.9 = 7.2
    # We want that to map to ~0.5 on the norm scale → CALIBRATION_RAW = 7.2 / 0.5 = 14.4
    CALIBRATION_RAW = 14.4
    total_raw    = sum(raw_scores)
    adjusted_raw = total_raw * consensus
    norm         = adjusted_raw / CALIBRATION_RAW   # typically -1.0 to +1.0
    norm         = max(-1.0, min(1.0, norm))         # hard clamp
    score        = round(50 + norm * 50)
    score        = max(0, min(100, score))

    # Step 7: Direction label
    if score >= 60:
        direction = "bullish"
    elif score <= 40:
        direction = "bearish"
    else:
        direction = "neutral"

    # Step 8: Build signals
    direct_arts   = [a for a in enriched if a["relevance"] == "direct"]
    indirect_arts = [a for a in enriched if a["relevance"] == "indirect"]
    unrel_arts    = [a for a in enriched if a["relevance"] == "unrelated"]

    pos_count = sum(1 for a in direct_arts + indirect_arts if a["sentiment"] == "positive")
    neg_count = sum(1 for a in direct_arts + indirect_arts if a["sentiment"] == "negative")
    neu_count = sum(1 for a in direct_arts + indirect_arts if a["sentiment"] == "neutral")
    major_pos = [a for a in direct_arts + indirect_arts
                 if a["sentiment"] == "positive" and a["impact"] == "major"]
    major_neg = [a for a in direct_arts + indirect_arts
                 if a["sentiment"] == "negative" and a["impact"] == "major"]

    signals = []
    signals.append(
        f"News: {len(direct_arts)} direct, {len(indirect_arts)} indirect, "
        f"{len(unrel_arts)} unrelated (of {len(enriched)} total)"
    )
    signals.append(
        f"Relevant sentiment: {pos_count} positive, "
        f"{neg_count} negative, {neu_count} neutral"
    )
    if major_pos:
        signals.append(
            f"Major positive: {major_pos[0]['title'][:60]}... "
            f"({major_pos[0]['source']}) ✅"
        )
    if major_neg:
        signals.append(
            f"Major negative: {major_neg[0]['title'][:60]}... "
            f"({major_neg[0]['source']}) ⚠️"
        )
    if consensus >= CONSENSUS_BOOST:
        signals.append("Strong consensus across relevant news ✅")
    elif consensus <= CONSENSUS_PENALTY:
        signals.append("Mixed signals -- high disagreement across news sources ⚠️")
    elif len(direct_arts) + len(indirect_arts) < MIN_RELEVANT:
        signals.append(
            f"Low relevant article count ({len(direct_arts)+len(indirect_arts)}) "
            f"-- sentiment score has low confidence ➡️"
        )

    icon = "✅" if direction == "bullish" else ("⚠️" if direction == "bearish" else "➡️")
    signals.append(f"Overall news tone: {direction} {icon}")

    return {
        "score":          score,
        "direction":      direction,
        "article_count":  len(enriched),
        "relevant_count": len(direct_arts) + len(indirect_arts),
        "articles":       enriched,
        "consensus":      consensus,
        "signals":        signals,
    }