# ============================================================
# composite.py -- Composite Scoring Engine
# ============================================================
# Combines all analysis dimension scores into one final
# composite investment score and decision quadrant.
#
# Current dimension weights:
#   Technical   : 50%
#   Fundamental : 30%
#   Sentiment   : 20%
#
# Future dimensions (macro, etc.) will be added here.
# ============================================================

THRESHOLD = 50  # kept for reference

# ============================================================
# Per-Dimension Thresholds
# ============================================================
# Each dimension has a different threshold for what counts as
# "meaningfully positive" given its scoring characteristics.
#
# Technical   : 50 -- scores spread across full 0-100 range,
#               any score above 50 is genuinely bullish
#
# Fundamental : 55 -- scores tend to cluster in 40-80 range,
#               50 is mediocre, 55+ signals real quality
#
# Sentiment   : 60 -- centred at 50 by design, noise is high,
#               requires clearer positive signal to count as high
#
THRESHOLD_TECHNICAL   = 50
THRESHOLD_FUNDAMENTAL = 55
THRESHOLD_SENTIMENT   = 60

# ============================================================
# Dimension Weights
# ============================================================
# All active weights must sum to 1.0.

WEIGHT_TECHNICAL   = 0.50
WEIGHT_FUNDAMENTAL = 0.30
WEIGHT_SENTIMENT   = 0.20

# Future:
# WEIGHT_TECHNICAL   = 0.45
# WEIGHT_FUNDAMENTAL = 0.25
# WEIGHT_SENTIMENT   = 0.15
# WEIGHT_MACRO       = 0.15


# ============================================================
# Composite Score
# ============================================================

def get_composite(technical: int,
                  fundamental,
                  sentiment=None) -> dict:
    """
    Combines all dimension scores into one composite investment
    score with quadrant decision label.

    Args:
        technical   : Technical overall score (0-100)
        fundamental : Fundamental score (0-100) or None
        sentiment   : Sentiment score (0-100) or None

    Returns dict with:
        score    : Composite score (0-100)
        verdict  : Trend label
        color    : Hex color for UI
        quadrant : Decision label
        action   : Explanation text
        q_color  : Quadrant hex color
        q_icon   : Quadrant emoji
        weights  : Active weights used (for UI display)
    """
    # -- Determine which dimensions are available -------------
    has_fund = fundamental is not None
    has_sent = sentiment   is not None

    # -- Redistribute weights for missing dimensions ----------
    if has_fund and has_sent:
        w_tech = WEIGHT_TECHNICAL
        w_fund = WEIGHT_FUNDAMENTAL
        w_sent = WEIGHT_SENTIMENT
    elif has_fund and not has_sent:
        # Redistribute sentiment weight proportionally
        total  = WEIGHT_TECHNICAL + WEIGHT_FUNDAMENTAL
        w_tech = round(WEIGHT_TECHNICAL   / total, 4)
        w_fund = round(WEIGHT_FUNDAMENTAL / total, 4)
        w_sent = 0
    elif not has_fund and has_sent:
        total  = WEIGHT_TECHNICAL + WEIGHT_SENTIMENT
        w_tech = round(WEIGHT_TECHNICAL / total, 4)
        w_fund = 0
        w_sent = round(WEIGHT_SENTIMENT / total, 4)
    else:
        w_tech = 1.0
        w_fund = 0
        w_sent = 0

    # -- Calculate composite score ----------------------------
    comp_score = round(
        technical          * w_tech +
        (fundamental or 0) * w_fund +
        (sentiment   or 0) * w_sent
    )
    comp_score = max(0, min(100, comp_score))

    # -- Composite verdict ------------------------------------
    if comp_score >= 75:
        comp_verdict, comp_color = "Strong Uptrend",   "#00C853"
    elif comp_score >= 60:
        comp_verdict, comp_color = "Uptrend",          "#69F0AE"
    elif comp_score >= 40:
        comp_verdict, comp_color = "Neutral",          "#FFD740"
    elif comp_score >= 25:
        comp_verdict, comp_color = "Downtrend",        "#FF6D00"
    else:
        comp_verdict, comp_color = "Strong Downtrend", "#FF1744"

    # -- Decision quadrant ------------------------------------
    high_t = technical          > THRESHOLD_TECHNICAL
    high_f = (fundamental or 0) > THRESHOLD_FUNDAMENTAL
    high_s = (sentiment   or 0) > THRESHOLD_SENTIMENT

    if not has_fund and not has_sent:
        quadrant = "Technical Only"
        action   = "Only technical data available -- decision based on price signals only."
        q_color  = "#888888"
        q_icon   = "⬜"
    elif high_t and high_f and (not has_sent or high_s):
        quadrant = "Strong Buy"
        action   = "Technical, fundamental, and sentiment all agree -- highest conviction entry."
        q_color  = "#00C853"
        q_icon   = "🟢"
    elif high_t and high_f and has_sent and not high_s:
        quadrant = "Cautious Buy"
        action   = "Strong price and business fundamentals but sentiment is weak -- watch for news catalysts."
        q_color  = "#69F0AE"
        q_icon   = "🟢"
    elif high_t and not high_f:
        quadrant = "Trader Play"
        action   = "Momentum exists but business fundamentals are weak -- short term only, set a tight stop loss."
        q_color  = "#FFD740"
        q_icon   = "🟡"
    elif not high_t and high_f and high_s:
        quadrant = "Value Opportunity"
        action   = "Solid business with positive sentiment in a technical pullback -- wait for price to stabilize."
        q_color  = "#69F0AE"
        q_icon   = "🔵"
    elif not high_t and high_f and not high_s:
        quadrant = "Value Watch"
        action   = "Good fundamentals but weak technicals and sentiment -- not yet time to enter."
        q_color  = "#FF6D00"
        q_icon   = "🟠"
    else:
        quadrant = "Avoid"
        action   = "Technical, fundamental, and sentiment are all weak -- no clear edge. Stay out."
        q_color  = "#FF1744"
        q_icon   = "🔴"

    # -- Build weight label for UI ----------------------------
    if has_fund and has_sent:
        weight_label = f"Technical {int(w_tech*100)}% + Fundamental {int(w_fund*100)}% + Sentiment {int(w_sent*100)}%"
    elif has_fund:
        weight_label = f"Technical {int(w_tech*100)}% + Fundamental {int(w_fund*100)}%"
    elif has_sent:
        weight_label = f"Technical {int(w_tech*100)}% + Sentiment {int(w_sent*100)}%"
    else:
        weight_label = "Technical 100%"

    return {
        "score":        comp_score,
        "verdict":      comp_verdict,
        "color":        comp_color,
        "quadrant":     quadrant,
        "action":       action,
        "q_color":      q_color,
        "q_icon":       q_icon,
        "weight_label": weight_label,
    }