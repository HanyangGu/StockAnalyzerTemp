# ============================================================
# fundamental_scorer.py -- Fundamental Analysis Scoring Engine
# ============================================================
# Scores fundamental data on a 0-100 scale:
#   Valuation    : 25pts  (P/E, Forward P/E)
#   Profitability: 25pts  (Net Margin, ROE)
#   Growth       : 25pts  (Revenue Growth, Earnings Growth)
#   Health       : 25pts  (Debt/Equity, Current Ratio)
# ============================================================


def get_fundamental_verdict(score: int) -> tuple:
    """Converts fundamental score to verdict label and emoji."""
    if score >= 75:
        return "Strong Fundamentals", "🟢"
    elif score >= 60:
        return "Good Fundamentals",   "🟩"
    elif score >= 40:
        return "Mixed Fundamentals",  "⬜"
    elif score >= 25:
        return "Weak Fundamentals",   "🟥"
    else:
        return "Poor Fundamentals",   "🔴"


def score_fundamentals(data: dict) -> dict:
    """
    Scores fundamental data on a 0-100 scale.

    Scoring breakdown:
      Valuation    : 25pts  (P/E, Forward P/E)
      Profitability: 25pts  (Net Margin, ROE)
      Growth       : 25pts  (Revenue Growth, Earnings Growth)
      Health       : 25pts  (Debt/Equity, Current Ratio)
    """
    score   = 0
    signals = []

    # -- Valuation (max 25pts) --------------------------------
    pe = data.get("pe_ratio")
    if pe is not None:
        if pe < 15:
            score += 15
            signals.append(f"P/E ({pe}) -- undervalued ✅")
        elif pe < 25:
            score += 12
            signals.append(f"P/E ({pe}) -- fairly valued ✅")
        elif pe < 40:
            score += 7
            signals.append(f"P/E ({pe}) -- slightly expensive ⚠️")
        else:
            score += 2
            signals.append(f"P/E ({pe}) -- expensive ⚠️")
    else:
        signals.append("P/E -- no data ➡️")

    fpe = data.get("forward_pe")
    if fpe is not None:
        if fpe < 0:
            score += 0
            signals.append(f"Forward P/E ({fpe}) -- negative, future losses expected ⚠️")
        elif fpe < 15:
            score += 10
            signals.append(f"Forward P/E ({fpe}) -- cheap ✅")
        elif fpe < 25:
            score += 7
            signals.append(f"Forward P/E ({fpe}) -- reasonable ✅")
        elif fpe < 40:
            score += 3
            signals.append(f"Forward P/E ({fpe}) -- pricey ⚠️")
        else:
            score += 0
            signals.append(f"Forward P/E ({fpe}) -- very expensive ⚠️")
    else:
        signals.append("Forward P/E -- no data ➡️")

    # -- Profitability (max 25pts) ----------------------------
    nm = data.get("net_margin")
    if nm is not None:
        if nm > 20:
            score += 13
            signals.append(f"Net Margin ({nm}%) -- excellent ✅")
        elif nm > 10:
            score += 10
            signals.append(f"Net Margin ({nm}%) -- good ✅")
        elif nm > 0:
            score += 5
            signals.append(f"Net Margin ({nm}%) -- thin ⚠️")
        else:
            score += 0
            signals.append(f"Net Margin ({nm}%) -- losing money ⚠️")
    else:
        signals.append("Net Margin -- no data ➡️")

    roe = data.get("roe")
    if roe is not None:
        if roe > 20:
            score += 12
            signals.append(f"ROE ({roe}%) -- excellent ✅")
        elif roe > 10:
            score += 8
            signals.append(f"ROE ({roe}%) -- good ✅")
        elif roe > 0:
            score += 3
            signals.append(f"ROE ({roe}%) -- weak ⚠️")
        else:
            score += 0
            signals.append(f"ROE ({roe}%) -- negative ⚠️")
    else:
        signals.append("ROE -- no data ➡️")

    # -- Growth (max 25pts) -----------------------------------
    rg = data.get("revenue_growth")
    revenue_declining = rg is not None and rg <= 0

    if rg is not None:
        if rg > 20:
            score += 13
            signals.append(f"Revenue Growth ({rg}%) -- strong ✅")
        elif rg > 10:
            score += 10
            signals.append(f"Revenue Growth ({rg}%) -- good ✅")
        elif rg > 0:
            score += 5
            signals.append(f"Revenue Growth ({rg}%) -- slow ⚠️")
        else:
            score += 0
            signals.append(f"Revenue Growth ({rg}%) -- declining ⚠️")
    else:
        signals.append("Revenue Growth -- no data ➡️")

    eg = data.get("earnings_growth")
    if eg is not None:
        # Cap earnings growth score at 3pts when revenue is declining
        # Earnings growth on a shrinking business is often a one-time anomaly
        if revenue_declining and eg > 0:
            score += 3
            signals.append(
                f"Earnings Growth ({eg}%) -- but revenue declining, "
                f"treat with caution ⚠️"
            )
        elif eg > 20:
            score += 12
            signals.append(f"Earnings Growth ({eg}%) -- strong ✅")
        elif eg > 10:
            score += 8
            signals.append(f"Earnings Growth ({eg}%) -- good ✅")
        elif eg > 0:
            score += 3
            signals.append(f"Earnings Growth ({eg}%) -- slow ⚠️")
        else:
            score += 0
            signals.append(f"Earnings Growth ({eg}%) -- declining ⚠️")
    else:
        signals.append("Earnings Growth -- no data ➡️")

    # -- Financial Health (max 25pts) -------------------------
    de = data.get("debt_to_equity")
    if de is not None:
        if de < 50:
            score += 13
            signals.append(f"Debt/Equity ({de}) -- low debt ✅")
        elif de < 100:
            score += 9
            signals.append(f"Debt/Equity ({de}) -- moderate debt ✅")
        elif de < 200:
            score += 4
            signals.append(f"Debt/Equity ({de}) -- high debt ⚠️")
        else:
            score += 0
            signals.append(f"Debt/Equity ({de}) -- very high debt ⚠️")
    else:
        signals.append("Debt/Equity -- no data ➡️")

    cr = data.get("current_ratio")
    if cr is not None:
        if cr > 2:
            score += 12
            signals.append(f"Current Ratio ({cr}) -- very liquid ✅")
        elif cr > 1.5:
            score += 9
            signals.append(f"Current Ratio ({cr}) -- healthy ✅")
        elif cr > 1:
            score += 5
            signals.append(f"Current Ratio ({cr}) -- adequate ⚠️")
        else:
            score += 0
            signals.append(f"Current Ratio ({cr}) -- liquidity risk ⚠️")
    else:
        signals.append("Current Ratio -- no data ➡️")

    score         = max(0, min(100, score))
    verdict, icon = get_fundamental_verdict(score)

    return {
        "score":   score,
        "verdict": verdict,
        "icon":    icon,
        "signals": signals,
    }
