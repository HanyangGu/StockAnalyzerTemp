# ============================================================
# app.py -- Streamlit UI
# ============================================================
# Main entry point for the Stock Analysis Chatbot.
# Handles all rendering and user interaction.
# All business logic is imported from separate modules.
# ============================================================

import os

import pandas as pd
import streamlit as st
import yfinance as yf

from engine.ai import StockChatbot
from scoring.composite import get_composite


# ============================================================
# Score Color Helper
# ============================================================

def get_score_color(score: int) -> str:
    if score >= 75:   return "#00C853"
    elif score >= 60: return "#69F0AE"
    elif score >= 40: return "#FFD740"
    elif score >= 25: return "#FF6D00"
    else:             return "#FF1744"



# ============================================================
# Render Functions
# ============================================================

def render_score_cards(data: dict):
    """
    Renders three main dimension score cards:
      Left   : Technical Overall score
      Centre : Fundamental score
      Right  : Sentiment score
    Detail breakdowns are in the dropdowns below.
    """
    fund       = data.get("fundamental", {})
    sent       = data.get("sentiment",   {})
    fund_score = fund.get("score")
    sent_score = sent.get("score")
    tech       = data["overall"]
    tcolor     = get_score_color(tech["score"])

    col_tech, col_fund, col_sent = st.columns(3)

    # -- Technical Overall card -------------------------------
    with col_tech:
        st.markdown(f"""
        <div style="
            border: 2px solid {tcolor};
            border-radius: 14px;
            padding: 28px 20px;
            text-align: center;
            background: rgba(0,0,0,0.2);">
            <div style="color:#aaa;font-size:14px;margin-bottom:4px;">
                📊 Technical
            </div>
            <div style="
                color:{tcolor};
                font-size:56px;
                font-weight:800;
                line-height:1.1;">
                {tech["score"]}
            </div>
            <div style="color:#aaa;font-size:11px;">out of 100</div>
            <div style="margin-top:10px;font-size:15px;font-weight:600;color:{tcolor};">
                {tech["icon"]} {tech["verdict"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -- Fundamental score card -------------------------------
    with col_fund:
        if fund_score is not None:
            fcolor   = get_score_color(fund_score)
            ficon    = fund.get("icon",    "➡️")
            fverdict = fund.get("verdict", "")
            fdisp    = fund_score
        else:
            fcolor   = "#888888"
            ficon    = "➡️"
            fverdict = "No data"
            fdisp    = "--"

        st.markdown(f"""
        <div style="
            border: 2px solid {fcolor};
            border-radius: 14px;
            padding: 28px 20px;
            text-align: center;
            background: rgba(0,0,0,0.2);">
            <div style="color:#aaa;font-size:14px;margin-bottom:4px;">
                🏦 Fundamental
            </div>
            <div style="
                color:{fcolor};
                font-size:56px;
                font-weight:800;
                line-height:1.1;">
                {fdisp}
            </div>
            <div style="color:#aaa;font-size:11px;">out of 100</div>
            <div style="margin-top:10px;font-size:15px;font-weight:600;color:{fcolor};">
                {ficon} {fverdict}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -- Sentiment score card ---------------------------------
    with col_sent:
        if sent_score is not None:
            scolor   = get_score_color(sent_score)
            sicon    = sent.get("icon",    "➡️")
            sverdict = sent.get("verdict", "")
            sdisp    = sent_score
        else:
            scolor   = "#888888"
            sicon    = "➡️"
            sverdict = "No data"
            sdisp    = "--"

        st.markdown(f"""
        <div style="
            border: 2px solid {scolor};
            border-radius: 14px;
            padding: 28px 20px;
            text-align: center;
            background: rgba(0,0,0,0.2);">
            <div style="color:#aaa;font-size:14px;margin-bottom:4px;">
                🗞️ Sentiment
            </div>
            <div style="
                color:{scolor};
                font-size:56px;
                font-weight:800;
                line-height:1.1;">
                {sdisp}
            </div>
            <div style="color:#aaa;font-size:11px;">out of 100</div>
            <div style="margin-top:10px;font-size:15px;font-weight:600;color:{scolor};">
                {sicon} {sverdict}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_price_line(data: dict):
    """Renders price with colored up/down arrow."""
    price  = data["current_price"]
    change = data["price_change"]
    pct    = data["price_change_pct"]
    arrow  = "▲" if change >= 0 else "▼"
    color  = "#00C853" if change >= 0 else "#FF1744"
    st.markdown(f"""
    <div style="font-size:16px;margin:10px 0;">
        <b>Price:</b> ${price}
        <span style="color:{color};">
            {arrow} {abs(change)} ({abs(pct)}%)
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_technical_dropdown(data: dict):
    """
    Expandable Technical breakdown dropdown.
    Shows three time horizon score cards + signal lists.
    """
    with st.expander("📊 Technical Breakdown"):

        # Three time horizon score cards
        horizons = [
            ("Short term",  data["short_term"]),
            ("Mid term",    data["mid_term"]),
            ("Long term",   data["long_term"]),
        ]
        cols = st.columns(3)
        for col, (label, h) in zip(cols, horizons):
            color = get_score_color(h["score"])
            with col:
                st.markdown(f"""
                <div style="
                    border: 1px solid {color};
                    border-radius: 10px;
                    padding: 16px;
                    text-align: center;
                    background: rgba(0,0,0,0.2);
                    margin-bottom: 16px;">
                    <div style="color:#aaa;font-size:12px;">
                        {label}
                    </div>
                    <div style="
                        color:{color};
                        font-size:38px;
                        font-weight:700;
                        line-height:1.1;">
                        {h["score"]}
                    </div>
                    <div style="color:#aaa;font-size:10px;">
                        out of 100
                    </div>
                    <div style="margin-top:6px;font-size:13px;color:{color};">
                        {h["icon"]} {h["verdict"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Signal lists beneath each card
        sig_cols = st.columns(3)
        for col, (label, h) in zip(sig_cols, horizons):
            with col:
                st.markdown(f"**{label} signals**")
                for s in h["signals"]:
                    st.markdown(f"- {s}")


def render_fundamental_dropdown(data: dict):
    """
    Expandable Fundamental breakdown dropdown.
    Shows fundamental signals + four data columns.
    """
    fund    = data.get("fundamental", {})
    details = data.get("fund_details", {})

    if not details:
        return

    with st.expander("🏦 Fundamental Breakdown"):

        # Fundamental signals
        if fund.get("signals"):
            st.markdown("**Fundamental Signals**")
            for s in fund["signals"]:
                st.markdown(f"- {s}")
            st.markdown("")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Valuation**")
            v = details.get("valuation", {})
            st.metric("P/E Ratio",   v.get("pe_ratio")       or "N/A")
            st.metric("Forward P/E", v.get("forward_pe")     or "N/A")
            st.metric("Price/Book",  v.get("price_to_book")  or "N/A")
            st.metric("Price/Sales", v.get("price_to_sales") or "N/A")

        with col2:
            st.markdown("**Profitability**")
            p = details.get("profitability", {})
            st.metric("Gross Margin", f"{p.get('gross_margin')}%"  if p.get("gross_margin")  else "N/A")
            st.metric("Net Margin",   f"{p.get('net_margin')}%"    if p.get("net_margin")    else "N/A")
            st.metric("ROE",          f"{p.get('roe')}%"           if p.get("roe")           else "N/A")
            st.metric("ROA",          f"{p.get('roa')}%"           if p.get("roa")           else "N/A")

        with col3:
            st.markdown("**Growth**")
            g = details.get("growth", {})
            st.metric("Revenue Growth",  f"{g.get('revenue_growth')}%"  if g.get("revenue_growth")  else "N/A")
            st.metric("Earnings Growth", f"{g.get('earnings_growth')}%" if g.get("earnings_growth") else "N/A")
            st.metric("EPS (TTM)",       g.get("eps")         or "N/A")
            st.metric("Forward EPS",     g.get("forward_eps") or "N/A")

        with col4:
            st.markdown("**Health & Analyst**")
            h = details.get("health", {})
            a = details.get("analyst", {})
            st.metric("Debt/Equity",    h.get("debt_to_equity") or "N/A")
            st.metric("Current Ratio",  h.get("current_ratio")  or "N/A")
            st.metric("Analyst Target", f"${a.get('target_price')}" if a.get("target_price") else "N/A")
            upside = a.get("upside_pct")
            st.metric(
                "Upside Potential",
                f"{upside}%" if upside is not None else "N/A",
                delta=f"{upside}%" if upside is not None else None
            )


def render_sentiment_dropdown(data: dict):
    """
    Expandable Sentiment breakdown dropdown.
    Layout:
      1. News signals
      2. Analyst card + signals + distribution + targets
      3. Insider card + signals + transaction list
      4. News card + articles
    """
    sent      = data.get("sentiment", {})
    breakdown = sent.get("breakdown", {})
    news      = breakdown.get("news",    {})
    analyst   = breakdown.get("analyst", {})
    insider   = breakdown.get("insider", {})
    articles  = news.get("articles",      [])
    txns      = insider.get("transactions", [])

    if not sent:
        return

    def _score_card(col, label, score, direction):
        color   = get_score_color(score) if score is not None else "#888888"
        disp    = score if score is not None else "--"
        if score is None:
            icon, verdict = "➡️", "No data"
        else:
            icon    = "✅" if direction == "bullish" else ("⚠️" if direction == "bearish" else "➡️")
            verdict = "Bullish" if direction == "bullish" else ("Bearish" if direction == "bearish" else "Neutral")
        with col:
            st.markdown(
                f"<div style='border:1px solid {color};border-radius:10px;"
                f"padding:16px;text-align:center;background:rgba(0,0,0,0.2);margin-bottom:16px;'>"
                f"<div style='color:#aaa;font-size:12px;'>{label}</div>"
                f"<div style='color:{color};font-size:38px;font-weight:700;line-height:1.1;'>{disp}</div>"
                f"<div style='color:#aaa;font-size:10px;'>out of 100</div>"
                f"<div style='margin-top:6px;font-size:13px;color:{color};'>{icon} {verdict}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    with st.expander("🗞️ Sentiment Breakdown"):

        # -- 1. News signals ----------------------------------
        news_signals = news.get("signals", [])
        if news_signals:
            st.markdown("**News Signals**")
            for s in news_signals:
                st.markdown(f"- {s}")

        # -- 2. Analyst section -------------------------------
        st.markdown("---")
        analyst_score = analyst.get("score") if analyst else None
        analyst_dir   = analyst.get("direction", "neutral") if analyst else "neutral"

        col_card, col_detail = st.columns([1, 2])
        _score_card(col_card, "📋 Analyst", analyst_score, analyst_dir)

        with col_detail:
            if analyst and analyst.get("rating_count", 0) > 0:
                for s in analyst.get("signals", []):
                    st.markdown(f"- {s}")
                st.markdown("")
                summary = analyst.get("summary", {})
                if summary:
                    sb = summary.get("strong_buy",  0)
                    b  = summary.get("buy",         0)
                    h  = summary.get("hold",        0)
                    s  = summary.get("sell",        0)
                    ss = summary.get("strong_sell", 0)
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1: st.metric("Strong Buy",  sb)
                    with c2: st.metric("Buy",         b)
                    with c3: st.metric("Hold",        h)
                    with c4: st.metric("Sell",        s)
                    with c5: st.metric("Strong Sell", ss)
                targets = analyst.get("targets", {})
                if targets.get("mean"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Mean Target", f"${round(targets['mean'], 2)}")
                    with c2:
                        st.metric("High Target", f"${round(targets['high'], 2)}" if targets.get("high") else "N/A")
                    with c3:
                        st.metric("Low Target",  f"${round(targets['low'],  2)}" if targets.get("low")  else "N/A")
            else:
                st.caption("No analyst rating data available.")

        # -- 3. Insider section -------------------------------
        st.markdown("---")
        insider_score = insider.get("score") if insider else None
        insider_dir   = insider.get("direction", "neutral") if insider else "neutral"

        col_card2, col_txns = st.columns([1, 2])
        _score_card(col_card2, "👤 Insider", insider_score, insider_dir)

        with col_txns:
            if insider and insider.get("transaction_count", 0) > 0:
                for s in insider.get("signals", []):
                    st.markdown(f"- {s}")
                st.markdown("")
                if txns:
                    st.markdown("**Transactions**")
                    for t in txns:
                        is_buy    = t.get("is_buy",   False)
                        is_sell   = t.get("is_sell",  False)
                        icon      = "✅" if is_buy else ("⚠️" if is_sell else "➡️")
                        name      = t.get("insider",          "Unknown")
                        title_t   = t.get("title",            "")
                        txn_type  = t.get("transaction_type", "")
                        shares    = t.get("shares",  0)
                        value     = t.get("value",   0)
                        date_str  = t.get("date_str", "")
                        t_weight  = t.get("time_weight", 1.0)
                        signal    = t.get("signal",   "neutral")
                        strength  = t.get("strength", "weak")
                        reason    = t.get("reason",   "")
                        value_str  = f"USD {value:,.0f}" if value else "N/A"
                        shares_str = f"{shares:,}" if shares else "N/A"
                        st.markdown(
                            f"<div style='margin-bottom:12px;'>"
                            f"{icon} <b>{name}</b> — <span style='color:#aaa;font-size:12px;'>{title_t}</span><br>"
                            f"<span style='font-size:12px;'>{txn_type} &nbsp;|&nbsp; {shares_str} shares &nbsp;|&nbsp; {value_str}</span><br>"
                            f"<span style='color:#888;font-size:11px;'>{date_str} &nbsp;|&nbsp; Time weight: {t_weight} &nbsp;|&nbsp; Signal: {signal} ({strength})</span><br>"
                            f"<span style='color:#aaa;font-size:11px;font-style:italic;'>{reason}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
            else:
                st.caption("No insider transaction data available.")

        # -- 4. News section ----------------------------------
        st.markdown("---")
        news_score = news.get("score", 50)
        news_dir   = news.get("direction", "neutral")

        col_card3, col_articles = st.columns([1, 2])
        _score_card(col_card3, "🗞️ News", news_score, news_dir)

        with col_articles:
            if articles:
                for a in articles:
                    relevance = a.get("relevance", "unrelated")
                    sentiment_a = a.get("sentiment", "neutral")
                    if relevance == "unrelated":
                        icon, opacity, rel_tag = "⬜", "0.35", " *(unrelated)*"
                    elif relevance == "indirect":
                        icon    = "↗️" if sentiment_a == "positive" else ("↘️" if sentiment_a == "negative" else "➡️")
                        opacity, rel_tag = "0.75", " *(indirect)*"
                    else:
                        icon    = "✅" if sentiment_a == "positive" else ("⚠️" if sentiment_a == "negative" else "➡️")
                        opacity, rel_tag = "1.0", ""
                    title_a = a.get("title",       "")[:70]
                    source  = a.get("source",      "")
                    impact  = a.get("impact",      "normal")
                    time_w  = a.get("time_weight", 1.0)
                    reason_a = a.get("reason",     "")
                    st.markdown(
                        f"<div style='opacity:{opacity};margin-bottom:10px;'>"
                        f"{icon} <b>{title_a}</b>{rel_tag}<br>"
                        f"<span style='color:#888;font-size:11px;'>"
                        f"{source} &nbsp;|&nbsp; Impact: {impact} &nbsp;|&nbsp; Time weight: {time_w}"
                        f"</span><br>"
                        f"<span style='color:#aaa;font-size:11px;font-style:italic;'>{reason_a}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No news articles available.")

def render_decision_panel(data: dict):
    """
    Renders the composite investment score + quadrant decision panel.
    This is the main 'should I invest?' answer.
    """
    tech_score = data["overall"]["score"]
    fund_score = data.get("fundamental", {}).get("score")
    sent_score = data.get("sentiment",   {}).get("score")
    comp       = get_composite(tech_score, fund_score, sent_score)

    st.markdown("---")
    st.markdown("#### Investment Decision")

    col_score, col_quad = st.columns([1, 2])

    # Left: Composite score card
    with col_score:
        st.markdown(f"""
        <div style="
            border: 2px solid {comp['color']};
            border-radius: 16px;
            padding: 24px 16px;
            text-align: center;
            background: rgba(0,0,0,0.25);">
            <div style="color:#aaa;font-size:13px;margin-bottom:4px;">
                Composite Score
            </div>
            <div style="color:#aaa;font-size:11px;margin-bottom:6px;">
                {comp['weight_label']}
            </div>
            <div style="
                color:{comp['color']};
                font-size:56px;
                font-weight:800;
                line-height:1.1;">
                {comp['score']}
            </div>
            <div style="color:#aaa;font-size:11px;">out of 100</div>
            <div style="
                margin-top:10px;
                font-size:16px;
                font-weight:600;
                color:{comp['color']};">
                {comp['verdict']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Right: Quadrant label + action
    with col_quad:
        st.markdown(f"""
        <div style="
            border: 2px solid {comp['q_color']};
            border-radius: 16px;
            padding: 24px 20px;
            background: rgba(0,0,0,0.25);
            height: 100%;">
            <div style="color:#aaa;font-size:13px;margin-bottom:8px;">
                Decision
            </div>
            <div style="
                font-size:28px;
                font-weight:700;
                color:{comp['q_color']};
                margin-bottom:12px;">
                {comp['q_icon']} {comp['quadrant']}
            </div>
            <div style="
                color:#ccc;
                font-size:15px;
                line-height:1.6;">
                {comp['action']}
            </div>
            <div style="
                margin-top:16px;
                color:#666;
                font-size:11px;">
                Technical: {tech_score}/100 &nbsp;|&nbsp;
                Fundamental: {fund_score if fund_score is not None else 'N/A'}/100 &nbsp;|&nbsp;
                Sentiment: {sent_score if sent_score is not None else 'N/A'}/100
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_backtest_comparison(data: dict):
    """
    Fetches current live price and compares to backtest price
    to validate whether the app verdict was correct.
    """
    st.markdown("---")
    st.markdown("#### Backtest Validation")

    try:
        current    = yf.Ticker(data["ticker"]).info
        live_price = (
            current.get("currentPrice") or
            current.get("regularMarketPrice")
        )
        past_price = data["current_price"]

        if live_price and past_price:
            price_change     = round(live_price - past_price, 2)
            price_change_pct = round(
                (price_change / past_price) * 100, 2
            )
            arrow = "▲" if price_change >= 0 else "▼"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Price on analysis date",
                    f"${past_price}"
                )
            with col2:
                st.metric(
                    "Current price today",
                    f"${live_price}"
                )
            with col3:
                st.metric(
                    "Actual move since then",
                    f"{arrow} {abs(price_change_pct)}%",
                    delta=f"{price_change_pct}%"
                )

            # Use composite decision for validation -- more accurate than technical alone
            fund_score = data.get("fundamental", {}).get("score")
            sent_score = data.get("sentiment",   {}).get("score")
            comp       = get_composite(data["overall"]["score"], fund_score, sent_score)
            decision   = comp["quadrant"]  # Strong Buy / Value Opportunity / Trader Play / Avoid
            verdict    = data["overall"]["verdict"]  # kept for display only
            st.markdown("")

            bullish_decision = decision in ["Strong Buy", "Value Opportunity", "Trader Play"]
            bearish_decision = decision == "Avoid"

            if price_change_pct > 5 and bullish_decision:
                st.success(
                    f"✅ Verdict was CORRECT -- "
                    f"App signaled {decision} and stock "
                    f"rose {price_change_pct}% since then"
                )
            elif price_change_pct < -5 and bearish_decision:
                st.success(
                    f"✅ Verdict was CORRECT -- "
                    f"App signaled {decision} and stock "
                    f"fell {abs(price_change_pct)}% since then"
                )
            elif -5 <= price_change_pct <= 5:
                st.info(
                    f"➡️ Verdict was NEUTRAL -- "
                    f"Stock moved only {price_change_pct}% "
                    f"since analysis date"
                )
            else:
                st.error(
                    f"❌ Verdict was INCORRECT -- "
                    f"App signaled {decision} but stock "
                    f"{'rose' if price_change > 0 else 'fell'} "
                    f"{abs(price_change_pct)}% since then"
                )

    except Exception as e:
        st.warning(f"Could not fetch current price for comparison: {e}")


def render_single_analysis(data: dict, summary: str):
    """Renders full single stock analysis panel."""
    st.markdown("---")

    # Backtest badge
    if data.get("backtest_date") and \
       data["backtest_date"] != "live":
        st.warning(
            f"📅 Backtesting as of: **{data['backtest_date']}** "
            f"-- This is historical analysis not live data"
        )

    ticker = data['ticker']
    name   = data.get('company', ticker)
    title  = ticker if name == ticker else f"{ticker} -- {name}"
    st.subheader(title)
    render_price_line(data)
    st.markdown("")
    render_score_cards(data)
    st.markdown("")
    render_decision_panel(data)
    st.markdown("")
    render_technical_dropdown(data)
    render_fundamental_dropdown(data)
    render_sentiment_dropdown(data)
    st.markdown("")

    # Backtest validation panel
    if data.get("backtest_date") and \
       data["backtest_date"] != "live":
        render_backtest_comparison(data)

    st.info(summary)


def render_price_only(data: dict, summary: str):
    """Renders price data only panel with 8 metrics."""
    st.markdown("---")
    ticker = data['ticker']
    name   = data['name']
    # Avoid "GC=F -- GC=F" when Yahoo returns no longName
    title  = ticker if name == ticker else f"{ticker} -- {name}"
    st.subheader(title)
    render_price_line(data)

    def fmt_mcap(v):
        if not v:      return "N/A"
        if v >= 1e12:  return f"${v/1e12:.2f}T"
        if v >= 1e9:   return f"${v/1e9:.2f}B"
        if v >= 1e6:   return f"${v/1e6:.2f}M"
        return f"${v:,.0f}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Open",      f"${data['open']}")
    with col2:
        st.metric("Day High",  f"${data['day_high']}")
    with col3:
        st.metric("Day Low",   f"${data['day_low']}")
    with col4:
        st.metric("Market Cap", fmt_mcap(data['market_cap']))

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(
            "Volume",
            f"{data['volume']:,.0f}"
            if data['volume'] else "N/A"
        )
    with col6:
        st.metric("P/E Ratio", round(data['pe_ratio'], 2) if data['pe_ratio'] else "N/A")
    with col7:
        st.metric(
            "52W High",
            f"${data['52w_high']}"
            if data['52w_high'] else "N/A"
        )
    with col8:
        st.metric(
            "52W Low",
            f"${data['52w_low']}"
            if data['52w_low'] else "N/A"
        )

    st.markdown("")
    st.info(summary)


def render_comparison(data: dict, summary: str):
    """Renders multi-stock comparison panel with table and best picks."""
    st.markdown("---")
    st.subheader("Stock Comparison")

    # Ranking cards
    st.markdown("#### Ranking")
    for r in data["ranking"]:
        color = get_score_color(r["score"])
        st.markdown(f"""
        <div style="
            border: 1px solid {color};
            border-radius: 8px;
            padding: 12px 20px;
            margin: 6px 0;
            display: flex;
            align-items: center;
            gap: 16px;">
            <span style="font-size:24px;">{r["medal"]}</span>
            <span style="font-size:18px;font-weight:700;">
                {r["ticker"]}
            </span>
            <span style="color:#aaa;">
                {r["company"]}
            </span>
            <span style="
                margin-left:auto;
                color:{color};
                font-weight:700;">
                {r["score"]}/100 -- {r["verdict"]} {r["icon"]}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Comparison table
    st.markdown("")
    st.markdown("#### Comparison Table")

    table   = data["comparison_table"]
    tickers = list(table.keys())

    rows = {
        "Price":        [f"${table[t]['price']}"                               for t in tickers],
        "Change":       [f"{table[t]['change_pct']}%"                          for t in tickers],
        "Short term":   [f"{table[t]['short_term']}/100"                       for t in tickers],
        "Mid term":     [f"{table[t]['mid_term']}/100"                         for t in tickers],
        "Long term":    [f"{table[t]['long_term']}/100"                        for t in tickers],
        "Overall":      [f"{table[t]['overall']}/100"                          for t in tickers],
        "Fundamentals": [
            f"{table[t]['fundamental']}/100" if table[t]['fundamental'] is not None
            else "N/A"
            for t in tickers
        ],
        "Sentiment":    [
            f"{table[t]['sentiment']}/100" if table[t].get('sentiment') is not None
            else "N/A"
            for t in tickers
        ],
        "Composite":    [
            f"{get_composite(table[t]['overall'], table[t]['fundamental'], table[t].get('sentiment'))['score']}/100"
            for t in tickers
        ],
        "Decision":     [
            get_composite(table[t]['overall'], table[t]['fundamental'], table[t].get('sentiment'))['quadrant']
            for t in tickers
        ],
        "RSI":          [f"{table[t]['rsi']} {table[t]['rsi_signal']}"         for t in tickers],
        "Stochastic":   [f"{table[t]['stoch']} {table[t]['stoch_signal']}"     for t in tickers],
        "ROC":          [f"{table[t]['roc']}% {table[t]['roc_signal']}"        for t in tickers],
        "MACD":         [f"{table[t]['macd']} {table[t]['macd_signal']}"       for t in tickers],
        "MA20":         [f"${table[t]['ma20']} {table[t]['ma20_signal']}"      for t in tickers],
        "MA50":         [f"${table[t]['ma50']} {table[t]['ma50_signal']}"      for t in tickers],
        "MA200":        [f"${table[t]['ma200']} {table[t]['ma200_signal']}"    for t in tickers],
        "Golden Cross": [f"{table[t]['golden_cross']} {table[t]['golden_signal']}" for t in tickers],
        "Bollinger":    [f"{table[t]['bb_pct']}% {table[t]['bb_signal']}"     for t in tickers],
        "ATR":          [f"${table[t]['atr']} {table[t]['atr_signal']}"        for t in tickers],
        "Volume":       [f"{table[t]['volume']}x {table[t]['volume_signal']}"  for t in tickers],
    }

    df = pd.DataFrame(rows, index=tickers).T
    st.dataframe(df, use_container_width=True)

    # Decision panels per stock
    st.markdown("")
    st.markdown("#### Investment Decision")
    dec_cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        comp = get_composite(
            table[t]["overall"],
            table[t]["fundamental"],
            table[t].get("sentiment"),
        )
        with dec_cols[i]:
            st.markdown(f"""
            <div style="
                border: 2px solid {comp['q_color']};
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                background: rgba(0,0,0,0.2);">
                <div style="font-size:20px;font-weight:700;
                    color:#fff;margin-bottom:4px;">{t}</div>
                <div style="font-size:13px;color:#aaa;
                    margin-bottom:10px;">
                    Composite: {comp['score']}/100
                </div>
                <div style="font-size:18px;font-weight:700;
                    color:{comp['q_color']};margin-bottom:8px;">
                    {comp['q_icon']} {comp['quadrant']}
                </div>
                <div style="font-size:12px;color:#bbb;
                    line-height:1.5;">
                    {comp['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Best picks
    st.markdown("")
    st.markdown("#### Best Picks")
    bp = data["best_picks"]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Best Overall",       bp["overall"])
    with col2:
        st.metric("Best Short term",    bp["short_term"])
    with col3:
        st.metric("Best Mid term",      bp["mid_term"])
    with col4:
        st.metric("Best Long term",     bp["long_term"])
    with col5:
        st.metric("Lowest Risk",        bp["lowest_risk"])
    with col6:
        st.metric("Best Fundamentals",  bp.get("fundamentals", "N/A"))

    st.markdown("")
    st.info(summary)


# ============================================================
# Main App
# ============================================================

def main():

    st.set_page_config(
        page_title = "Stock Analysis Chatbot",
        page_icon  = "📈",
        layout     = "wide",
    )

    # -- Session state initialization -------------------------
    if "bot" not in st.session_state:
        st.session_state.bot            = None
        st.session_state.messages       = []
        st.session_state.last_result    = None
        st.session_state.selected_model = "gpt-4.1-mini"
        st.session_state.current_model  = None
        st.session_state.api_key        = None
        st.session_state.backtest_date  = None

    # -- Sidebar ----------------------------------------------
    with st.sidebar:
        st.title("📈 Stock Chatbot")
        st.markdown("---")

        # API Key
        st.markdown("#### OpenAI API Key")
        api_key = st.text_input(
            label       = "Enter your API key",
            type        = "password",
            placeholder = "sk-...",
            help        = "Your key is never stored or shared."
        )

        if api_key:
            st.session_state.api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API key set ✅")
        else:
            st.session_state.api_key = None
            st.warning("Please enter your API key to continue.")

        st.markdown("---")

        # Model selector
        st.markdown("#### Model Selection")
        model_options = {
            "GPT-4o (Best quality)":      "gpt-4o",
            "GPT-4.1 (Latest)":           "gpt-4.1",
            "GPT-4.1-mini (Recommended)": "gpt-4.1-mini",
            "GPT-4o-mini (Fast & cheap)": "gpt-4o-mini",
            "GPT-3.5-turbo (Basic)":      "gpt-3.5-turbo",
        }
        selected_model = st.selectbox(
            label   = "Choose AI model",
            options = list(model_options.keys()),
            index   = 2,
            help    = (
                "Higher quality models give better summaries "
                "but have lower rate limits. "
                "Use mini models for testing."
            )
        )
        st.session_state.selected_model = model_options[selected_model]

        if "mini" in st.session_state.selected_model or \
           "turbo" in st.session_state.selected_model:
            st.success("✅ High rate limits -- good for testing")
        else:
            st.warning("⚠️ Lower rate limits -- use sparingly")

        st.markdown("---")

        # Backtesting mode
        st.markdown("#### Backtesting Mode")

        # Initialize backtest toggle state if not set
        if "backtest_mode" not in st.session_state:
            st.session_state.backtest_mode = False

        backtest_mode = st.toggle(
            "Enable backtesting",
            key  = "backtest_mode",
            help = "Analyze a stock as of a specific past date"
        )

        if backtest_mode:
            # Initialize date if not set
            if "backtest_date_value" not in st.session_state:
                st.session_state.backtest_date_value = (
                    pd.Timestamp.now() - pd.DateOffset(months=6)
                ).date()

            backtest_date = st.date_input(
                label     = "Select analysis date",
                value     = st.session_state.backtest_date_value,
                max_value = pd.Timestamp.now() - pd.DateOffset(days=1),
                min_value = pd.Timestamp("2021-01-01"),
                key       = "backtest_date_picker",
                help      = "App will analyze the stock as of this date"
            )
            st.session_state.backtest_date_value = backtest_date
            st.session_state.backtest_date       = str(backtest_date)
            print(f"Backtest date set: {st.session_state.backtest_date}")
            st.info(
                f"Analyzing as of: **{backtest_date}**\n\n"
                f"Compare verdict to actual price movement after this date."
            )
        else:
            st.session_state.backtest_date = None

        st.markdown("---")

        # Example queries
        st.markdown("#### Example Queries")
        examples = [
            "Give me real time data for Apple",
            "Is NVDA a good stock to buy?",
            "Run technical analysis on Tesla",
            "Show full analysis for AMD",
            "Compare NVDA, AMD and Intel",
            "Which of Apple or Microsoft is a better buy?",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state.pending_input = ex

        st.markdown("---")

        # Token usage
        st.markdown("#### Session Usage")
        if st.session_state.bot is not None:
            bot = st.session_state.bot
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total tokens",    f"{bot.total_tokens:,}")
                st.metric("Requests",        bot.total_requests)
            with col2:
                st.metric("Prompt tokens",   f"{bot.prompt_tokens:,}")
                st.metric("Response tokens", f"{bot.completion_tokens:,}")

            if bot.total_tokens > 150000:
                st.error(
                    "⚠️ Approaching rate limit! "
                    "Please wait before sending more queries."
                )
            elif bot.total_tokens > 80000:
                st.warning("⚠️ High token usage this session.")
            else:
                st.success("✅ Token usage normal")

            cost = bot.total_tokens * 0.000005
            st.caption(f"Estimated session cost: ~${cost:.4f}")
        else:
            st.caption("No session active yet.")

        st.markdown("---")

        # Reset button
        if st.button("🔄 Reset Conversation",
                     use_container_width=True):
            st.session_state.bot         = None
            st.session_state.messages    = []
            st.session_state.last_result = None
            st.rerun()

        st.caption(
            "Powered by GPT + Yahoo Finance\n\n"
            "Technical analysis is not financial advice."
        )

    # -- Bot creation and model management --------------------
    if st.session_state.get("api_key") and \
       st.session_state.bot is None:
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key
        st.session_state.bot           = StockChatbot()
        st.session_state.current_model = st.session_state.selected_model

    if st.session_state.get("api_key") and \
       st.session_state.bot is not None:
        os.environ["OPENAI_API_KEY"] = st.session_state.api_key
        if st.session_state.selected_model != \
           st.session_state.current_model:
            st.session_state.bot           = StockChatbot()
            st.session_state.current_model = st.session_state.selected_model
            st.session_state.messages      = []
            st.session_state.last_result   = None

    # -- Main area --------------------------------------------
    st.title("📈 Real-Time Stock Analysis Chatbot")
    st.caption(
        "Ask about any stock -- get real-time data and "
        "technical analysis across 3 time horizons."
    )
    st.markdown("---")

    # -- Chat history -----------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            content = msg.get("content", "")
            if msg["role"] == "assistant" and \
               isinstance(content, dict):
                rtype = content.get("type", "text")
                if rtype == "error":
                    st.error(content.get("summary", ""))
                elif rtype == "single_stock":
                    ticker = content["data"]["ticker"]
                    st.markdown(
                        f"Analysis complete for **{ticker}**. "
                        f"See results below."
                    )
                elif rtype == "comparison":
                    tickers = list(
                        content["data"]["comparison_table"].keys()
                    )
                    st.markdown(
                        f"Comparison complete for "
                        f"**{', '.join(tickers)}**. "
                        f"See results below."
                    )
                elif rtype == "price":
                    ticker = content["data"]["ticker"]
                    st.markdown(
                        f"Price data fetched for **{ticker}**. "
                        f"See results below."
                    )
                else:
                    st.markdown(content.get("summary", ""))
            else:
                st.markdown(content)

    # -- Render last analysis result --------------------------
    if st.session_state.last_result:
        result = st.session_state.last_result
        rtype  = result.get("type")

        if rtype == "single_stock":
            render_single_analysis(
                result["data"],
                result["summary"]
            )
        elif rtype == "comparison":
            render_comparison(
                result["data"],
                result["summary"]
            )
        elif rtype == "price":
            render_price_only(
                result["data"],
                result["summary"]
            )

    # -- Process input function -------------------------------
    def process_input(user_input: str):
        print(f"Processing input: {user_input}")
        st.session_state.messages.append({
            "role":    "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                result = st.session_state.bot.chat(user_input)
                print(f"Result type   : {result.get('type')}")
                print(f"Result summary: {result.get('summary', '')[:100]}")

            rtype = result.get("type", "text")

            # Handle error
            if rtype == "error":
                st.error(result.get("summary", "Unknown error"))
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result
                })
                st.rerun()
                return

            # Store result for rendering
            st.session_state.last_result = result

            # Show confirmation in chat bubble
            if rtype == "single_stock":
                ticker = result["data"]["ticker"]
                st.markdown(
                    f"Analysis complete for **{ticker}**. "
                    f"See results below."
                )
            elif rtype == "comparison":
                tickers = list(
                    result["data"]["comparison_table"].keys()
                )
                st.markdown(
                    f"Comparison complete for "
                    f"**{', '.join(tickers)}**. "
                    f"See results below."
                )
            elif rtype == "price":
                ticker = result["data"]["ticker"]
                st.markdown(
                    f"Price data fetched for **{ticker}**. "
                    f"See results below."
                )
            else:
                st.markdown(result.get("summary", ""))

            st.session_state.messages.append({
                "role":    "assistant",
                "content": result
            })

        st.rerun()

    # -- Handle sidebar button clicks -------------------------
    if "pending_input" in st.session_state:
        user_input = st.session_state.pending_input
        del st.session_state.pending_input
        if st.session_state.get("api_key"):
            process_input(user_input)
        else:
            st.warning("Please enter your API key first.")

    # -- Chat input -------------------------------------------
    if not st.session_state.get("api_key"):
        st.info(
            "👈 Please enter your OpenAI API key "
            "in the sidebar to get started."
        )
    else:
        user_input = st.chat_input("Ask about any stock...")
        if user_input:
            process_input(user_input)


if __name__ == "__main__":
    main()
