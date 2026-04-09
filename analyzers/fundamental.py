# ============================================================
# fundamental.py -- Fundamental Data Fetcher
# ============================================================
# Fetches raw fundamental data from Yahoo Finance:
#   Valuation    : P/E, Forward P/E, P/B, P/S
#   Profitability: Gross Margin, Net Margin, ROE
#   Growth       : Revenue Growth, Earnings Growth, EPS
#   Health       : Debt/Equity, Current Ratio, Free Cash Flow
#   Analyst      : Target Price, Recommendation
#
# Scoring logic lives in scoring/fundamental_scorer.py
# ============================================================

import time
import yfinance as yf


# ============================================================
# Data Fetcher
# ============================================================

def fetch_fundamental_data(ticker_symbol: str) -> dict:
    """
    Fetches fundamental data for a single stock from Yahoo Finance.
    All data points are already available in yf.Ticker().info
    so no additional API is needed.
    """
    try:
        time.sleep(1)
        stock = yf.Ticker(ticker_symbol)
        info  = stock.info

        # -- Valuation ----------------------------------------
        pe_ratio       = info.get("trailingPE")
        forward_pe     = info.get("forwardPE")
        price_to_book  = info.get("priceToBook")
        price_to_sales = info.get("priceToSalesTrailing12Months")

        # -- Profitability ------------------------------------
        gross_margin   = info.get("grossMargins")
        net_margin     = info.get("profitMargins")
        roe            = info.get("returnOnEquity")
        roa            = info.get("returnOnAssets")

        # -- Growth -------------------------------------------
        revenue_growth  = info.get("revenueGrowth")
        earnings_growth = info.get("earningsGrowth")
        eps             = info.get("trailingEps")
        forward_eps     = info.get("forwardEps")

        # -- Financial Health ---------------------------------
        debt_to_equity  = info.get("debtToEquity")
        current_ratio   = info.get("currentRatio")
        free_cash_flow  = info.get("freeCashflow")
        total_revenue   = info.get("totalRevenue")

        # -- Analyst Targets ----------------------------------
        target_price    = info.get("targetMeanPrice")
        target_high     = info.get("targetHighPrice")
        target_low      = info.get("targetLowPrice")
        recommendation  = info.get("recommendationKey")
        analyst_count   = info.get("numberOfAnalystOpinions")
        current_price   = (
            info.get("currentPrice") or
            info.get("regularMarketPrice")
        )

        # Calculate upside potential
        upside_pct = None
        if target_price and current_price:
            upside_pct = round(
                ((target_price - current_price) / current_price) * 100, 2
            )

        return {
            "ticker":          ticker_symbol,
            "name":            info.get("longName", ticker_symbol),
            "sector":          info.get("sector"),
            "industry":        info.get("industry"),

            # Valuation
            "pe_ratio":        round(pe_ratio, 2)        if pe_ratio        else None,
            "forward_pe":      round(forward_pe, 2)      if forward_pe      else None,
            "price_to_book":   round(price_to_book, 2)   if price_to_book   else None,
            "price_to_sales":  round(price_to_sales, 2)  if price_to_sales  else None,

            # Profitability
            "gross_margin":    round(gross_margin * 100, 2)  if gross_margin  else None,
            "net_margin":      round(net_margin * 100, 2)    if net_margin    else None,
            "roe":             round(roe * 100, 2)           if roe           else None,
            "roa":             round(roa * 100, 2)           if roa           else None,

            # Growth
            "revenue_growth":  round(revenue_growth * 100, 2)  if revenue_growth  else None,
            "earnings_growth": round(earnings_growth * 100, 2) if earnings_growth else None,
            "eps":             round(eps, 2)         if eps         else None,
            "forward_eps":     round(forward_eps, 2) if forward_eps else None,

            # Financial Health
            "debt_to_equity":  round(debt_to_equity, 2)  if debt_to_equity  else None,
            "current_ratio":   round(current_ratio, 2)   if current_ratio   else None,
            "free_cash_flow":  free_cash_flow,
            "total_revenue":   total_revenue,

            # Analyst Targets
            "target_price":    round(target_price, 2) if target_price else None,
            "target_high":     round(target_high, 2)  if target_high  else None,
            "target_low":      round(target_low, 2)   if target_low   else None,
            "recommendation":  recommendation,
            "analyst_count":   analyst_count,
            "current_price":   round(current_price, 2) if current_price else None,
            "upside_pct":      upside_pct,
        }

    except Exception as e:
        return {"error": str(e)}
