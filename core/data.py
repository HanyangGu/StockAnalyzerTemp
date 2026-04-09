# ============================================================
# data.py -- Data Fetching & Ticker Resolution
# ============================================================
# Handles all Yahoo Finance data retrieval including:
#   - Ticker resolution (name to symbol)
#   - Company validation
#   - Real-time price data
#   - Historical price data (backtest mode)
#   - Historical OHLCV data for indicators
# ============================================================

import time
from datetime import datetime

import pandas as pd
import yfinance as yf

from core.config import (
    DATA_PERIOD,
    DATA_INTERVAL,
    MIN_DATA_POINTS,
    MIN_STOCKS,
    MAX_STOCKS,
)


# ============================================================
# Ticker Resolution
# ============================================================

def resolve_ticker(company: str) -> str:
    """
    Resolves a company name or ticker symbol into a valid
    Yahoo Finance ticker symbol.

    Priority:
      1. Try input directly as a ticker (fastest path)
      2. Search Yahoo Finance by company name
      3. Fall back to cleaned input if both fail
    """
    candidate = company.upper().strip()

    # Step 1: Try as direct ticker
    try:
        info = yf.Ticker(candidate).info
        has_price = (
            info.get("currentPrice") or
            info.get("regularMarketPrice")
        )
        if has_price:
            return candidate
    except Exception:
        pass

    # Step 2: Search by company name
    try:
        results = yf.Search(company, max_results=5).quotes
        if results:
            for r in results:
                if r.get("quoteType") == "EQUITY":
                    return r["symbol"]
            return results[0]["symbol"]
    except Exception:
        pass

    # Step 3: Last resort fallback
    return candidate


def validate_companies(companies: list) -> dict:
    """
    Validates a list of companies for the comparison function.
    Enforces MIN_STOCKS and MAX_STOCKS limits.
    Removes duplicates while preserving order.
    """
    seen   = set()
    unique = []
    for c in companies:
        cleaned = c.strip().upper()
        if cleaned not in seen:
            seen.add(cleaned)
            unique.append(c.strip())

    if len(unique) < MIN_STOCKS:
        return {
            "valid": False,
            "error": f"Please provide at least {MIN_STOCKS} stocks to compare."
        }
    if len(unique) > MAX_STOCKS:
        return {
            "valid": False,
            "error": (
                f"Maximum {MAX_STOCKS} stocks can be compared at once. "
                f"You provided {len(unique)}. Please reduce your selection."
            )
        }

    return {"valid": True, "tickers": unique}


# ============================================================
# Real-Time Price Data
# ============================================================

def fetch_price_data(ticker_symbol: str) -> dict:
    """
    Fetches real-time price and market data for a single stock.
    Includes price, volume, market cap, and other key metrics.
    """
    try:
        time.sleep(1)
        stock = yf.Ticker(ticker_symbol)
        info  = stock.info

        current_price = (
            info.get("currentPrice") or
            info.get("regularMarketPrice")
        )
        if not current_price:
            return {
                "error": (
                    f"Could not retrieve price data for "
                    f"'{ticker_symbol}'. Market may be closed "
                    f"or ticker may be invalid."
                )
            }

        prev_close       = info.get("previousClose", 0) or 0
        price_change     = round(current_price - prev_close, 2)
        price_change_pct = round(
            (price_change / prev_close * 100)
            if prev_close else 0, 2
        )

        intraday = stock.history(period="1d", interval="1m")
        recent_closes = (
            intraday["Close"].tail(5).round(2).tolist()
            if not intraday.empty else []
        )

        return {
            "ticker":            ticker_symbol,
            "name":              info.get("longName", ticker_symbol),
            "exchange":          info.get("exchange"),
            "sector":            info.get("sector"),
            "industry":          info.get("industry"),
            "currency":          info.get("currency", "USD"),
            "current_price":     round(current_price, 2),
            "prev_close":        round(prev_close, 2),
            "price_change":      price_change,
            "price_change_pct":  price_change_pct,
            "open":              info.get("open"),
            "day_high":          info.get("dayHigh"),
            "day_low":           info.get("dayLow"),
            "recent_closes":     recent_closes,
            "volume":            info.get("volume"),
            "avg_volume":        info.get("averageVolume"),
            "market_cap":        info.get("marketCap"),
            "pe_ratio":          info.get("trailingPE"),
            "52w_high":          info.get("fiftyTwoWeekHigh"),
            "52w_low":           info.get("fiftyTwoWeekLow"),
            "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        error_msg = str(e)
        if "too many requests" in error_msg.lower() or \
           "rate limit" in error_msg.lower():
            return {
                "error": (
                    "Yahoo Finance is temporarily rate limited. "
                    "Please wait 30 seconds and try again."
                )
            }
        return {"error": error_msg}


# ============================================================
# Historical Price Data (Backtest Mode)
# ============================================================

def fetch_price_data_historical(ticker_symbol: str,
                                 date: str) -> dict:
    """
    Fetches price data for a specific historical date.
    Used for backtesting -- returns price as of that date
    instead of current live price.
    """
    try:
        time.sleep(1)
        stock      = yf.Ticker(ticker_symbol)
        end_date   = pd.Timestamp(date)
        start_date = end_date - pd.DateOffset(days=5)

        df = stock.history(
            start    = start_date.strftime("%Y-%m-%d"),
            end      = (end_date + pd.DateOffset(days=1)
                       ).strftime("%Y-%m-%d"),
            interval = "1d"
        )

        if df.empty:
            return {
                "error": (
                    f"No price data found for "
                    f"{ticker_symbol} on {date}"
                )
            }

        # Get the closest trading day to selected date
        row              = df.iloc[-1]
        current_price    = round(float(row["Close"]), 2)
        prev_price       = round(float(df.iloc[-2]["Close"]), 2) \
                           if len(df) > 1 else current_price
        price_change     = round(current_price - prev_price, 2)
        price_change_pct = round(
            (price_change / prev_price * 100)
            if prev_price else 0, 2
        )

        info = stock.info

        return {
            "ticker":           ticker_symbol,
            "name":             info.get("longName", ticker_symbol),
            "exchange":         info.get("exchange"),
            "sector":           info.get("sector"),
            "industry":         info.get("industry"),
            "currency":         info.get("currency", "USD"),
            "current_price":    current_price,
            "prev_close":       prev_price,
            "price_change":     price_change,
            "price_change_pct": price_change_pct,
            "open":             round(float(row["Open"]), 2),
            "day_high":         round(float(row["High"]), 2),
            "day_low":          round(float(row["Low"]), 2),
            "recent_closes":    [],
            "volume":           int(row["Volume"]),
            "avg_volume":       info.get("averageVolume"),
            "market_cap":       info.get("marketCap"),
            "pe_ratio":         info.get("trailingPE"),
            "52w_high":         info.get("fiftyTwoWeekHigh"),
            "52w_low":          info.get("fiftyTwoWeekLow"),
            "timestamp":        str(date),
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Historical OHLCV Data (For Indicators)
# ============================================================

def fetch_historical_data(ticker_symbol: str,
                          end_date: str = None) -> dict:
    """
    Fetches 1 year of daily OHLCV historical data
    for technical indicator calculation.

    If end_date is provided, fetches data up to that date
    for backtesting purposes.

    Returns:
      {"success": True,  "df": DataFrame, "closes": Series}
      {"success": False, "error": "..."}
    """
    try:
        time.sleep(1)
        stock = yf.Ticker(ticker_symbol)

        if end_date:
            # Backtest mode -- fetch data up to selected date
            end   = pd.Timestamp(end_date)
            start = end - pd.DateOffset(years=1)
            df    = stock.history(
                start    = start.strftime("%Y-%m-%d"),
                end      = end.strftime("%Y-%m-%d"),
                interval = DATA_INTERVAL
            )
        else:
            # Normal mode -- fetch latest 1 year of data
            df = stock.history(
                period   = DATA_PERIOD,
                interval = DATA_INTERVAL
            )

        if df.empty:
            return {
                "success": False,
                "error":   f"No historical data found for {ticker_symbol}."
            }

        if len(df) < MIN_DATA_POINTS:
            return {
                "success": False,
                "error": (
                    f"Insufficient data for {ticker_symbol}. "
                    f"Need at least {MIN_DATA_POINTS} trading days."
                )
            }

        return {
            "success": True,
            "df":      df,
            "closes":  df["Close"],
            "days":    len(df),
        }

    except Exception as e:
        error_msg = str(e)
        if "too many requests" in error_msg.lower() or \
           "rate limit" in error_msg.lower():
            return {
                "success": False,
                "error": (
                    "Yahoo Finance is temporarily rate limited. "
                    "Please wait 30 seconds and try again."
                )
            }
        return {"success": False, "error": error_msg}
