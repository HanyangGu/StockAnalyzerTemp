# ============================================================
# ai.py -- AI Tools, System Prompt & StockChatbot
# ============================================================
# Contains:
#   - GPT tool definitions (fetch_price_data,
#     technical_analysis, compare_stocks)
#   - System prompt
#   - Tool router (TOOL_REGISTRY + handle_tool_call)
#   - StockChatbot class with token tracking
# ============================================================

import json
import time

import streamlit as st
from openai import OpenAI

from core.config import GPT_MODEL, TEMPERATURE
from core.data import fetch_price_data
from scoring.orchestrator import run_analysis
from engine.comparison import compare_stocks


# ============================================================
# GPT Tool Definitions
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_price_data",
            "description": (
                "Fetches real-time price and market data for a single stock. "
                "Use this when the user asks for current price, market cap, "
                "volume, 52 week high or low, or any other real-time market data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_symbol": {
                        "type":        "string",
                        "description": "The stock ticker symbol e.g. AAPL, NVDA, MSFT."
                    }
                },
                "required": ["ticker_symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "technical_analysis",
            "description": (
                "Runs a full technical analysis on a single stock using 9 indicators "
                "across 3 time horizons. Returns short term, mid term, long term and "
                "overall scores (0-100) with verdicts. "
                "Use this when the user asks: "
                "is a stock good to buy, "
                "should I invest in X, "
                "run analysis on X, "
                "show X, "
                "what is the technical outlook for X, "
                "or any question about a single stock performance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {
                        "type":        "string",
                        "description": "Company name or ticker symbol e.g. Apple, NVDA, Microsoft"
                    }
                },
                "required": ["company"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": (
                "Compares 2 to 3 stocks side by side using technical analysis scores "
                "across all time horizons. Ranks stocks and identifies the best pick "
                "overall, per time horizon, and by lowest risk. "
                "Use this when the user mentions multiple companies or asks to compare, "
                "rank, or choose between stocks. Maximum 3 stocks only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "companies": {
                        "type":  "array",
                        "items": {"type": "string"},
                        "description": "List of 2 to 3 company names or ticker symbols.",
                        "minItems": 2,
                        "maxItems": 3,
                    }
                },
                "required": ["companies"]
            }
        }
    },
]


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are an expert stock market analyst assistant.

Your job is to analyze stock data and provide insights.

IMPORTANT: Always use the technical_analysis tool when
the user asks to analyze any stock regardless of whether
backtesting mode is active or not. Never respond with
just text when a stock analysis is requested.

When tools return data, your response must be ONLY a
short 2-4 sentence summary covering:
- Overall technical picture
- Strongest and weakest time horizon
- Any conflicts between horizons
- Optimal entry context if relevant

For comparisons, your summary must cover:
- Who leads and why
- Best pick for different investor profiles
- Key risks to watch

Rules:
- Never reproduce the raw numbers in your summary
- Never include scores or prices in your summary
- Write in plain natural language
- Always end with: Disclaimer: Technical analysis is not financial advice.
- If no tool was called, answer the question conversationally
"""


# ============================================================
# Tool Router
# ============================================================

def _run_analysis(args: dict) -> dict:
    """
    Wrapper for run_analysis that injects
    backtest_date from Streamlit session state.
    """
    backtest_date = st.session_state.get("backtest_date")
    print(f"  Backtest date: {backtest_date}")
    return run_analysis(
        args["company"],
        backtest_date=backtest_date
    )


TOOL_REGISTRY = {
    "fetch_price_data":   lambda args: fetch_price_data(
                              args["ticker_symbol"]
                          ),
    "technical_analysis": lambda args: _run_analysis(args),
    "compare_stocks":     lambda args: compare_stocks(
                              args["companies"]
                          ),
}


def handle_tool_call(name: str, args: dict) -> str:
    """
    Routes a GPT tool call to the correct Python function.
    Returns JSON string of the result.
    """
    try:
        if name not in TOOL_REGISTRY:
            return json.dumps({
                "error": (
                    f"Unknown tool '{name}'. "
                    f"Available tools: {list(TOOL_REGISTRY.keys())}"
                )
            })

        print(f"  Calling tool: {name}...")
        result = TOOL_REGISTRY[name](args)

        if not isinstance(result, dict):
            return json.dumps({
                "error": f"Tool '{name}' returned invalid result."
            })

        return json.dumps(result)

    except KeyError as e:
        return json.dumps({
            "error": f"Missing required argument for tool '{name}': {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "error": f"Tool '{name}' failed with error: {str(e)}"
        })


# ============================================================
# StockChatbot Class
# ============================================================

class StockChatbot:

    def __init__(self):
        self.client             = OpenAI()
        self.history            = []
        self.turn               = 0
        self.model              = st.session_state.get(
                                      "selected_model", GPT_MODEL
                                  )
        self.total_tokens       = 0
        self.prompt_tokens      = 0
        self.completion_tokens  = 0
        self.total_requests     = 0
        print(f"Bot created with model: {self.model}")

    def chat(self, user_message: str) -> dict:
        """
        Sends user message to GPT, executes any tool calls,
        and returns a structured dict for Streamlit to render.

        Return types:
          {"type": "single_stock", "data": {...}, "summary": "..."}
          {"type": "comparison",   "data": {...}, "summary": "..."}
          {"type": "price",        "data": {...}, "summary": "..."}
          {"type": "text",                        "summary": "..."}
          {"type": "error",                       "summary": "..."}
        """
        self.turn += 1
        self.history.append({
            "role":    "user",
            "content": user_message
        })

        tool_name      = None
        tool_result    = None
        max_iterations = 5
        iterations     = 0

        while iterations < max_iterations:
            iterations += 1

            try:
                print(f"Using model: {self.model}")

                # Trim history to last 6 messages (3 ask/answer pairs)
                # to reduce token usage while maintaining context
                trimmed_history = self.history[-6:] \
                    if len(self.history) > 6 \
                    else self.history

                response = self.client.chat.completions.create(
                    model       = self.model,
                    messages    = [
                        {
                            "role":    "system",
                            "content": SYSTEM_PROMPT
                        }
                    ] + trimmed_history,
                    tools       = tools,
                    tool_choice = "auto",
                    temperature = TEMPERATURE,
                )

                # Track token usage
                if response.usage:
                    self.prompt_tokens     += response.usage.prompt_tokens
                    self.completion_tokens += response.usage.completion_tokens
                    self.total_tokens      += response.usage.total_tokens
                    self.total_requests    += 1
                    print(f"  Request tokens    : {response.usage.prompt_tokens}")
                    print(f"  Response tokens   : {response.usage.completion_tokens}")
                    print(f"  Total this request: {response.usage.total_tokens}")
                    print(f"  Session total     : {self.total_tokens}")
                    print(f"  Session requests  : {self.total_requests}")

            except Exception as e:
                error_msg = str(e)
                if "rate limit" in error_msg.lower() or \
                   "too many requests" in error_msg.lower() or \
                   "429" in error_msg:
                    print(f"Rate limited after {self.total_tokens} tokens")
                    print(f"Total requests this session: {self.total_requests}")
                    return {
                        "type": "error",
                        "summary": (
                            f"Rate limit reached after "
                            f"{self.total_tokens:,} tokens and "
                            f"{self.total_requests} requests this session. "
                            f"Please wait 60 seconds then try again."
                        )
                    }
                elif "insufficient_quota" in error_msg.lower() or \
                     "exceeded" in error_msg.lower():
                    return {
                        "type": "error",
                        "summary": (
                            "Monthly quota exceeded. "
                            "Please add credits at "
                            "platform.openai.com/account/billing"
                        )
                    }
                else:
                    print(f"API Error: {error_msg}")
                    return {
                        "type":    "error",
                        "summary": f"API Error: {error_msg}"
                    }

            message = response.choices[0].message

            # -- Tool call path -----------------------------------
            if message.tool_calls:
                self.history.append(message)

                for tool_call in message.tool_calls:
                    args   = json.loads(tool_call.function.arguments)
                    name   = tool_call.function.name
                    result = handle_tool_call(name, args)

                    tool_name   = name
                    tool_result = json.loads(result)

                    self.history.append({
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      result,
                    })

            # -- Text response path -------------------------------
            else:
                summary = message.content
                self.history.append({
                    "role":    "assistant",
                    "content": summary
                })

                # No tool was called -- pure text response
                if tool_result is None:
                    return {
                        "type":    "text",
                        "summary": summary
                    }

                # Tool returned an error
                if "error" in tool_result:
                    return {
                        "type":    "error",
                        "summary": tool_result["error"]
                    }

                # Route to correct response type
                if tool_name == "technical_analysis":
                    return {
                        "type":    "single_stock",
                        "data":    tool_result,
                        "summary": summary
                    }
                elif tool_name == "compare_stocks":
                    return {
                        "type":    "comparison",
                        "data":    tool_result,
                        "summary": summary
                    }
                elif tool_name == "fetch_price_data":
                    return {
                        "type":    "price",
                        "data":    tool_result,
                        "summary": summary
                    }
                else:
                    return {
                        "type":    "text",
                        "summary": summary
                    }

        return {
            "type":    "error",
            "summary": "Analysis took too many steps. Please try again."
        }

    def reset(self):
        """Clears conversation history and resets all counters."""
        self.history           = []
        self.turn              = 0
        self.total_tokens      = 0
        self.prompt_tokens     = 0
        self.completion_tokens = 0
        self.total_requests    = 0
        print("Conversation reset.")

    def get_history_summary(self):
        """Prints conversation stats to terminal."""
        if not self.history:
            print("No conversation history yet.")
            return
        print(f"Total turns   : {self.turn}")
        print(f"Total messages: {len(self.history)}")
        print(f"Total tokens  : {self.total_tokens:,}")
        print(f"Total requests: {self.total_requests}")
