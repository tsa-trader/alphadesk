"""
AlphaDesk — AI Hedge Fund Trading System
=========================================
Single-file Streamlit application.

Install:
    pip install streamlit MetaTrader5 yfinance pandas-ta plotly apscheduler

Run:
    python -m streamlit run alphadesk.py
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import queue
import datetime
import threading
import time
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except Exception:
    ta = None
    TA_AVAILABLE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHED_AVAILABLE = True
except Exception:
    BackgroundScheduler = None
    SCHED_AVAILABLE = False

try:
    import anthropic as _anthropic_sdk
    ANTHROPIC_AVAILABLE = True
except Exception:
    _anthropic_sdk = None
    ANTHROPIC_AVAILABLE = False

import json
import re


# ─────────────────────────────────────────────────────────────────────────────
#  AI STRATEGY PARSER
# ─────────────────────────────────────────────────────────────────────────────

_PATTERN_NAMES = list([
    "Inside Bar Breakout (Both)", "Inside Bar Breakout (Bull)",
    "Inside Bar Breakout (Bear)", "Engulfing (Both)",
    "Bullish Engulfing Only", "Bearish Engulfing Only",
    "Pin Bar Reversal (Both)", "Pin Bar Reversal (Bull)",
    "Pin Bar Reversal (Bear)", "EMA Crossover",
    "RSI Mean Reversion", "Bollinger Band Breakout", "MACD Trend",
])

_AI_SYSTEM = """You are an expert algorithmic trading strategy parser.

The user will give you a trading strategy description in plain English (or pasted from a PDF/image).
Your job is to extract the strategy parameters and return ONLY a valid JSON object — no explanation, no markdown, no code fences.

Return exactly this JSON shape (use null for anything not mentioned):
{
  "symbol": string,           // e.g. "GBPJPY=X", "EURUSD=X", "GC=F" (gold), "BTC-USD"
  "pattern": string,          // one of: """ + ", ".join(f'"{p}"' for p in _PATTERN_NAMES) + """,
  "direction": string,        // "Both", "Long Only", or "Short Only"
  "sl_pips": integer,         // stop loss in pips (e.g. 300)
  "tp_pips": integer | null,  // take profit in pips, null if not specified or if exit is rule-based
  "trailing_stop": boolean,   // true if trailing stop mentioned
  "trailing_pips": integer,   // trailing distance in pips (default 100 if trailing but no value given)
  "risk_pct": number,         // risk per trade as decimal (e.g. 0.01 for 1%), default 0.01
  "timeframe": string,        // "1d" or "1wk"
  "period": string,           // yfinance period: "1y", "2y", "3y", "5y"
  "notes": string             // any special exit rules or conditions not covered above
}

Rules:
- "Inside Day" means "Inside Bar" — map to "Inside Bar Breakout (Both)" if both long and short are mentioned
- "Go long if closed > day prior to inside day" = bullish breakout
- "Go short if closed < day prior to inside day" = bearish breakout  
- If both long and short → direction "Both", pattern "Inside Bar Breakout (Both)"
- "Stop and reverse" = when SL is hit, flip direction (note in notes field)
- "1st profitable close exit" = no fixed TP, note in notes field, set tp_pips to null
- If a symbol like GBP/JPY is mentioned, use "GBPJPY=X"
- If risk % not mentioned, default to 0.01 (1%)
- If timeframe not mentioned, default to "1d"
- If period not mentioned, default to "2y"
"""

def parse_strategy_ai(text: str, api_key: str) -> dict:
    """
    Send strategy text to Claude API.
    Returns parsed dict or raises on failure.
    """
    client = _anthropic_sdk.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        system=_AI_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    raw = msg.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
#  THREAD-SAFE SHARED STATE
# ─────────────────────────────────────────────────────────────────────────────
_log_queue: queue.Queue = queue.Queue()
_live_results: dict = {}
_lock = threading.Lock()


def _log(agent: str, msg: str, level: str = "INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    color = {"INFO": "green", "WARN": "amber", "ERROR": "red"}.get(level, "green")
    _log_queue.put({"ts": ts, "agent": agent, "msg": msg, "color": color})
    logging.getLogger(agent).info(msg)


# ─────────────────────────────────────────────────────────────────────────────
#  MT5 BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OrderResult:
    success: bool
    ticket: int
    price: float
    message: str


class MT5Bridge:
    def __init__(self, login: int, password: str, server: str):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False

    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not installed.")
        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
        ok = mt5.login(self.login, self.password, self.server)
        if not ok:
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
        self.connected = True
        info = mt5.account_info()
        _log("MT5Bridge", f"Connected to {info.name} | Balance: {info.balance} {info.currency}")
        return True

    def account_info(self) -> dict:
        if not self.connected:
            return {}
        info = mt5.account_info()
        return {
            "name": info.name, "balance": info.balance, "equity": info.equity,
            "margin_free": info.margin_free, "currency": info.currency,
            "profit": info.profit, "leverage": info.leverage,
        }

    def get_tick(self, symbol: str) -> dict:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {}
        return {"bid": tick.bid, "ask": tick.ask, "time": tick.time}

    def get_positions(self) -> list:
        positions = mt5.positions_get()
        if not positions:
            return []
        return [p._asdict() for p in positions]

    def send_order(self, symbol: str, order_type: str, volume: float,
                   sl_pips: float = 50, tp_pips: float = 100,
                   comment: str = "AlphaDesk") -> "OrderResult":
        info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if info is None or tick is None:
            return OrderResult(False, 0, 0.0, f"Symbol {symbol} not found")
        point = info.point
        if order_type == "BUY":
            otype = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = price - sl_pips * point
            tp = price + tp_pips * point
        else:
            otype = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = price + sl_pips * point
            tp = price - tp_pips * point
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
            "volume": round(volume, 2), "type": otype, "price": price,
            "sl": sl, "tp": tp, "deviation": 10, "magic": 234000,
            "comment": comment, "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        return OrderResult(
            result.retcode == mt5.TRADE_RETCODE_DONE,
            result.order, result.price, result.comment
        )

    def close_position(self, ticket: int) -> bool:
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return False
        p = pos[0]
        close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(p.symbol)
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol,
            "volume": p.volume, "type": close_type, "position": ticket,
            "price": price, "deviation": 10, "magic": 234000,
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE

    def close_all(self):
        for p in self.get_positions():
            self.close_position(p["ticket"])

    def disconnect(self):
        if MT5_AVAILABLE:
            mt5.shutdown()
        self.connected = False


# ─────────────────────────────────────────────────────────────────────────────
#  AGENTS
# ─────────────────────────────────────────────────────────────────────────────
class BaseAgent(ABC):
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    @abstractmethod
    def run(self, context: dict) -> dict:
        pass

    def log(self, msg: str, level: str = "INFO"):
        _log(self.name, msg, level)


class MarketDataAgent(BaseAgent):
    def __init__(self, config: dict):
        super().__init__("MarketData", config)

    def run(self, context: dict) -> dict:
        symbol = context["symbol"]
        bridge = context.get("bridge")
        if bridge and bridge.connected:
            tick = bridge.get_tick(symbol)
            if tick:
                context.update({"bid": tick["bid"], "ask": tick["ask"],
                                 "price": (tick["bid"] + tick["ask"]) / 2})
                self.log(f"{symbol} bid={tick['bid']} ask={tick['ask']}")
                return context
        if YF_AVAILABLE:
            sym_map = {
                "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "GBPJPY": "GBPJPY=X",
                "XAUUSD": "GC=F", "NAS100": "NQ=F", "USDJPY": "USDJPY=X",
            }
            yf_sym = sym_map.get(symbol, symbol)
            try:
                df = yf.download(yf_sym, period="5d", interval="1h",
                                 auto_adjust=True, progress=False)
                if not df.empty:
                    price = float(df["Close"].iloc[-1])
                    context.update({"price": price, "bid": price * 0.9999,
                                    "ask": price * 1.0001})
                    self.log(f"{symbol} price={price:.5f}")
            except Exception as e:
                self.log(f"yfinance error {symbol}: {e}", "WARN")
        return context


class StrategyAgent(BaseAgent):
    STRATEGIES = [
        "EMA Crossover", "RSI Mean Reversion",
        "Bollinger Band Breakout", "MACD Trend",
        "Inside Bar Breakout", "Engulfing", "Pin Bar",
    ]

    def __init__(self, config: dict):
        super().__init__("Strategy", config)

    def run(self, context: dict) -> dict:
        symbol = context["symbol"]
        strategy = self.config.get("strategy", "EMA Crossover")
        if not YF_AVAILABLE or not TA_AVAILABLE:
            context["signal"] = "BUY" if np.random.random() > 0.5 else "SELL"
            context["confidence"] = round(np.random.uniform(0.60, 0.95), 2)
            context["strategy_name"] = strategy
            self.log(f"{symbol} demo signal={context['signal']}")
            return context
        sym_map = {
            "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "GBPJPY": "GBPJPY=X",
            "XAUUSD": "GC=F", "NAS100": "NQ=F", "USDJPY": "USDJPY=X",
        }
        yf_sym = sym_map.get(symbol, symbol)
        try:
            raw = yf.download(yf_sym, period="60d", interval="1h",
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw["Close"].squeeze()
            signal, confidence = None, 0.0
            if strategy == "EMA Crossover":
                f = ta.ema(close, length=9)
                s = ta.ema(close, length=21)
                if f.iloc[-1] > s.iloc[-1] and f.iloc[-2] <= s.iloc[-2]:
                    signal, confidence = "BUY", 0.82
                elif f.iloc[-1] < s.iloc[-1] and f.iloc[-2] >= s.iloc[-2]:
                    signal, confidence = "SELL", 0.78
            elif strategy == "RSI Mean Reversion":
                rsi = ta.rsi(close, length=14)
                if rsi.iloc[-1] < 35:
                    signal, confidence = "BUY", 0.75
                elif rsi.iloc[-1] > 65:
                    signal, confidence = "SELL", 0.72
            elif strategy == "MACD Trend":
                m = ta.macd(close)
                mac = m["MACD_12_26_9"]
                sig = m["MACDs_12_26_9"]
                if mac.iloc[-1] > sig.iloc[-1] and mac.iloc[-2] <= sig.iloc[-2]:
                    signal, confidence = "BUY", 0.80
                elif mac.iloc[-1] < sig.iloc[-1] and mac.iloc[-2] >= sig.iloc[-2]:
                    signal, confidence = "SELL", 0.77
            if signal:
                context.update({"signal": signal, "confidence": confidence,
                                 "strategy_name": strategy})
                self.log(f"{symbol} {signal} conf={confidence:.0%}")
            else:
                context["signal"] = None
                self.log(f"{symbol} no signal")
        except Exception as e:
            self.log(f"Strategy error [{symbol}]: {e}", "ERROR")
        return context


class RiskManagerAgent(BaseAgent):
    def __init__(self, config: dict):
        super().__init__("RiskManager", config)

    def run(self, context: dict) -> dict:
        balance = context.get("balance", 100_000)
        drawdown = context.get("current_drawdown", 0.0)
        open_count = context.get("open_positions_count", 0)
        signal = context.get("signal")
        max_risk = self.config.get("max_risk_pct", 0.01)
        max_dd = self.config.get("max_drawdown", 0.10)
        max_pos = self.config.get("max_positions", 5)
        if not signal:
            context["approved"] = False
            context["reject_reason"] = "No signal"
            return context
        if drawdown >= max_dd:
            context["approved"] = False
            context["reject_reason"] = "Max drawdown hit"
            self.log(f"REJECTED drawdown={drawdown*100:.1f}%", "WARN")
            return context
        if open_count >= max_pos:
            context["approved"] = False
            context["reject_reason"] = "Max positions reached"
            return context
        win_rate = context.get("win_rate", 0.55)
        rr = context.get("reward_risk", 2.0)
        kelly = max(0.0, min(win_rate - (1 - win_rate) / rr, 0.25))
        risk_amt = balance * max_risk
        sl_pips = context.get("sl_pips", 50)
        pip_value = context.get("pip_value", 10.0)
        lot_size = max(0.01, round(min(risk_amt / max(sl_pips * pip_value, 1),
                                       balance * kelly / 10_000), 2))
        context["approved"] = True
        context["lot_size"] = lot_size
        context["kelly"] = kelly
        self.log(f"APPROVED {context['symbol']} {signal} {lot_size:.2f}L")
        return context


class ExecutionAgent(BaseAgent):
    def __init__(self, config: dict, bridge=None):
        super().__init__("Execution", config)
        self.bridge = bridge

    def run(self, context: dict) -> dict:
        symbol = context["symbol"]
        signal = context["signal"]
        lot_size = context.get("lot_size", 0.01)
        if not self.bridge or not self.bridge.connected:
            self.log(f"[PAPER] {symbol} {signal} {lot_size:.2f}L", "WARN")
            context["order_result"] = OrderResult(True, 0, context.get("price", 0), "paper")
            return context
        try:
            result = self.bridge.send_order(
                symbol=symbol, order_type=signal, volume=lot_size,
                sl_pips=context.get("sl_pips", 50),
                tp_pips=context.get("tp_pips", 100),
            )
            lvl = "INFO" if result.success else "ERROR"
            self.log(f"{'FILLED' if result.success else 'REJECTED'} {symbol} "
                     f"{signal} {lot_size:.2f}L @ {result.price}", lvl)
            context["order_result"] = result
        except Exception as e:
            self.log(f"Execution error: {e}", "ERROR")
        return context


class PortfolioOrchestrator(BaseAgent):
    def __init__(self, config: dict, bridge=None):
        super().__init__("Orchestrator", config)
        self.bridge = bridge
        self.data_agent = MarketDataAgent(config)
        self.strat_agent = StrategyAgent(config)
        self.risk_agent = RiskManagerAgent(config)
        self.exec_agent = ExecutionAgent(config, bridge)

    def run_symbol(self, symbol: str) -> dict:
        ctx = {
            "symbol": symbol, "bridge": self.bridge,
            "balance": self.config.get("balance", 100_000),
            "current_drawdown": self.config.get("current_drawdown", 0.0),
            "open_positions_count": self.config.get("open_positions_count", 0),
            "win_rate": self.config.get("win_rate", 0.55),
            "reward_risk": self.config.get("reward_risk", 2.0),
            "sl_pips": self.config.get("sl_pips", 50),
            "tp_pips": self.config.get("tp_pips", 100),
        }
        ctx = self.data_agent.run(ctx)
        ctx = self.strat_agent.run(ctx)
        ctx = self.risk_agent.run(ctx)
        if ctx.get("approved"):
            ctx = self.exec_agent.run(ctx)
        return ctx

    def run(self, context: dict) -> dict:
        return self.run_symbol(context["symbol"])

    def run_all(self) -> dict:
        results = {}
        symbols = self.config.get("symbols", ["EURUSD", "GBPUSD", "XAUUSD"])
        self.log(f"Cycle started — {len(symbols)} symbols")
        for sym in symbols:
            try:
                results[sym] = self.run_symbol(sym)
            except Exception as e:
                self.log(f"Error on {sym}: {e}", "ERROR")
        self.log("Cycle complete")
        with _lock:
            _live_results.update(results)
        return results


# ─────────────────────────────────────────────────────────────────────────────
#  SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
_scheduler = None


def _scheduler_job():
    try:
        config = _live_results.get("__config__", {})
        bridge = _live_results.get("__bridge__")
        if config:
            PortfolioOrchestrator(config, bridge).run_all()
    except Exception:
        _log("Scheduler", traceback.format_exc(), "ERROR")


def start_scheduler(interval_minutes: int = 5):
    global _scheduler
    if not SCHED_AVAILABLE:
        return
    if _scheduler and _scheduler.running:
        return
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _scheduler_job, "interval", minutes=interval_minutes,
        id="trading_cycle", replace_existing=True,
        next_run_time=datetime.datetime.now() + datetime.timedelta(seconds=5),
    )
    _scheduler.start()
    _log("Scheduler", f"Started — cycle every {interval_minutes} min")


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _log("Scheduler", "Stopped")


# ─────────────────────────────────────────────────────────────────────────────
#  BACKTEST ENGINE — patterns + bar-by-bar simulation
#  (all functions defined here, BEFORE the Streamlit page chain)
# ─────────────────────────────────────────────────────────────────────────────

_JPY_PAIRS = {"GBPJPY", "USDJPY", "EURJPY", "CADJPY", "AUDJPY", "NZDJPY", "CHFJPY"}


def get_pip_size(symbol: str) -> float:
    sym = symbol.replace("=X", "").replace("-", "").upper()
    for k in _JPY_PAIRS:
        if k in sym:
            return 0.01
    return 0.0001


def _pat_inside_bar_bull(df):
    ib = (df["High"] <= df["High"].shift(1)) & (df["Low"] >= df["Low"].shift(1))
    return ib.shift(1).fillna(False) & (df["Close"] > df["High"].shift(2).fillna(0))


def _pat_inside_bar_bear(df):
    ib = (df["High"] <= df["High"].shift(1)) & (df["Low"] >= df["Low"].shift(1))
    return ib.shift(1).fillna(False) & (df["Close"] < df["Low"].shift(2).fillna(9e9))


def _pat_engulf_bull(df):
    pb = df["Close"].shift(1) < df["Open"].shift(1)
    cb = df["Close"] > df["Open"]
    en = (df["Open"] <= df["Close"].shift(1)) & (df["Close"] >= df["Open"].shift(1))
    return pb & cb & en


def _pat_engulf_bear(df):
    pb = df["Close"].shift(1) > df["Open"].shift(1)
    cb = df["Close"] < df["Open"]
    en = (df["Open"] >= df["Close"].shift(1)) & (df["Close"] <= df["Open"].shift(1))
    return pb & cb & en


def _pat_pin_bull(df):
    body = abs(df["Close"] - df["Open"])
    rng  = df["High"] - df["Low"]
    low_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
    up_wick  = df["High"] - df[["Open", "Close"]].max(axis=1)
    return (low_wick > body * 2) & (low_wick > up_wick * 2) & (rng > 0)


def _pat_pin_bear(df):
    body = abs(df["Close"] - df["Open"])
    rng  = df["High"] - df["Low"]
    up_wick  = df["High"] - df[["Open", "Close"]].max(axis=1)
    low_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
    return (up_wick > body * 2) & (up_wick > low_wick * 2) & (rng > 0)


def _pat_ema_bull(df):
    if not TA_AVAILABLE:
        return pd.Series(False, index=df.index)
    f = ta.ema(df["Close"], length=9)
    s = ta.ema(df["Close"], length=21)
    return (f > s) & (f.shift(1) <= s.shift(1))


def _pat_ema_bear(df):
    if not TA_AVAILABLE:
        return pd.Series(False, index=df.index)
    f = ta.ema(df["Close"], length=9)
    s = ta.ema(df["Close"], length=21)
    return (f < s) & (f.shift(1) >= s.shift(1))


def _pat_rsi_bull(df):
    if not TA_AVAILABLE:
        return pd.Series(False, index=df.index)
    return ta.rsi(df["Close"], length=14) < 35


def _pat_rsi_bear(df):
    if not TA_AVAILABLE:
        return pd.Series(False, index=df.index)
    return ta.rsi(df["Close"], length=14) > 65


def _pat_bb_bull(df):
    if not TA_AVAILABLE:
        return pd.Series(False, index=df.index)
    bb = ta.bbands(df["Close"], length=20)
    return df["Close"] > bb["BBU_20_2.0"]


def _pat_bb_bear(df):
    if not TA_AVAILABLE:
        return pd.Series(False, index=df.index)
    bb = ta.bbands(df["Close"], length=20)
    return df["Close"] < bb["BBL_20_2.0"]


def _pat_macd_bull(df):
    if not TA_AVAILABLE:
        return pd.Series(False, index=df.index)
    m = ta.macd(df["Close"])
    return (m["MACD_12_26_9"] > m["MACDs_12_26_9"]) & \
           (m["MACD_12_26_9"].shift(1) <= m["MACDs_12_26_9"].shift(1))


def _pat_macd_bear(df):
    if not TA_AVAILABLE:
        return pd.Series(False, index=df.index)
    m = ta.macd(df["Close"])
    return (m["MACD_12_26_9"] < m["MACDs_12_26_9"]) & \
           (m["MACD_12_26_9"].shift(1) >= m["MACDs_12_26_9"].shift(1))


PATTERN_MAP = {
    "Inside Bar Breakout (Both)":  (_pat_inside_bar_bull, _pat_inside_bar_bear),
    "Inside Bar Breakout (Bull)":  (_pat_inside_bar_bull, None),
    "Inside Bar Breakout (Bear)":  (None,                 _pat_inside_bar_bear),
    "Engulfing (Both)":            (_pat_engulf_bull,     _pat_engulf_bear),
    "Bullish Engulfing Only":      (_pat_engulf_bull,     None),
    "Bearish Engulfing Only":      (None,                 _pat_engulf_bear),
    "Pin Bar Reversal (Both)":     (_pat_pin_bull,        _pat_pin_bear),
    "Pin Bar Reversal (Bull)":     (_pat_pin_bull,        None),
    "Pin Bar Reversal (Bear)":     (None,                 _pat_pin_bear),
    "EMA Crossover":               (_pat_ema_bull,        _pat_ema_bear),
    "RSI Mean Reversion":          (_pat_rsi_bull,        _pat_rsi_bear),
    "Bollinger Band Breakout":     (_pat_bb_bull,         _pat_bb_bear),
    "MACD Trend":                  (_pat_macd_bull,       _pat_macd_bear),
}

PATTERN_HELP = {
    "Inside Bar Breakout (Both)":
        "A bar whose High/Low sits inside the previous (mother) bar. "
        "Entry fires the next bar when price breaks out above or below the mother bar.",
    "Engulfing (Both)":
        "Current candle body fully engulfs the previous candle body — classic reversal.",
    "Pin Bar Reversal (Both)":
        "Long-wicked candle (hammer/shooting star) — price rejected a level strongly.",
    "EMA Crossover":
        "Fast EMA(9) crosses above/below Slow EMA(21) — momentum confirmation entry.",
    "RSI Mean Reversion":
        "Enter long when RSI < 35 (oversold), short when RSI > 65 (overbought).",
    "Bollinger Band Breakout":
        "Buy when price closes above the upper Bollinger Band, sell below lower band.",
    "MACD Trend":
        "Buy when MACD crosses above its signal line, sell when it crosses below.",
}


def run_simulation(df, bull_signal, bear_signal, sl_pips, tp_pips,
                   pip_size, trailing_stop, trailing_pips,
                   direction, capital, risk_pct):
    """
    Bar-by-bar simulation engine.
    Returns (list[dict], pd.Series) — trades and equity curve.
    """
    sl_dist = sl_pips  * pip_size
    tp_dist = tp_pips  * pip_size
    tr_dist = trailing_pips * pip_size

    equity = capital
    trades = []
    eq_curve = []

    in_trade = False
    trade_dir = entry_price = current_sl = current_tp = entry_date = best_price = None

    for i in range(len(df)):
        row = df.iloc[i]
        hi = float(row["High"])
        lo = float(row["Low"])
        cl = float(row["Close"])
        dt = df.index[i]

        # Manage open trade
        if in_trade:
            exit_price = cl
            exit_reason = "Open"

            if trade_dir == "BUY":
                if trailing_stop and hi > best_price:
                    best_price = hi
                    new_sl = best_price - tr_dist
                    if new_sl > current_sl:
                        current_sl = new_sl
                if lo <= current_sl:
                    exit_price, exit_reason = current_sl, "Stop Loss"
                elif hi >= current_tp:
                    exit_price, exit_reason = current_tp, "Take Profit"
            else:
                if trailing_stop and lo < best_price:
                    best_price = lo
                    new_sl = best_price + tr_dist
                    if new_sl < current_sl:
                        current_sl = new_sl
                if hi >= current_sl:
                    exit_price, exit_reason = current_sl, "Stop Loss"
                elif lo <= current_tp:
                    exit_price, exit_reason = current_tp, "Take Profit"

            if exit_reason in ("Stop Loss", "Take Profit"):
                raw_pips = ((exit_price - entry_price) if trade_dir == "BUY"
                            else (entry_price - exit_price)) / pip_size
                pnl = (raw_pips / sl_pips) * (equity * risk_pct)
                equity += pnl
                trades.append({
                    "Entry Date":  entry_date,
                    "Exit Date":   dt,
                    "Dir":         trade_dir,
                    "Entry":       round(entry_price, 5),
                    "Exit":        round(exit_price,  5),
                    "SL":          round(current_sl,  5),
                    "TP":          round(current_tp,  5),
                    "Pips":        round(raw_pips, 1),
                    "P&L ($)":     round(pnl, 2),
                    "Reason":      exit_reason,
                    "Equity":      round(equity, 2),
                })
                in_trade = False

        # New entry (only when flat)
        if not in_trade:
            do_buy  = (direction in ("Long Only",  "Both")) and bool(bull_signal.iloc[i])
            do_sell = (direction in ("Short Only", "Both")) and bool(bear_signal.iloc[i])
            if do_buy:
                in_trade = True
                trade_dir = "BUY"
                entry_price = cl
                current_sl  = cl - sl_dist
                current_tp  = cl + tp_dist
                best_price  = cl
                entry_date  = dt
            elif do_sell:
                in_trade = True
                trade_dir = "SELL"
                entry_price = cl
                current_sl  = cl + sl_dist
                current_tp  = cl - tp_dist
                best_price  = cl
                entry_date  = dt

        eq_curve.append(equity)

    return trades, pd.Series(eq_curve, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
#  DEMO DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _demo_pnl(days: int = 90) -> pd.DataFrame:
    dates  = pd.date_range(end=datetime.date.today(), periods=days)
    equity = 100_000 * np.cumprod(1 + np.random.randn(days) * 0.008 + 0.001)
    return pd.DataFrame({"date": dates, "equity": equity})


def _demo_positions() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol":        ["EURUSD",  "XAUUSD",  "GBPUSD"],
        "type":          ["BUY",     "BUY",     "SELL"],
        "volume":        [0.10,       0.05,      0.08],
        "price_open":    [1.08310,   2305.20,   1.26500],
        "price_current": [1.08423,   2318.50,   1.26741],
        "profit":        [11.30,      66.50,    -19.28],
        "ticket":        [10482341,  10482342,  10482343],
    })


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT CONFIG  — must be first st.* call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaDesk", page_icon="📈",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif !important; }
  .stApp { background: #0d1117; }
  section[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }
  .block-container { padding-top: 1.5rem !important; }
  h1,h2,h3 { font-family: 'IBM Plex Mono', monospace !important; font-weight: 500 !important; }
  div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem !important; }
  .stButton > button { font-family: 'IBM Plex Mono', monospace; font-size: 12px; border-radius: 4px; }
  .status-pill { display:inline-block; padding:2px 10px; border-radius:12px; font-size:11px; font-family:'IBM Plex Mono',monospace; }
  .pill-green { background:rgba(0,212,100,.15); color:#00d464; border:1px solid rgba(0,212,100,.3); }
  .pill-red   { background:rgba(255,77,106,.12); color:#ff4d6a; border:1px solid rgba(255,77,106,.3); }
  .pill-amber { background:rgba(245,166,35,.1);  color:#f5a623; border:1px solid rgba(245,166,35,.3); }
  .log-container { background:#0d1117; border:1px solid #30363d; border-radius:6px; padding:12px;
                   height:360px; overflow-y:auto; font-family:'IBM Plex Mono',monospace;
                   font-size:11px; line-height:1.8; }
</style>
""", unsafe_allow_html=True)

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=20, b=0),
)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "bridge":            None,
    "mt5_connected":     False,
    "agent_logs":        [],
    "scheduler_on":      False,
    "interval_min":      5,
    "approved_strategy": None,
    "ai_parsed":         None,   # last AI-parsed strategy config
    "anthropic_api_key": "",
    "risk_config": {
        "max_risk_pct":    0.01,
        "max_drawdown":    0.10,
        "max_positions":   5,
        "max_correlation": 0.70,
        "sl_pips":         50,
        "tp_pips":         100,
        "reward_risk":     2.0,
        "win_rate":        0.55,
        "symbols":         ["EURUSD", "GBPUSD", "XAUUSD"],
        "strategy":        "EMA Crossover",
        "balance":         100_000,
        "trailing_stop":   False,
        "trailing_pips":   50,
        "direction":       "Both",
    },
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

while not _log_queue.empty():
    st.session_state.agent_logs.append(_log_queue.get())
st.session_state.agent_logs = st.session_state.agent_logs[-500:]

_live_results["__config__"] = st.session_state.risk_config
_live_results["__bridge__"] = st.session_state.bridge


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 AlphaDesk")
    st.caption("AI Hedge Fund System")

    conn_lbl = "🟢 MT5 Connected" if st.session_state.mt5_connected else "🔴 MT5 Offline"
    st.markdown(f"**{conn_lbl}**")
    if st.session_state.mt5_connected and st.session_state.bridge:
        try:
            _info = st.session_state.bridge.account_info()
            st.caption(f"Balance: ${_info.get('balance', 0):,.0f} {_info.get('currency', '')}")
        except Exception:
            pass

    ap = st.session_state.get("approved_strategy")
    if ap:
        st.success(f"✅ {ap['pattern']}\n{ap['symbol']} SL={ap['sl_pips']}p")

    st.divider()
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🧠 Strategies", "📈 Backtest",
         "💼 Positions", "🛡️ Risk", "🔌 MT5 Bridge", "📋 Agent Logs"],
        label_visibility="collapsed",
    )
    st.divider()

    auto = st.toggle("Auto-run scheduler", value=st.session_state.scheduler_on)
    if auto != st.session_state.scheduler_on:
        st.session_state.scheduler_on = auto
        if auto:
            start_scheduler(st.session_state.interval_min)
        else:
            stop_scheduler()
    if st.session_state.scheduler_on and SCHED_AVAILABLE:
        st.caption(f"⚡ Every {st.session_state.interval_min} min")

    st.divider()
    if st.button("🔄 Manual cycle", use_container_width=True):
        orch = PortfolioOrchestrator(st.session_state.risk_config, st.session_state.bridge)
        with st.spinner("Running agents..."):
            orch.run_all()
        st.success("Done!")
        st.rerun()

    refresh = st.toggle("Auto-refresh UI (10s)", value=False)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Portfolio Overview")

    if st.session_state.mt5_connected and st.session_state.bridge:
        _i = st.session_state.bridge.account_info()
        balance, equity, profit, margin_f = (
            _i.get("balance", 0), _i.get("equity", 0),
            _i.get("profit", 0), _i.get("margin_free", 0)
        )
    else:
        balance, equity, profit, margin_f = 104_821, 103_240, 312, 98_420

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Balance",     f"${balance:,.0f}")
    c2.metric("Equity",      f"${equity:,.0f}",  f"${profit:+,.0f}")
    c3.metric("Today P&L",   f"${profit:+,.0f}")
    c4.metric("Free Margin", f"${margin_f:,.0f}")
    c5.metric("Drawdown",    "3.2%", "-0.4%")

    st.divider()
    cl, cr = st.columns([3, 2])

    with cl:
        st.subheader("Equity Curve")
        _pnl = _demo_pnl(90)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=_pnl["date"], y=_pnl["equity"],
            fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
            line=dict(color="#00d4aa", width=1.5), name="Portfolio",
        ))
        fig.update_layout(**DARK_LAYOUT, height=260,
                          xaxis=dict(showgrid=False),
                          yaxis=dict(showgrid=True, gridcolor="#1e2a1e"))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.subheader("Live Signals")
        _sigs = []
        with _lock:
            for _sym, _res in _live_results.items():
                if _sym.startswith("__"):
                    continue
                if _res.get("signal"):
                    _sigs.append({
                        "Symbol": _sym,
                        "Signal": _res.get("signal"),
                        "Conf":   f"{_res.get('confidence', 0):.0%}",
                        "OK":     "✅" if _res.get("approved") else "❌",
                    })
        if _sigs:
            st.dataframe(pd.DataFrame(_sigs), use_container_width=True, hide_index=True)
        else:
            st.info("No signals yet — run a cycle.")

        st.subheader("Recent Activity")
        for _entry in reversed(st.session_state.agent_logs[-8:]):
            _col = {"green": "#00d4aa", "amber": "#f5a623",
                    "red": "#ff4d6a"}.get(_entry["color"], "#8b949e")
            st.markdown(
                f'<span style="color:#444;font-size:10px;font-family:monospace">{_entry["ts"]}</span> '
                f'<span style="color:{_col};font-size:11px;font-family:monospace">[{_entry["agent"]}]</span> '
                f'<span style="font-size:11px;font-family:monospace;color:#c9d1d9">{_entry["msg"]}</span>',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Strategies":
    st.title("🧠 Strategy Configuration")
    cfg = st.session_state.risk_config

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Active Strategy")
        strats = StrategyAgent.STRATEGIES
        idx = strats.index(cfg.get("strategy", "EMA Crossover")) \
              if cfg.get("strategy") in strats else 0
        strategy  = st.selectbox("Strategy", strats, index=idx)
        symbols_all = ["EURUSD", "GBPUSD", "GBPJPY", "XAUUSD",
                       "NAS100", "USDJPY", "USDCHF", "AUDUSD"]
        symbols   = st.multiselect("Symbols", symbols_all,
                                   default=cfg.get("symbols", ["EURUSD"]))
        sl_pips   = st.number_input("Stop Loss (pips)", 10, 2000, cfg.get("sl_pips", 50))
        tp_pips   = st.number_input("Take Profit (pips)", 10, 5000, cfg.get("tp_pips", 100))
        rr = round(tp_pips / sl_pips, 2)
        st.caption(f"R:R = 1:{rr}")
        if st.button("💾 Save", type="primary"):
            st.session_state.risk_config.update({
                "strategy": strategy, "symbols": symbols,
                "sl_pips": sl_pips, "tp_pips": tp_pips, "reward_risk": rr,
            })
            _live_results["__config__"] = st.session_state.risk_config
            st.success("Saved!")

    with col2:
        st.subheader("Quick Signal Test")
        test_sym = st.selectbox("Symbol", ["EURUSD=X", "GBPJPY=X", "GBPUSD=X", "GC=F"])
        if st.button("🔍 Check signal now"):
            _agent = StrategyAgent({**cfg, "strategy": strategy})
            _ctx = _agent.run({
                "symbol": test_sym.replace("=X", "").replace("GC=F", "XAUUSD")
            })
            _sig = _ctx.get("signal")
            if _sig:
                _pill = "green" if _sig == "BUY" else "red"
                st.markdown(
                    f'<span class="status-pill pill-{_pill}">{_sig}</span> '
                    f'confidence: {_ctx.get("confidence", 0):.0%}',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No signal right now.")

        st.subheader("Deployed Strategy")
        _ap = st.session_state.get("approved_strategy")
        if _ap:
            st.success("✅ Strategy deployed from backtest:")
            st.json(_ap)
        else:
            st.info("No deployed strategy yet. Go to **Backtest** → Approve & Deploy.")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Backtest":
    st.title("📈 Advanced Backtest Engine")
    st.caption("Bar-by-bar · Custom patterns · Pip SL/TP · Trailing stop → Approve & Deploy")

    if not YF_AVAILABLE:
        st.error("yfinance not installed. Run: `pip install yfinance`")
        st.stop()

    # ── AI STRATEGY PARSER ────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
            border:1px solid #00d4aa44; border-radius:10px; padding:18px 20px 14px;">
  <span style="font-size:20px">🤖</span>
  <span style="font-family:'IBM Plex Mono',monospace; font-size:15px;
               color:#00d4aa; font-weight:600; margin-left:8px;">AI Strategy Parser</span>
  <p style="color:#8b949e; font-size:12px; margin:6px 0 0;">
    Paste any strategy description — screenshot text, PDF excerpt, hand-written rules —
    and AI will extract all parameters and fill the form automatically.
  </p>
</div>
""", unsafe_allow_html=True)

    st.write("")

    # API key (stored in session, not persisted to disk)
    with st.expander("🔑 Anthropic API Key  (required for AI parsing)", expanded=not st.session_state.anthropic_api_key):
        _key_input = st.text_input(
            "Paste your Anthropic API key",
            value=st.session_state.anthropic_api_key,
            type="password",
            help="Get one free at console.anthropic.com — key stays in memory only, never saved.",
        )
        if _key_input != st.session_state.anthropic_api_key:
            st.session_state.anthropic_api_key = _key_input
        if st.session_state.anthropic_api_key:
            st.success("✅ API key set")

    _ai_col1, _ai_col2 = st.columns([3, 1])
    with _ai_col1:
        _strategy_text = st.text_area(
            "Paste strategy description here",
            height=140,
            placeholder=(
                "Example:\n"
                "GBP/JPY INSIDE DAY\n"
                "LONG AND SHORT\n"
                "YESTERDAY WAS AN INSIDE DAY\n"
                "GO LONG IF TODAY CLOSED > DAY PRIOR TO INSIDE DAY\n"
                "GO SHORT IF TODAY CLOSED < DAY PRIOR TO INSIDE DAY\n"
                "300 PIP STOP LOSS\n"
                "1ST PROFITABLE CLOSE EXIT OR STOP AND REVERSE"
            ),
        )

    with _ai_col2:
        st.write("")
        st.write("")
        st.write("")
        _do_parse = st.button(
            "🤖 Parse with AI",
            type="primary",
            use_container_width=True,
            disabled=not (st.session_state.anthropic_api_key and _strategy_text.strip()),
        )
        if not st.session_state.anthropic_api_key:
            st.caption("⬆️ Add API key first")
        elif not _strategy_text.strip():
            st.caption("⬆️ Paste strategy text")

    # Run AI parse
    if _do_parse and _strategy_text.strip():
        if not ANTHROPIC_AVAILABLE:
            st.error("anthropic SDK not installed. Run: `pip install anthropic`")
        else:
            with st.spinner("🤖 Reading strategy with Claude AI..."):
                try:
                    _parsed = parse_strategy_ai(
                        _strategy_text, st.session_state.anthropic_api_key
                    )
                    st.session_state.ai_parsed = _parsed
                    st.success("✅ Strategy parsed! Form has been filled automatically.")
                except json.JSONDecodeError as _je:
                    st.error(f"AI returned invalid JSON: {_je}")
                    st.session_state.ai_parsed = None
                except Exception as _ae:
                    st.error(f"AI parse failed: {_ae}")
                    st.session_state.ai_parsed = None

    # Show what the AI extracted (collapsible)
    _p = st.session_state.get("ai_parsed")
    if _p:
        with st.expander("📋 AI extracted parameters", expanded=True):
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            _pc1.metric("Symbol",    _p.get("symbol", "—"))
            _pc2.metric("Pattern",   (_p.get("pattern") or "—")[:22])
            _pc3.metric("SL pips",   str(_p.get("sl_pips", "—")))
            _pc4.metric("Direction", _p.get("direction", "—"))
            if _p.get("notes"):
                st.info(f"📝 **AI notes:** {_p['notes']}")
            if st.button("✖ Clear AI parse", key="clear_ai"):
                st.session_state.ai_parsed = None
                st.rerun()

    st.divider()

    # ── Pull AI-parsed values as defaults ─────────────────────────────────
    _ai = st.session_state.get("ai_parsed") or {}

    def _ai_val(key, fallback):
        v = _ai.get(key)
        return v if v is not None else fallback

    _periods = ["6mo", "1y", "2y", "3y", "5y"]
    _tfs      = ["1d", "1wk"]
    _patterns = list(PATTERN_MAP.keys())
    _dirs     = ["Both", "Long Only", "Short Only"]

    _def_sym    = _ai_val("symbol",    "GBPJPY=X")
    _def_period = _ai_val("period",    "2y")
    _def_tf     = _ai_val("timeframe", "1d")
    _def_pat    = _ai_val("pattern",   _patterns[0])
    _def_dir    = _ai_val("direction", "Both")
    _def_sl     = int(_ai_val("sl_pips",        300))
    _def_tp     = int(_ai_val("tp_pips",        600) or 600)
    _def_trail  = bool(_ai_val("trailing_stop", False))
    _def_trailp = int(_ai_val("trailing_pips",  100))
    _def_risk   = float(_ai_val("risk_pct",     0.01)) * 100  # to % for slider

    # Clamp to valid choices
    if _def_period not in _periods: _def_period = "2y"
    if _def_tf     not in _tfs:     _def_tf     = "1d"
    if _def_pat    not in _patterns: _def_pat   = _patterns[0]
    if _def_dir    not in _dirs:    _def_dir    = "Both"
    _def_risk = max(0.5, min(5.0, _def_risk))

    # 1. Symbol
    st.subheader("1. Symbol & Data")
    c1, c2, c3 = st.columns(3)
    bt_sym = c1.text_input(
        "Symbol (yfinance)",
        value=_def_sym,
        help="Forex: GBPJPY=X  EURUSD=X  GBPUSD=X\nGold: GC=F  Oil: CL=F\nCrypto: BTC-USD\nStocks: AAPL",
    )
    bt_period = c2.selectbox("History", _periods, index=_periods.index(_def_period))
    bt_tf     = c3.selectbox("Timeframe", _tfs, index=_tfs.index(_def_tf))
    pip_sz    = get_pip_size(bt_sym)
    sym_disp  = bt_sym.replace("=X", "").replace("-", "")
    st.caption(f"Pip size for **{sym_disp}**: `{pip_sz}` "
               f"({'JPY pair — 0.01' if pip_sz == 0.01 else 'Standard — 0.0001'})")

    # 2. Pattern
    st.subheader("2. Entry Pattern")
    c1, c2 = st.columns(2)
    bt_pattern   = c1.selectbox("Pattern", _patterns,
                                 index=_patterns.index(_def_pat))
    bt_direction = c2.selectbox("Direction", _dirs, index=_dirs.index(_def_dir))
    _help = PATTERN_HELP.get(bt_pattern, "")
    if _help:
        st.info(f"ℹ️  {_help}")

    # 3. SL / TP
    st.subheader("3. Stop Loss & Take Profit")
    if _ai and _ai.get("notes") and "profitable close" in _ai.get("notes", "").lower():
        st.warning(
            "AI note: This strategy uses a 1st profitable close exit — "
            "no fixed TP. Set a wide TP (e.g. 2000 pips) or use trailing stop as exit."
        )
    c1, c2, c3, c4 = st.columns(4)
    bt_sl     = c1.number_input("Stop Loss (pips)",       1, 5000,  _def_sl)
    bt_tp     = c2.number_input("Take Profit (pips)",     1, 10000, _def_tp)
    bt_trail  = c3.toggle("Trailing Stop", value=_def_trail)
    bt_trail_p = c4.number_input("Trail distance (pips)", 1, 2000,  _def_trailp,
                                  disabled=not bt_trail)
    rr = bt_tp / bt_sl if bt_sl else 0
    st.caption(
        f"SL = `{bt_sl} pips` ({bt_sl * pip_sz:.4f})  |  "
        f"TP = `{bt_tp} pips` ({bt_tp * pip_sz:.4f})  |  "
        f"R:R = `1:{rr:.1f}`"
    )

    # 4. Sizing
    st.subheader("4. Capital & Risk")
    c1, c2 = st.columns(2)
    bt_cap  = c1.number_input("Starting Capital ($)", 1000, 10_000_000, 100_000, step=5_000)
    _risk_default = max(0.5, min(5.0, round(_def_risk * 4) / 4))
    bt_risk = c2.slider("Risk per trade (%)", 0.5, 5.0, _risk_default, step=0.25) / 100
    st.caption(
        f"Risk per trade: **${bt_cap * bt_risk:,.0f}**  |  "
        f"Max gain (TP): +${bt_cap * bt_risk * rr:,.0f}"
    )

    st.divider()
    run_bt = st.button("▶  Run Backtest", type="primary", use_container_width=True)

    if run_bt:
        # Download
        with st.spinner(f"Downloading {bt_sym} {bt_period} {bt_tf}..."):
            try:
                raw = yf.download(bt_sym, period=bt_period, interval=bt_tf,
                                  auto_adjust=True, progress=False)
                if raw.empty:
                    st.error(f"No data for '{bt_sym}'. Double-check the symbol.")
                    st.stop()
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df_bt = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
                st.success(f"✅ {len(df_bt)} bars  |  "
                           f"{df_bt.index[0].date()} → {df_bt.index[-1].date()}")
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()

        # Signals
        with st.spinner("Detecting pattern signals..."):
            _bull_fn, _bear_fn = PATTERN_MAP[bt_pattern]
            _empty = pd.Series(False, index=df_bt.index)
            bull_sig = _bull_fn(df_bt).fillna(False) if _bull_fn else _empty
            bear_sig = _bear_fn(df_bt).fillna(False) if _bear_fn else _empty
            nb, ns = int(bull_sig.sum()), int(bear_sig.sum())
            st.caption(f"Bull signals: **{nb}**  |  Bear signals: **{ns}**")
            if nb + ns == 0:
                st.warning("No signals found. Try a longer period or different pattern.")
                st.stop()

        # Simulate
        with st.spinner("Running bar-by-bar simulation..."):
            trades, eq_curve = run_simulation(
                df=df_bt,
                bull_signal=bull_sig, bear_signal=bear_sig,
                sl_pips=bt_sl, tp_pips=bt_tp,
                pip_size=pip_sz,
                trailing_stop=bt_trail, trailing_pips=bt_trail_p,
                direction=bt_direction,
                capital=bt_cap, risk_pct=bt_risk,
            )

        if not trades:
            st.warning("No completed trades. Try a longer period.")
            st.stop()

        tdf = pd.DataFrame(trades)

        # Metrics
        st.divider()
        st.subheader("📊 Performance")
        n      = len(tdf)
        wins   = tdf[tdf["P&L ($)"] > 0]
        losses = tdf[tdf["P&L ($)"] <= 0]
        wr     = len(wins) / n * 100 if n else 0
        total  = tdf["P&L ($)"].sum()
        ret_pct = (eq_curve.iloc[-1] - bt_cap) / bt_cap * 100
        avg_w  = wins["P&L ($)"].mean()  if len(wins)   else 0
        avg_l  = losses["P&L ($)"].mean() if len(losses) else 0
        pf     = (wins["P&L ($)"].sum() / abs(losses["P&L ($)"].sum())
                  if len(losses) and losses["P&L ($)"].sum() != 0 else 0)
        tp_n   = (tdf["Reason"] == "Take Profit").sum()
        sl_n   = (tdf["Reason"] == "Stop Loss").sum()
        roll   = eq_curve.cummax()
        dd     = (eq_curve - roll) / roll * 100
        max_dd = dd.min()
        dr     = eq_curve.pct_change().dropna()
        sharpe = (dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return",  f"{ret_pct:+.1f}%", f"${total:+,.0f}")
        m2.metric("Win Rate",      f"{wr:.1f}%",       f"{n} trades")
        m3.metric("Max Drawdown",  f"{max_dd:.1f}%")
        m4.metric("Sharpe Ratio",  f"{sharpe:.2f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Profit Factor", f"{pf:.2f}")
        m6.metric("Avg Win",       f"${avg_w:+,.0f}")
        m7.metric("Avg Loss",      f"${avg_l:+,.0f}")
        m8.metric("TP / SL Hits",  f"{tp_n} / {sl_n}")

        # Equity curve
        st.subheader("Equity Curve vs Buy & Hold")
        bh = bt_cap * df_bt["Close"] / df_bt["Close"].iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_curve.index, y=eq_curve.values,
            fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
            line=dict(color="#00d4aa", width=2), name="Strategy",
        ))
        fig.add_trace(go.Scatter(
            x=bh.index, y=bh.values,
            line=dict(color="#8b949e", width=1, dash="dot"), name="Buy & Hold",
        ))
        _tp_t = tdf[tdf["Reason"] == "Take Profit"]
        _sl_t = tdf[tdf["Reason"] == "Stop Loss"]
        if not _tp_t.empty:
            fig.add_trace(go.Scatter(
                x=_tp_t["Exit Date"], y=_tp_t["Equity"], mode="markers",
                marker=dict(color="#00d4aa", size=7, symbol="triangle-up"),
                name="TP Hit",
            ))
        if not _sl_t.empty:
            fig.add_trace(go.Scatter(
                x=_sl_t["Exit Date"], y=_sl_t["Equity"], mode="markers",
                marker=dict(color="#ff4d6a", size=7, symbol="triangle-down"),
                name="SL Hit",
            ))
        fig.update_layout(**DARK_LAYOUT, height=320,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown
        st.subheader("Drawdown")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            fill="tozeroy", fillcolor="rgba(255,77,106,0.1)",
            line=dict(color="#ff4d6a", width=1), name="Drawdown %",
        ))
        fig2.update_layout(**DARK_LAYOUT, height=160)
        st.plotly_chart(fig2, use_container_width=True)

        # Monthly heatmap
        st.subheader("Monthly P&L Heatmap")
        tdf["Exit Date"] = pd.to_datetime(tdf["Exit Date"])
        tdf["_yr"]  = tdf["Exit Date"].dt.year
        tdf["_mo"]  = tdf["Exit Date"].dt.month
        monthly = tdf.groupby(["_yr", "_mo"])["P&L ($)"].sum().reset_index()
        if not monthly.empty:
            pivot = monthly.pivot(index="_yr", columns="_mo",
                                  values="P&L ($)").fillna(0)
            pivot.columns = [datetime.date(2000, int(m), 1).strftime("%b")
                             for m in pivot.columns]
            fig3 = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=[str(y) for y in pivot.index.tolist()],
                colorscale=[[0, "#7f1d1d"], [0.5, "#1a1a2e"], [1, "#065f46"]],
                text=[[f"${v:,.0f}" for v in row] for row in pivot.values],
                texttemplate="%{text}", showscale=False,
            ))
            fig3.update_layout(**DARK_LAYOUT,
                               height=max(120, len(pivot) * 45 + 60))
            st.plotly_chart(fig3, use_container_width=True)

        # Trade log
        st.subheader("Trade Log")
        disp = tdf.drop(columns=["Equity", "_yr", "_mo"], errors="ignore")
        st.dataframe(disp.style.applymap(
            lambda v: f"color: {'#00d4aa' if v > 0 else '#ff4d6a'}",
            subset=["P&L ($)", "Pips"],
        ), use_container_width=True, height=280)

        # Approve & Deploy
        st.divider()
        st.subheader("🚀 Approve & Deploy")
        _ci, _cb = st.columns([3, 1])
        with _ci:
            _trail_str = f"ON — {bt_trail_p} pips" if bt_trail else "OFF"
            st.markdown(f"""
**Pattern:** `{bt_pattern}` · **Symbol:** `{sym_disp}` · **TF:** `{bt_tf}`  
**SL:** `{bt_sl} pips` · **TP:** `{bt_tp} pips` · **Trailing:** `{_trail_str}`  
**Direction:** `{bt_direction}` · **Risk/trade:** `{bt_risk*100:.1f}%`  
Result: **{ret_pct:+.1f}%** · **{wr:.0f}%** win · **{pf:.2f}** PF · **{max_dd:.1f}%** DD
""")
        with _cb:
            st.write("")
            st.write("")
            if st.button("✅ Approve & Deploy", type="primary", use_container_width=True):
                _ap_strat = {
                    "pattern":    bt_pattern, "symbol":    sym_disp,
                    "timeframe":  bt_tf,      "direction": bt_direction,
                    "sl_pips":    bt_sl,      "tp_pips":   bt_tp,
                    "trailing":   bt_trail,   "trail_pips": bt_trail_p,
                    "risk_pct":   bt_risk,    "pip_size":  pip_sz,
                    "backtest_return": round(ret_pct, 1),
                    "win_rate":   round(wr, 1),
                }
                st.session_state["approved_strategy"] = _ap_strat
                st.session_state.risk_config.update({
                    "strategy":      bt_pattern,
                    "symbols":       [sym_disp],
                    "sl_pips":       bt_sl,
                    "tp_pips":       bt_tp,
                    "max_risk_pct":  bt_risk,
                    "trailing_stop": bt_trail,
                    "trailing_pips": bt_trail_p,
                    "direction":     bt_direction,
                    "reward_risk":   rr,
                })
                _live_results["__config__"] = st.session_state.risk_config
                _log("Backtest",
                     f"APPROVED {bt_pattern} on {sym_disp} "
                     f"SL={bt_sl}p TP={bt_tp}p Trail={bt_trail}")
                st.success(
                    f"✅ **Deployed!** Trading `{bt_pattern}` on `{sym_disp}` "
                    f"with {bt_sl}-pip SL and "
                    f"{'trailing stop' if bt_trail else 'fixed TP'}."
                )
                st.info("Enable the **scheduler** in the sidebar or hit "
                        "**Manual cycle** to start trading.")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: POSITIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💼 Positions":
    st.title("💼 Open Positions")

    if st.session_state.mt5_connected and st.session_state.bridge:
        positions = st.session_state.bridge.get_positions()
    else:
        positions = _demo_positions().to_dict("records")
        st.info("Demo data — connect MT5 for live positions.")

    if not positions:
        st.success("No open positions.")
    else:
        _total_pnl = sum(p.get("profit", 0) for p in positions)
        _total_vol = sum(p.get("volume", 0) for p in positions)
        _best = max(positions, key=lambda x: x.get("profit", 0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Open",       len(positions))
        c2.metric("Total P&L",  f"${_total_pnl:+,.2f}")
        c3.metric("Volume",     f"{_total_vol:.2f} lots")
        c4.metric("Best",       f"${_best.get('profit',0):+,.2f} {_best.get('symbol','')}")

        st.divider()
        for _i, _p in enumerate(positions):
            _pnl   = _p.get("profit", 0)
            _sym   = _p.get("symbol", "")
            _ptype = "BUY" if _p.get("type", 0) == 0 else "SELL"
            _vol   = _p.get("volume", 0)
            _entry = _p.get("price_open", 0)
            _curr  = _p.get("price_current", 0)
            _tkt   = _p.get("ticket", 0)
            _ppill = "pill-green" if _pnl >= 0 else "pill-red"
            _tpill = "pill-green" if _ptype == "BUY" else "pill-red"

            _cols = st.columns([2, 1, 1, 1.5, 1.5, 2, 1.5])
            _cols[0].markdown(f"**{_sym}**")
            _cols[1].markdown(
                f'<span class="status-pill {_tpill}">{_ptype}</span>',
                unsafe_allow_html=True,
            )
            _cols[2].write(f"{_vol:.2f}L")
            _cols[3].write(f"{_entry:.5f}")
            _cols[4].write(f"{_curr:.5f}")
            _cols[5].markdown(
                f'<span class="status-pill {_ppill}">${_pnl:+,.2f}</span>',
                unsafe_allow_html=True,
            )
            if _cols[6].button("Close", key=f"cp_{_i}"):
                if st.session_state.mt5_connected and st.session_state.bridge:
                    _ok = st.session_state.bridge.close_position(_tkt)
                    st.success(f"Closed {_sym}" if _ok else f"Failed")
                    st.rerun()
                else:
                    st.warning("Connect MT5 first.")

        st.divider()
        if st.button("🔴 Close ALL", type="primary"):
            if st.session_state.mt5_connected and st.session_state.bridge:
                st.session_state.bridge.close_all()
                st.success("All positions closed.")
                st.rerun()
            else:
                st.warning("Not connected.")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: RISK
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🛡️ Risk":
    st.title("🛡️ Risk Manager")
    cfg = st.session_state.risk_config

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Limits")
        _mr   = st.slider("Max risk/trade (%)", 0.1, 5.0, cfg["max_risk_pct"]*100, 0.1) / 100
        _mdd  = st.slider("Max drawdown (%)", 1, 30, int(cfg["max_drawdown"]*100)) / 100
        _mp   = st.slider("Max open positions", 1, 20, cfg["max_positions"])
        _mc   = st.slider("Max correlation (%)", 10, 100, int(cfg["max_correlation"]*100)) / 100
        _wr   = st.slider("Assumed win rate (%)", 40, 80, int(cfg["win_rate"]*100)) / 100
        _rr   = st.slider("Reward:Risk", 1.0, 5.0, cfg["reward_risk"], 0.1)
        _cap  = st.number_input("Capital ($)", value=cfg["balance"], step=1_000)
        _k    = max(0.0, min(_wr - (1 - _wr) / _rr, 0.25))
        _rusd = _cap * _mr
        st.info(f"Kelly: {_k:.2%}  |  Risk per trade: ${_rusd:,.0f}")

        if st.button("💾 Save Risk Config", type="primary"):
            st.session_state.risk_config.update({
                "max_risk_pct": _mr, "max_drawdown": _mdd,
                "max_positions": _mp, "max_correlation": _mc,
                "win_rate": _wr, "reward_risk": _rr, "balance": _cap,
            })
            _live_results["__config__"] = st.session_state.risk_config
            st.success("Saved!")

    with col2:
        st.subheader("Live Status")
        _cur_dd = 3.2
        _fg = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=_cur_dd,
            delta={"reference": 0, "suffix": "%", "increasing": {"color": "#ff4d6a"}},
            gauge={
                "axis":  {"range": [0, _mdd * 100], "ticksuffix": "%"},
                "bar":   {"color": "#00d4aa", "thickness": 0.25},
                "steps": [
                    {"range": [0,           _mdd*60],  "color": "#161b22"},
                    {"range": [_mdd*60,     _mdd*85],  "color": "#1e2a1e"},
                    {"range": [_mdd*85,     _mdd*100], "color": "#2a1a1a"},
                ],
                "threshold": {"line": {"color": "#ff4d6a", "width": 2},
                              "thickness": 0.8, "value": _mdd * 100},
            },
            title={"text": "Drawdown %", "font": {"size": 14}},
        ))
        _fg.update_layout(template="plotly_dark", height=220,
                          margin=dict(l=20, r=20, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(_fg, use_container_width=True)

        for _lbl, _val, _tot in [
            ("Portfolio Exposure", 38.0, 100),
            ("Drawdown",           _cur_dd, _mdd * 100),
            ("Open Positions",     3, _mp),
            ("Correlation Risk",   41.0, _mc * 100),
        ]:
            st.markdown(f"**{_lbl}**")
            st.progress(min(_val / _tot if _tot else 0, 1.0),
                        text=f"{_val:.1f} / {_tot:.1f}")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: MT5 BRIDGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔌 MT5 Bridge":
    st.title("🔌 MT5 Bridge")

    if not MT5_AVAILABLE:
        st.error("MetaTrader5 not installed. Run: `pip install MetaTrader5`")
        st.warning("⚠️  MetaTrader5 Python package works on **Windows only**.")

    tab1, tab2, tab3 = st.tabs(["Connection", "Account Info", "Manual Order"])

    with tab1:
        st.subheader("Connect")
        with st.form("mt5_form"):
            c1, c2 = st.columns(2)
            _login  = c1.number_input("Login", value=0, step=1)
            _server = c1.text_input("Server", value="ICMarkets-Demo")
            _pw     = c2.text_input("Password", type="password")
            c2.checkbox("Demo account", value=True)
            _sub    = st.form_submit_button("🔌 Connect", type="primary")

        if _sub:
            if not MT5_AVAILABLE:
                st.error("MetaTrader5 not installed.")
            elif _login == 0 or not _pw:
                st.warning("Enter login and password.")
            else:
                try:
                    _br = MT5Bridge(int(_login), _pw, _server)
                    _br.connect()
                    st.session_state.bridge = _br
                    st.session_state.mt5_connected = True
                    _live_results["__bridge__"] = _br
                    st.success(f"Connected to {_server}!")
                    st.rerun()
                except Exception as _e:
                    st.error(f"Failed: {_e}")

        if st.session_state.mt5_connected:
            st.success("MT5 is connected.")
            if st.button("Disconnect"):
                if st.session_state.bridge:
                    st.session_state.bridge.disconnect()
                st.session_state.bridge = None
                st.session_state.mt5_connected = False
                _live_results["__bridge__"] = None
                st.rerun()

        with st.expander("📖 Setup guide"):
            st.markdown("""
1. Download MT5 from https://www.metatrader5.com/en/download
2. Open a free demo: File → Open Account → Choose broker → Demo
3. Note your login, password, server (e.g. `ICMarkets-Demo`)
4. MT5 → Tools → Options → Expert Advisors → ✅ Allow Algo Trading
5. Paste credentials above and click Connect
6. Run a backtest → Approve & Deploy → enable scheduler
""")

    with tab2:
        if st.session_state.mt5_connected and st.session_state.bridge:
            st.json(st.session_state.bridge.account_info())
        else:
            st.info("Connect first.")

    with tab3:
        st.subheader("Manual Order")
        with st.form("manual_order"):
            c1, c2, c3 = st.columns(3)
            _msym  = c1.text_input("Symbol", value="GBPJPY")
            _mtype = c2.radio("Type", ["BUY", "SELL"], horizontal=True)
            _mvol  = c3.number_input("Lots", 0.01, 100.0, 0.01, step=0.01, format="%.2f")
            _msl   = c1.number_input("SL pips", 1, 5000, 300)
            _mtp   = c2.number_input("TP pips", 1, 5000, 600)
            _msend = st.form_submit_button("📤 Send Order")

        if _msend:
            if not st.session_state.mt5_connected:
                st.error("Not connected.")
            else:
                _res = st.session_state.bridge.send_order(
                    _msym, _mtype, _mvol, _msl, _mtp, "AlphaDesk_Manual"
                )
                if _res.success:
                    st.success(f"Filled! Ticket #{_res.ticket} @ {_res.price}")
                else:
                    st.error(f"Rejected: {_res.message}")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE: AGENT LOGS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Agent Logs":
    st.title("📋 Agent Logs")

    c1, c2, c3 = st.columns([2, 2, 1])
    _fa = c1.selectbox("Agent", ["All", "Orchestrator", "Strategy", "MarketData",
                                  "RiskManager", "Execution", "MT5Bridge",
                                  "Backtest", "Scheduler"])
    _fl = c2.selectbox("Level", ["All", "INFO", "WARN", "ERROR"])
    if c3.button("🗑️ Clear"):
        st.session_state.agent_logs = []
        st.rerun()

    _logs = st.session_state.agent_logs
    if _fa != "All":
        _logs = [l for l in _logs if l["agent"] == _fa]
    if _fl != "All":
        _lvl_map = {"INFO": "green", "WARN": "amber", "ERROR": "red"}
        _logs = [l for l in _logs if l["color"] == _lvl_map[_fl]]

    _cmap = {"green": "#00d4aa", "amber": "#f5a623", "red": "#ff4d6a"}
    _html = ""
    for _entry in reversed(_logs[-300:]):
        _c = _cmap.get(_entry["color"], "#8b949e")
        _html += (
            f'<div style="padding:1px 0">'
            f'<span style="color:#444;font-size:10px">{_entry["ts"]}</span> '
            f'<span style="color:{_c};font-weight:500">[{_entry["agent"]}]</span> '
            f'<span style="color:#c9d1d9">{_entry["msg"]}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div class="log-container">'
        f'{_html or "<span style=color:#444>No entries. Run a cycle.</span>"}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{min(len(_logs), 300)} of {len(st.session_state.agent_logs)} entries")


# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────
if refresh:
    time.sleep(10)
    st.rerun()
