"""
🚀 MEAN REVERSION MAX YIELD Bot — GBP/USD H1
===============================================
Same mechanism validated on AUD/USD, re-tested BLIND (no tuning)
on 5 other instruments -- GBP/USD was a clean hit, then
re-optimized specifically for it (with a stop-floor of 1.0x ATR
enforced during the search -- the first unfiltered search found a
13.1% win rate / 0.2x ATR "winner" that was almost certainly the
same fragile-tight-stop trap that cost real money on a different
bot earlier in this portfolio's build; this version is the
healthier, sensibly-stopped result instead):

  - Backtest (Master-mode, 1 pip placeholder slippage -- no real
      GBP/USD fill data yet, same starting point every new
      instrument in this portfolio began from):
      478 trades, 31.0% win rate, +$1,694 net profit, 8.58% max DD
  - Walk-Forward Optimization: 5/5 folds profitable out-of-sample
      (strict per-fold re-optimizing)
  - Monte Carlo (2000 resamples of the actual trades): 88.3%
      probability of profit
  - Standalone annualized: ~2.86%/yr. Portfolio-level, added
      alongside the existing bots: total return 13.38% -> 15.50%/yr,
      drawdown 3.42% -> 3.34% (real diversification, not just added risk)

Lower win rate than this portfolio's other bots (31% vs the usual
38-45%) -- not a red flag on its own, Mean Reversion targets the
Bollinger Band midline rather than a fixed reward:risk ratio, so
win rate and payoff size trade off differently here than in the
ATR-target bots. Worth watching once live data comes in, same as
everything else new in this portfolio.

GBP/USD is not shared with any other bot in this portfolio.

⚠️ Real slippage on GBP/USD is unmeasured -- the 1-pip backtest
assumption is a placeholder. Watch real fills once live and
re-check the numbers, same process as BTC and Gold both needed.

⚠️ Same standing caveat as every validated bot here: recommend
demo-account-first before live/funded capital.
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from match_trader_api import MatchTraderClient
from trade_logger import init_db, log_trade_open, log_trade_close
from master_account_safety import handle_master_account_safety

BOT_NAME = "mean_reversion_gbpusd"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

SYMBOL       = "GBP_USD"
GRANULARITY  = "H1"
CANDLE_COUNT = 100   # BB_PERIOD=60 needs at least 60+5=65 candles; 100 gives headroom

# ForexLab-validated parameters (see caveats above)
BB_PERIOD    = 90
BB_STD       = 1.6
RSI_PERIOD   = 50
RSI_LOWER    = 17
RSI_UPPER    = 61
ATR_PERIOD   = 35
ATR_SL_MULT  = 1.25

RISK_PCT     = 0.0020   # 0.20% -- new bot, no live track record yet (blind-test hit: 5/5 OOS folds, 88.3% MC)
LOOP_SLEEP   = 3600

# ── Telegram ───────────────────────────────────────────────────────────────────
def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        logger.warning(f"Telegram error: {e}")

# ── Indicators ─────────────────────────────────────────────────────────────────
def compute_indicators(df):
    df = df.copy()
    df["bb_mid"]   = df["close"].rolling(BB_PERIOD).mean()
    bb_std_val     = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std_val
    df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std_val
    
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss     = (-delta).clip(lower=0).rolling(RSI_PERIOD).mean()
    rs       = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = df.apply(
        lambda r: max(r["high"] - r["low"],
                      abs(r["high"] - r["prev_close"]),
                      abs(r["low"]  - r["prev_close"])), axis=1)
    df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()
    return df

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    client = MatchTraderClient()
    if not client.login():
        logger.error("❌ Login Failed.")
        return

    logger.info("🚀 GBP/USD Mean Reversion Bot Started.")
    send_telegram("🚀 GBP/USD Mean Reversion Bot Started | Risk: 0.20%")
    init_db()

    logged_trade = None  # logging-only state -- does not affect trading logic
    last_signal_candle_ts = None  # candle-cooldown guard

    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.weekday() >= 5:
                time.sleep(3600)
                continue

            balance = client.get_balance()
            if balance is None:
                time.sleep(60)
                continue

            positions = client.get_open_positions(SYMBOL)

            if handle_master_account_safety(client, logged_trade, BOT_NAME, send_telegram):
                logged_trade = None
                time.sleep(LOOP_SLEEP)
                continue

            if not positions and logged_trade:
                balance_after = balance
                realized_pnl = None
                if balance_after is not None and logged_trade.get("balance_before") is not None:
                    realized_pnl = round(balance_after - logged_trade["balance_before"], 2)
                log_trade_close(
                    bot_name=BOT_NAME, order_id=logged_trade["position_id"],
                    exit_price=None, exit_time=datetime.now(timezone.utc), realized_pnl=realized_pnl,
                )
                logged_trade = None

            if positions: # Only one position at a time per symbol
                time.sleep(LOOP_SLEEP)
                continue

            df = client.get_candles(SYMBOL, CANDLE_COUNT, GRANULARITY)
            if df is None or len(df) < BB_PERIOD + 5:
                time.sleep(60)
                continue

            # ── Candle-cooldown guard ────────────────────────────────────────
            # This bot polls every 5 minutes but trades H1 candles. Without this,
            # the same candle's signal re-fires on every poll until the candle
            # closes -- so a stop-out can be immediately followed by re-entry on
            # the identical signal, repeatedly. (That exact loop cost ~$200 on a
            # BTC bot on 2026-08-06.) The Match Trader candle payload carries no
            # timestamp, so each candle is fingerprinted by its OHLC values.
            # ─────────────────────────────────────────────────────────────────
            _last = df.iloc[-1]
            candle_ts = (round(_last["close"],5), round(_last["high"],5), round(_last["low"],5))
            if candle_ts == last_signal_candle_ts:
                time.sleep(LOOP_SLEEP)
                continue

            df   = compute_indicators(df)
            last = df.iloc[-1]

            bb_upper, bb_lower, bb_mid, rsi, atr, close = last["bb_upper"], last["bb_lower"], last["bb_mid"], last["rsi"], last["atr"], last["close"]

            if any(np.isnan(v) for v in [bb_upper, bb_lower, bb_mid, rsi, atr]):
                time.sleep(60)
                continue

            sl_dist = ATR_SL_MULT * atr
            lots    = client.calculate_lots(balance, RISK_PCT, sl_dist, SYMBOL)
            if lots <= 0:
                time.sleep(60)
                continue

            # LONG Signal
            if close < bb_lower and rsi < RSI_LOWER:
                sl = round(close - sl_dist, 5)
                tp = round(bb_mid, 5)
                logger.info(f"🔼 LONG {SYMBOL} | Entry:{close} SL:{sl} TP:{tp}")
                order_id, err = client.open_position(SYMBOL, "BUY", lots, sl, tp)
                if order_id:
                    logged_trade = {"position_id": order_id, "balance_before": balance}
                    log_trade_open(bot_name=BOT_NAME, symbol=SYMBOL, direction="BUY",
                                    order_id=order_id, entry_price=close,
                                    entry_time=datetime.now(timezone.utc), sl=sl, tp=tp, lot_size=lots)
                    last_signal_candle_ts = candle_ts
                    send_telegram(f"✅ LONG {SYMBOL} Opened\nEntry: {close} | SL: {sl} | TP: {tp}")

            # SHORT Signal
            elif close > bb_upper and rsi > RSI_UPPER:
                sl = round(close + sl_dist, 5)
                tp = round(bb_mid, 5)
                logger.info(f"🔽 SHORT {SYMBOL} | Entry:{close} SL:{sl} TP:{tp}")
                order_id, err = client.open_position(SYMBOL, "SELL", lots, sl, tp)
                if order_id:
                    logged_trade = {"position_id": order_id, "balance_before": balance}
                    log_trade_open(bot_name=BOT_NAME, symbol=SYMBOL, direction="SELL",
                                    order_id=order_id, entry_price=close,
                                    entry_time=datetime.now(timezone.utc), sl=sl, tp=tp, lot_size=lots)
                    last_signal_candle_ts = candle_ts
                    send_telegram(f"✅ SHORT {SYMBOL} Opened\nEntry: {close} | SL: {sl} | TP: {tp}")

        except Exception as e:
            logger.error(f"🔥 Error: {e}")

        time.sleep(LOOP_SLEEP)

if __name__ == "__main__":
    main()
