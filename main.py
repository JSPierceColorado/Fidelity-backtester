from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except Exception as e:
    print("alpaca-py is required. pip install alpaca-py", file=sys.stderr)
    raise

# -----------------------
# Config via ENV
# -----------------------
TZ = os.getenv("TZ", "America/Denver")
START_DATE = os.getenv("START_DATE")  # e.g. "2024-10-22"
END_DATE   = os.getenv("END_DATE")    # e.g. "2025-10-22"
INIT_CASH  = float(os.getenv("INIT_CASH", "10000"))
BUY_AMOUNT = float(os.getenv("BUY_AMOUNT", "50"))      # $ per signal
MONTHLY_CAP = float(os.getenv("MONTHLY_CAP", "300"))   # max invested per calendar month

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

# Universe (Fidelity platform backtest)
STOCKS = [s.strip().upper() for s in os.getenv("STOCKS", "VIG,GLD,BND").split(',') if s.strip()]
CRYPTO = [c.strip().upper() for c in os.getenv("CRYPTO", "BTC/USD").split(',') if c.strip()]

# Indicators
RSI_LEN  = int(os.getenv("RSI_LEN", "14"))
MA_SHORT = int(os.getenv("MA_SHORT", "60"))     # 60 x 15min
MA_LONG  = int(os.getenv("MA_LONG", "240"))     # 240 x 15min

# Data frequency
BAR_MINUTES = int(os.getenv("BAR_MINUTES", "15"))  # 15-min bars


def _parse_dates() -> Tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp.now(tz=TZ).normalize()
    end = pd.Timestamp(END_DATE, tz=TZ) if END_DATE else today
    start = pd.Timestamp(START_DATE, tz=TZ) if START_DATE else (end - pd.Timedelta(days=365))
    return start, end


def _alpaca_clients():
    if not (ALPACA_API_KEY and ALPACA_API_SECRET):
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET env vars")
    stock = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    crypto = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    return stock, crypto


def _ensure_tz(index: pd.Index, target_tz: str) -> pd.DatetimeIndex:
    """Make index tz-aware in target_tz. If already tz-aware, just convert."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(target_tz)


def _fetch_stock_bars(symbols: List[str], start: pd.Timestamp, end: pd.Timestamp, minutes: int) -> pd.DataFrame:
    stock_client, _ = _alpaca_clients()
    tf = TimeFrame(minutes, TimeFrameUnit.Minute)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf,
        start=start.tz_convert("UTC").to_pydatetime(),
        end=end.tz_convert("UTC").to_pydatetime(),
        adjustment=None,
    )
    bars = stock_client.get_stock_bars(req)
    df = bars.df
    if df.empty:
        return pd.DataFrame()
    close = df.reset_index().pivot(index="timestamp", columns="symbol", values="close")
    close.index = _ensure_tz(close.index, TZ)
    return close.sort_index()


def _fetch_crypto_bars(pairs: List[str], start: pd.Timestamp, end: pd.Timestamp, minutes: int) -> pd.DataFrame:
    _, crypto_client = _alpaca_clients()
    tf = TimeFrame(minutes, TimeFrameUnit.Minute)
    req = CryptoBarsRequest(
        symbol_or_symbols=pairs,
        timeframe=tf,
        start=start.tz_convert("UTC").to_pydatetime(),
        end=end.tz_convert("UTC").to_pydatetime(),
    )
    bars = crypto_client.get_crypto_bars(req)
    df = bars.df
    if df.empty:
        return pd.DataFrame()
    close = df.reset_index().pivot(index="timestamp", columns="symbol", values="close")
    close.index = _ensure_tz(close.index, TZ)
    # Normalize crypto column names (BTC/USD -> BTCUSD)
    close.columns = [c.replace("/", "") for c in close.columns]
    return close.sort_index()


def _rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def build_signals(close: pd.DataFrame) -> pd.DataFrame:
    rsi  = close.apply(_rsi, length=RSI_LEN)
    ma_s = close.rolling(MA_SHORT, min_periods=MA_SHORT).mean()
    ma_l = close.rolling(MA_LONG,  min_periods=MA_LONG).mean()
    signals = (rsi < 30) & (ma_s < ma_l)
    return signals.fillna(False)


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, float]


def run_backtest(close: pd.DataFrame, signals: pd.DataFrame) -> Tuple[pd.DataFrame, PortfolioState, Dict[str, int]]:
    state = PortfolioState(cash=INIT_CASH, positions={sym: 0.0 for sym in close.columns})
    buys_count: Dict[str, int] = {sym: 0 for sym in close.columns}

    # New: enforce one buy per asset per day + monthly cap
    last_buy_date: Dict[str, pd.Timestamp.date] = {sym: None for sym in close.columns}
    monthly_spend: Dict[str, float] = {}  # "YYYY-MM" -> dollars invested this month

    equity_curve = []

    for ts, row in close.iterrows():
        # Current calendar date & month in target TZ (index is already TZ-adjusted)
        current_date  = ts.date()                 # per-day constraint
        current_month = ts.strftime("%Y-%m")      # monthly cap bucket
        month_spent = monthly_spend.get(current_month, 0.0)

        # If monthly cap reached, skip all buys this bar
        if month_spent >= MONTHLY_CAP:
            # compute equity and continue
            equity = state.cash + float(np.nansum([row[s]*state.positions[s] for s in close.columns]))
            equity_curve.append((ts, equity))
            continue

        # Process signals for this timestamp
        sigs = signals.loc[ts]
        for sym, fire in sigs.items():
            if not fire:
                continue

            # Enforce one buy per asset per calendar day
            if last_buy_date[sym] == current_date:
                continue

            # Check monthly cap again before each buy
            if monthly_spend.get(current_month, 0.0) >= MONTHLY_CAP:
                break  # stop attempting buys this bar; cap reached

            px = row[sym]
            if not (np.isfinite(px) and px > 0):
                continue
            if state.cash < BUY_AMOUNT:
                continue  # out of cash

            # Execute buy ($BUY_AMOUNT)
            qty = BUY_AMOUNT / px
            state.cash -= BUY_AMOUNT
            state.positions[sym] += qty
            buys_count[sym] += 1
            last_buy_date[sym] = current_date
            monthly_spend[current_month] = monthly_spend.get(current_month, 0.0) + BUY_AMOUNT

            # If we hit the monthly cap exactly, stop further buys this bar
            if monthly_spend[current_month] >= MONTHLY_CAP:
                break

        # Compute equity after processing this bar
        equity = state.cash + float(np.nansum([row[s]*state.positions[s] for s in close.columns]))
        equity_curve.append((ts, equity))

    equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"]).set_index("timestamp")
    return equity_df, state, buys_count


def summarize(equity: pd.DataFrame, init_cash: float, state: PortfolioState, buys: Dict[str, int], close: pd.DataFrame) -> None:
    final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else init_cash
    total_return = (final_equity / init_cash - 1.0) * 100.0
    peak = equity["equity"].cummax()
    drawdown = (equity["equity"] / peak - 1.0)
    max_dd = float(drawdown.min() * 100.0) if not drawdown.empty else 0.0

    invested = INIT_CASH - state.cash
    print("\n========== FIDELITY BACKTEST (15m RSI<30 & MA60<MA240, BUY $50; 1 buy/asset/day; $300 monthly cap) ==========")
    print(f"Window: {close.index.min()}  →  {close.index.max()}  [{BAR_MINUTES}m bars, TZ={TZ}]")
    print(f"Universe: {list(close.columns)}")
    print(f"Initial Cash: ${init_cash:,.2f}")
    print(f"Total Invested: ${invested:,.2f}")
    print(f"Final Equity: ${final_equity:,.2f}  |  Total Return: {total_return:.2f}%  |  Max DD: {max_dd:.2f}%")

    print("\n--- Buys per asset ---")
    for sym in close.columns:
        print(f"{sym:>8}: {buys.get(sym,0)} buys | Last Price: {close[sym].iloc[-1]:.2f} | Position: {state.positions[sym]:.6f} units | Market Value: ${state.positions[sym]*close[sym].iloc[-1]:.2f}")

    print("\n--- Sample of last 10 equity points ---")
    print(equity.tail(10).to_string())

    print("\n--- P&L by asset ---")
    for sym in close.columns:
        mv = state.positions[sym]*close[sym].iloc[-1]
        print(f"{sym:>8}: Market Value ${mv:,.2f}")


def main():
    start, end = _parse_dates()
    stock_close  = _fetch_stock_bars(STOCKS, start, end, BAR_MINUTES) if STOCKS else pd.DataFrame()
    crypto_close = _fetch_crypto_bars(CRYPTO, start, end, BAR_MINUTES) if CRYPTO else pd.DataFrame()

    frames = [df for df in [stock_close, crypto_close] if not df.empty]
    if not frames:
        raise RuntimeError("No price data returned. Check API keys, symbols, or date range.")

    close = pd.concat(frames, axis=1).sort_index()
    desired = [*STOCKS, *[c.replace('/', '') for c in CRYPTO]]
    close = close.loc[:, [c for c in close.columns if c in desired]]
    close = close.dropna(how='all', axis=1)

    signals = build_signals(close)
    equity, state, buys = run_backtest(close, signals)
    summarize(equity, INIT_CASH, state, buys, close)


if __name__ == "__main__":
    main()
