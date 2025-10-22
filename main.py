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

def _parse_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default

BUY_AMOUNT  = _parse_float("BUY_AMOUNT", 50.0)          # $ per buy (default $50)
MONTHLY_CAP = _parse_float("MONTHLY_CAP", 300.0)        # max invested per calendar month (all assets combined)

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

# Universe (Fidelity platform backtest)
STOCKS = [s.strip().upper() for s in os.getenv("STOCKS", "VIG,GLD,BND").split(',') if s.strip()]
CRYPTO = [c.strip().upper() for c in os.getenv("CRYPTO", "BTC/USD").split(',') if c.strip()]

# Indicators (global defaults for non-BTC)
RSI_LEN  = int(os.getenv("RSI_LEN", "14"))
MA_SHORT = int(os.getenv("MA_SHORT", "60"))     # 60 x 15min
MA_LONG  = int(os.getenv("MA_LONG", "240"))     # 240 x 15min

# Data frequency
BAR_MINUTES = int(os.getenv("BAR_MINUTES", "15"))  # 15-min bars
# Optional Alpaca stock feed override (e.g., "iex" on free plan). Leave empty to use API default.
STOCK_FEED = os.getenv("STOCK_FEED", "").strip()

# -----------------------
# Date helpers
# -----------------------
def _parse_dates() -> Tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp.now(tz=TZ).normalize()
    end = pd.Timestamp(END_DATE, tz=TZ) if END_DATE else today
    start = pd.Timestamp(START_DATE, tz=TZ) if START_DATE else (end - pd.Timedelta(days=365))
    return start, end

# -----------------------
# API clients
# -----------------------
def _alpaca_clients():
    if not (ALPACA_API_KEY and ALPACA_API_SECRET):
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_API_SECRET env vars")
    stock = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    crypto = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    return stock, crypto

def _ensure_tz(index: pd.Index, target_tz: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(target_tz)

# -----------------------
# Data fetchers (+ fallback)
# -----------------------
def _stock_bars_request(symbols: List[str], minutes: int, start: pd.Timestamp, end: pd.Timestamp):
    kwargs = dict(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(minutes, TimeFrameUnit.Minute),
        start=start.tz_convert("UTC").to_pydatetime(),
        end=end.tz_convert("UTC").to_pydatetime(),
        adjustment=None,
    )
    if STOCK_FEED:
        kwargs["feed"] = STOCK_FEED
    return StockBarsRequest(**kwargs)

def _fetch_stock_bars(symbols: List[str], start: pd.Timestamp, end: pd.Timestamp, minutes: int) -> pd.DataFrame:
    stock_client, _ = _alpaca_clients()

    # Try requested timeframe first (e.g., 15m)
    try:
        req = _stock_bars_request(symbols, minutes, start, end)
        bars = stock_client.get_stock_bars(req)
        df = bars.df
    except Exception as e:
        print(f"[WARN] Stock {minutes}m fetch error: {e}", file=sys.stderr)
        df = pd.DataFrame()

    if df is not None and not df.empty:
        close = df.reset_index().pivot(index="timestamp", columns="symbol", values="close")
        close.index = _ensure_tz(close.index, TZ)
        return close.sort_index()

    # Fallback: fetch 1m and resample to target minutes for CLOSE
    try:
        print(f"[INFO] No stock data at {minutes}m; falling back to 1m -> resample {minutes}m")
        req1 = _stock_bars_request(symbols, 1, start, end)
        bars1 = stock_client.get_stock_bars(req1)
        df1 = bars1.df
        if df1 is None or df1.empty:
            return pd.DataFrame()
        close1 = df1.reset_index().pivot(index="timestamp", columns="symbol", values="close")
        close1.index = _ensure_tz(close1.index, TZ)
        close = close1.resample(f"{minutes}T").last().dropna(how="all")
        return close.sort_index()
    except Exception as e:
        print(f"[WARN] Stock 1m fallback failed: {e}", file=sys.stderr)
        return pd.DataFrame()

def _fetch_crypto_bars(pairs: List[str], start: pd.Timestamp, end: pd.Timestamp, minutes: int) -> pd.DataFrame:
    _, crypto_client = _alpaca_clients()
    tf = TimeFrame(minutes, TimeFrameUnit.Minute)
    req = CryptoBarsRequest(
        symbol_or_symbols=pairs,
        timeframe=tf,
        start=start.tz_convert("UTC").to_pydatetime(),
        end=end.tz_convert("UTC").to_pydatetime(),
    )
    try:
        bars = crypto_client.get_crypto_bars(req)
        df = bars.df
        if df is None or df.empty:
            return pd.DataFrame()
        close = df.reset_index().pivot(index="timestamp", columns="symbol", values="close")
        close.index = _ensure_tz(close.index, TZ)
        close.columns = [c.replace("/", "") for c in close.columns]
        return close.sort_index()
    except Exception as e:
        print(f"[WARN] Crypto {minutes}m fetch error: {e}", file=sys.stderr)
        return pd.DataFrame()

# -----------------------
# Strategy bits
# -----------------------
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
    """
    Base rule for non-BTC assets:
        RSI14 < 30 AND MA_SHORT < MA_LONG  (e.g., 60 < 240 on 15m)
    Special rule for BTC only:
        RSI14 < 30 AND MA180 < MA720      (15m bars)
    """
    # --- Base signals for all assets ---
    rsi  = close.apply(_rsi, length=RSI_LEN)
    ma_s = close.rolling(MA_SHORT, min_periods=MA_SHORT).mean()
    ma_l = close.rolling(MA_LONG,  min_periods=MA_LONG).mean()
    signals = ((rsi < 30) & (ma_s < ma_l)).fillna(False)

    # --- BTC override: stricter MA windows (180 / 720 on the same 15m bars) ---
    btc_cols = [c for c in close.columns if c.upper() in ("BTCUSD", "BTCUSDT", "BTC")]
    if btc_cols:
        btc_ma_s = close[btc_cols].rolling(180, min_periods=180).mean()
        btc_ma_l = close[btc_cols].rolling(720, min_periods=720).mean()
        btc_signals = ((rsi[btc_cols] < 30) & (btc_ma_s < btc_ma_l)).fillna(False)
        signals.loc[:, btc_cols] = btc_signals

    return signals.fillna(False)

@dataclass
class PortfolioState:
    positions: Dict[str, float]
    total_invested: float

def run_backtest(close: pd.DataFrame, signals: pd.DataFrame) -> Tuple[pd.DataFrame, PortfolioState, Dict[str, int], Dict[str, float]]:
    # Contribution-only model (no cash account)
    state = PortfolioState(positions={sym: 0.0 for sym in close.columns}, total_invested=0.0)
    buys_count: Dict[str, int] = {sym: 0 for sym in close.columns}

    last_buy_date: Dict[str, pd.Timestamp.date] = {sym: None for sym in close.columns}  # one buy per asset/day
    monthly_spend: Dict[str, float] = {}  # "YYYY-MM" -> dollars invested this month

    equity_curve = []

    for ts, row in close.iterrows():
        current_date  = ts.date()
        current_month = ts.strftime("%Y-%m")
        month_spent = monthly_spend.get(current_month, 0.0)

        # If monthly cap reached, skip buys this bar
        if MONTHLY_CAP > 0 and month_spent >= MONTHLY_CAP:
            portfolio_value = float(np.nansum([row[s]*state.positions[s] for s in close.columns]))
            equity_curve.append((ts, portfolio_value))
            continue

        sigs = signals.loc[ts]
        for sym, fire in sigs.items():
            if not fire:
                continue
            if last_buy_date[sym] == current_date:
                continue  # one buy per asset per day
            if MONTHLY_CAP > 0 and monthly_spend.get(current_month, 0.0) + BUY_AMOUNT > MONTHLY_CAP + 1e-9:
                break      # shared cap reached at this bar
            px = row.get(sym)
            if not (np.isfinite(px) and px > 0):
                continue
            if BUY_AMOUNT <= 0:
                continue

            # Execute "buy" by adding fractional units; track invested contributions
            qty = BUY_AMOUNT / px
            state.positions[sym] += qty
            buys_count[sym] += 1
            last_buy_date[sym] = current_date
            monthly_spend[current_month] = monthly_spend.get(current_month, 0.0) + BUY_AMOUNT
            state.total_invested += BUY_AMOUNT

            if MONTHLY_CAP > 0 and monthly_spend[current_month] >= MONTHLY_CAP:
                break

        portfolio_value = float(np.nansum([row[s]*state.positions[s] for s in close.columns]))
        equity_curve.append((ts, portfolio_value))

    equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"]).set_index("timestamp")
    return equity_df, state, buys_count, monthly_spend

# -----------------------
# Reporting
# -----------------------
def summarize(
    equity: pd.DataFrame,
    state: PortfolioState,
    buys: Dict[str, int],
    close: pd.DataFrame,
    monthly_spend: Dict[str, float],
) -> None:
    final_value = float(equity["equity"].iloc[-1]) if not equity.empty else 0.0
    invested = state.total_invested
    if invested > 0:
        roi_pct = (final_value / invested - 1.0) * 100.0
        roi_str = f"{roi_pct:.2f}%"
    else:
        roi_str = "N/A (no buys)"

    # drawdown on portfolio value
    if not equity.empty:
        peak = equity["equity"].cummax()
        drawdown = (equity["equity"] / peak - 1.0)
        max_dd = float(drawdown.min() * 100.0)
    else:
        max_dd = 0.0

    print("\n========== FIDELITY BACKTEST (15m; BTC uses MA180/720; $ DCA on signal; 1 buy/asset/day; monthly cap) ==========")
    print(f"Window: {close.index.min()}  →  {close.index.max()}  [{BAR_MINUTES}m bars, TZ={TZ}]")
    print(f"Universe: {list(close.columns)}")
    print(f"Total Invested (contributions): ${invested:,.2f}")
    print(f"Final Portfolio Value:           ${final_value:,.2f}")
    print(f"Net P&L:                         ${final_value - invested:,.2f}")
    print(f"ROI:                             {roi_str}  |  Max DD: {max_dd:.2f}%")

    print("\n--- Buys per asset ---")
    for sym in close.columns:
        last_px = close[sym].dropna().iloc[-1] if close[sym].notna().any() else np.nan
        mv = (state.positions[sym] * last_px) if np.isfinite(last_px) else 0.0
        last_px_str = f"{last_px:.2f}" if np.isfinite(last_px) else "NA"
        print(f"{sym:>8}: {buys.get(sym,0)} buys | Last Price: {last_px_str} | Position: {state.positions[sym]:.6f} units | Market Value: ${mv:,.2f}")

    print("\n--- Monthly spend (applied vs cap) ---")
    for month in sorted(monthly_spend.keys()):
        spent = monthly_spend[month]
        pct = min(100.0, (spent / MONTHLY_CAP) * 100.0) if MONTHLY_CAP > 0 else 0.0
        print(f"{month}: ${spent:,.2f} / ${MONTHLY_CAP:,.2f} ({pct:.0f}%)")

    print("\n--- Sample of last 10 equity points ---")
    print(equity.tail(10).to_string())

# -----------------------
# Main
# -----------------------
def main():
    start, end = _parse_dates()
    stock_close  = _fetch_stock_bars(STOCKS, start, end, BAR_MINUTES) if STOCKS else pd.DataFrame()
    crypto_close = _fetch_crypto_bars(CRYPTO, start, end, BAR_MINUTES) if CRYPTO else pd.DataFrame()

    frames = [df for df in [stock_close, crypto_close] if not df.empty]
    if not frames:
        raise RuntimeError("No price data returned for any asset. Check API keys, feeds, symbols, or date range.")

    close = pd.concat(frames, axis=1).sort_index()

    # Keep only requested instruments and drop columns with no data at all
    desired = [*STOCKS, *[c.replace('/', '') for c in CRYPTO]]
    close = close.loc[:, [c for c in close.columns if c in desired]]
    empty_cols = [c for c in close.columns if close[c].isna().all()]
    if empty_cols:
        print(f"[INFO] Dropping assets with no data: {empty_cols}")
        close = close.drop(columns=empty_cols)

    if close.shape[1] == 0:
        raise RuntimeError("All requested assets had no data.")

    signals = build_signals(close)
    equity, state, buys, monthly_spend = run_backtest(close, signals)
    summarize(equity, state, buys, close, monthly_spend)

if __name__ == "__main__":
    main()
