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
START_DATE = os.getenv("START_DATE")  # e.g. "2020-10-22"
END_DATE   = os.getenv("END_DATE")    # e.g. "2025-10-22"

def _parse_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default

BUY_AMOUNT  = _parse_float("BUY_AMOUNT", 50.0)          # $ per buy
MONTHLY_CAP = _parse_float("MONTHLY_CAP", 300.0)        # shared cap per calendar month

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

# Universe (Fidelity platform backtest)
STOCKS = [s.strip().upper() for s in os.getenv("STOCKS", "VIG,GLD,BND").split(',') if s.strip()]
CRYPTO = [c.strip().upper() for c in os.getenv("CRYPTO", "BTC/USD").split(',') if c.strip()]

# Indicators (global defaults for non-BTC)
RSI_LEN  = int(os.getenv("RSI_LEN", "14"))
MA_SHORT = int(os.getenv("MA_SHORT", "60"))     # 60 x 15m
MA_LONG  = int(os.getenv("MA_LONG", "240"))     # 240 x 15m

# Data frequency
BAR_MINUTES = int(os.getenv("BAR_MINUTES", "15"))  # 15-min bars
# Optional Alpaca stock feed override (e.g., "iex"). Leave empty to use API default.
STOCK_FEED = os.getenv("STOCK_FEED", "").strip()

# -----------------------
# Date helpers
# -----------------------
def _parse_dates() -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Default to a 5-year window ending today (TZ-aware) if START_DATE not provided."""
    today = pd.Timestamp.now(tz=TZ).normalize()
    end = pd.Timestamp(END_DATE, tz=TZ) if END_DATE else today
    if START_DATE:
        start = pd.Timestamp(START_DATE, tz=TZ)
    else:
        start = end - pd.Timedelta(days=365 * 5)
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
    # Try requested timeframe first
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

    # Fallback: 1m → resample to target minutes (last close)
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
        close.columns = [c.replace("/", "") for c in close.columns]  # BTC/USD -> BTCUSD
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
    # Base signals
    rsi  = close.apply(_rsi, length=RSI_LEN)
    ma_s = close.rolling(MA_SHORT, min_periods=MA_SHORT).mean()
    ma_l = close.rolling(MA_LONG,  min_periods=MA_LONG).mean()
    signals = ((rsi < 30) & (ma_s < ma_l)).fillna(False)

    # BTC override
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

def run_backtest(close: pd.DataFrame, signals: pd.DataFrame) -> Tuple[
    pd.DataFrame, PortfolioState, Dict[str, int], Dict[str, float], Dict[str, pd.Timestamp]
]:
    # Contribution-only (no cash)
    state = PortfolioState(positions={sym: 0.0 for sym in close.columns}, total_invested=0.0)
    buys_count: Dict[str, int] = {sym: 0 for sym in close.columns}

    last_buy_date: Dict[str, pd.Timestamp.date] = {sym: None for sym in close.columns}  # one buy per asset/day
    monthly_spend: Dict[str, float] = {}  # "YYYY-MM" -> $
    first_buy_time: Dict[str, pd.Timestamp] = {sym: None for sym in close.columns}

    equity_curve = []

    for ts, row in close.iterrows():
        current_date  = ts.date()
        current_month = ts.strftime("%Y-%m")
        month_spent = monthly_spend.get(current_month, 0.0)

        if MONTHLY_CAP > 0 and month_spent >= MONTHLY_CAP:
            portfolio_value = float(np.nansum([row[s]*state.positions[s] for s in close.columns]))
            equity_curve.append((ts, portfolio_value))
            continue

        sigs = signals.loc[ts]
        for sym, fire in sigs.items():
            if not fire:
                continue
            if last_buy_date[sym] == current_date:
                continue
            if MONTHLY_CAP > 0 and monthly_spend.get(current_month, 0.0) + BUY_AMOUNT > MONTHLY_CAP + 1e-9:
                break
            px = row.get(sym)
            if not (np.isfinite(px) and px > 0):
                continue
            if BUY_AMOUNT <= 0:
                continue

            qty = BUY_AMOUNT / px
            state.positions[sym] += qty
            buys_count[sym] += 1
            last_buy_date[sym] = current_date
            monthly_spend[current_month] = monthly_spend.get(current_month, 0.0) + BUY_AMOUNT
            state.total_invested += BUY_AMOUNT

            if first_buy_time[sym] is None:
                first_buy_time[sym] = ts

            if MONTHLY_CAP > 0 and monthly_spend[current_month] >= MONTHLY_CAP:
                break

        portfolio_value = float(np.nansum([row[s]*state.positions[s] for s in close.columns]))
        equity_curve.append((ts, portfolio_value))

    equity_df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"]).set_index("timestamp")
    return equity_df, state, buys_count, monthly_spend, first_buy_time

# -----------------------
# Diagnostics helpers (no strategy changes)
# -----------------------
def _to_et(ts: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return ts.tz_convert("America/New_York")

def _rth_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Regular Trading Hours mask for US equities: 09:30–16:00 ET, Mon–Fri."""
    et = _to_et(index)
    is_weekday = et.dayofweek < 5
    # minutes since midnight
    minutes = et.hour * 60 + et.minute
    in_rth = (minutes >= 9*60+30) & (minutes < 16*60)
    return pd.Series(is_weekday & in_rth, index=index)

def _hour_hist(series: pd.Series, max_bins: int = 6) -> str:
    """Compact hour-of-day distribution string of non-NaN bars in ET."""
    et = _to_et(series.dropna().index)
    if len(et) == 0:
        return "none"
    hours = et.hour.value_counts().sort_index()
    # show top bins
    top = hours.sort_values(ascending=False).head(max_bins)
    return ", ".join([f"{h}: {int(n)}" for h, n in top.items()])

# -----------------------
# Reporting
# -----------------------
def summarize(
    equity: pd.DataFrame,
    state: PortfolioState,
    buys: Dict[str, int],
    close: pd.DataFrame,
    monthly_spend: Dict[str, float],
    signals: pd.DataFrame,
    first_buy_time: Dict[str, pd.Timestamp],
) -> None:
    start_dt, end_dt = close.index.min(), close.index.max()
    years = max((end_dt - start_dt).days / 365.25, 1e-9)

    final_value = float(equity["equity"].iloc[-1]) if not equity.empty else 0.0
    invested = state.total_invested

    if invested > 0:
        total_return = final_value / invested
        roi_pct = (total_return - 1.0) * 100.0
        cagr = (total_return ** (1 / years) - 1) * 100.0
        roi_str = f"{roi_pct:.2f}%"
        cagr_str = f"{cagr:.2f}%/yr"
    else:
        roi_str, cagr_str = "N/A", "N/A"

    # drawdown
    if not equity.empty:
        peak = equity["equity"].cummax()
        drawdown = (equity["equity"] / peak - 1.0)
        max_dd = float(drawdown.min() * 100.0)
    else:
        max_dd = 0.0

    print("\n========== FIDELITY BACKTEST (15m; BTC uses MA180/720; $ DCA; 1 buy/asset/day; monthly cap) ==========")
    print(f"Feed: {'default' if not STOCK_FEED else STOCK_FEED} | Timeframe: {BAR_MINUTES}m | TZ: {TZ}")
    print(f"Window: {start_dt} → {end_dt}  ({years:.2f} years)")
    print(f"Universe: {list(close.columns)}")
    print(f"Total Invested: ${invested:,.2f} | Final Value: ${final_value:,.2f} | Net P&L: ${final_value - invested:,.2f}")
    print(f"ROI: {roi_str} | CAGR: {cagr_str} | Max DD: {max_dd:.2f}%")

    # ---- Per-asset ROI & CAGR ----
    print("\n--- Per-asset returns ---")
    for sym in close.columns:
        px = close[sym].dropna()
        mv = px.iloc[-1] * state.positions[sym] if not px.empty else 0.0
        invested_sym = buys[sym] * BUY_AMOUNT
        if invested_sym > 0 and mv > 0:
            roi_sym = (mv / invested_sym - 1) * 100.0
            fb = first_buy_time.get(sym)
            yrs_sym = max((end_dt - (fb if fb is not None else start_dt)).days / 365.25, 1e-9)
            cagr_sym = ((mv / invested_sym) ** (1 / yrs_sym) - 1) * 100.0
            print(f"{sym:>8}: {buys[sym]:>4} buys | ROI {roi_sym:7.2f}% | CAGR {cagr_sym:7.2f}%/yr | MV ${mv:,.2f}")
        else:
            print(f"{sym:>8}: {buys[sym]:>4} buys | ROI    N/A  | CAGR    N/A   | MV ${mv:,.2f}")

    # ---- Monthly spend ----
    print("\n--- Monthly spend (applied vs cap) ---")
    for month in sorted(monthly_spend.keys()):
        spent = monthly_spend[month]
        pct = min(100.0, (spent / MONTHLY_CAP) * 100.0) if MONTHLY_CAP > 0 else 0.0
        print(f"{month}: ${spent:,.2f} / ${MONTHLY_CAP:,.2f} ({pct:.0f}%)")

    # ---- Signal diagnostics ----
    sig_counts = signals.sum()
    print("\n--- Signal diagnostics (total TRUE bars) ---")
    for sym in close.columns:
        c = int(sig_counts.get(sym, 0))
        if c == 0:
            print(f"{sym:>8}: 0 signals")
        else:
            first_dt = signals.index[signals[sym]].min()
            last_dt  = signals.index[signals[sym]].max()
            print(f"{sym:>8}: {c} signals | first: {first_dt} | last: {last_dt}")

    # ---- Monthly signal counts per asset ----
    print("\n--- Monthly signal counts ---")
    month_index = signals.index.strftime("%Y-%m")
    for sym in close.columns:
        counts = signals[sym].groupby(month_index).sum()
        nonzero = counts[counts > 0]
        if nonzero.empty:
            print(f"{sym:>8}: none")
        else:
            # show up to 12 most recent non-zero months
            recent = nonzero.tail(12)
            items = ", ".join([f"{m}:{int(v)}" for m, v in recent.items()])
            print(f"{sym:>8}: {items}")

    # ---- Data diagnostics ----
    print("\n--- Data diagnostics ---")
    for sym in close.columns:
        px = close[sym]
        non_nan = px.notna()
        rth = _rth_mask(px.index)
        rth_non_nan = (non_nan & rth).sum()
        rth_share = (rth_non_nan / max(rth.sum(), 1)) * 100.0
        print(f"{sym:>8}: bars={len(px)} | NaN={px.isna().sum()} | first={px.index.min()} | last={px.index.max()} | RTH non-NaN share: {rth_share:.1f}% | ET hour hist: {_hour_hist(px)}")

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
    equity, state, buys, monthly_spend, first_buy_time = run_backtest(close, signals)
    summarize(equity, state, buys, close, monthly_spend, signals, first_buy_time)

if __name__ == "__main__":
    main()
