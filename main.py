def summarize(
    equity: pd.DataFrame,
    state: PortfolioState,
    buys: Dict[str, int],
    close: pd.DataFrame,
    monthly_spend: Dict[str, float],
    signals: pd.DataFrame,
) -> None:
    # --- timing ---
    start_dt, end_dt = close.index.min(), close.index.max()
    years = (end_dt - start_dt).days / 365.25

    final_value = float(equity["equity"].iloc[-1]) if not equity.empty else 0.0
    invested = state.total_invested

    # portfolio-level metrics
    if invested > 0:
        total_return = final_value / invested
        roi_pct = (total_return - 1.0) * 100.0
        cagr = (total_return ** (1 / max(years, 1e-6)) - 1) * 100.0
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
    print(f"Window: {start_dt} → {end_dt}  ({years:.2f} years)  [{BAR_MINUTES}m bars, TZ={TZ}]")
    print(f"Universe: {list(close.columns)}")
    print(f"Total Invested: ${invested:,.2f} | Final Value: ${final_value:,.2f} | Net P&L: ${final_value - invested:,.2f}")
    print(f"ROI: {roi_str} | CAGR: {cagr_str} | Max DD: {max_dd:.2f}%")

    # ---- Per-asset ROI & CAGR ----
    print("\n--- Per-asset returns ---")
    for sym in close.columns:
        px = close[sym].dropna()
        mv = px.iloc[-1] * state.positions[sym] if not px.empty else 0.0
        # total invested per asset = buys * BUY_AMOUNT
        invested_sym = buys[sym] * BUY_AMOUNT
        roi_sym = (mv / invested_sym - 1) * 100 if invested_sym > 0 else 0
        # find first buy timestamp for CAGR horizon
        buy_dates = signals.index[signals[sym]]
        first_buy_dt = buy_dates.min() if not buy_dates.empty else start_dt
        yrs_sym = (end_dt - first_buy_dt).days / 365.25
        cagr_sym = ((mv / invested_sym) ** (1 / max(yrs_sym, 1e-6)) - 1) * 100 if invested_sym > 0 else 0
        print(f"{sym:>8}: {buys[sym]:>4} buys | ROI {roi_sym:6.2f}% | CAGR {cagr_sym:6.2f}%/yr | MV ${mv:,.2f}")

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

    # ---- Data info ----
    print("\n--- Data diagnostics ---")
    for sym in close.columns:
        px = close[sym]
        print(f"{sym:>8}: bars={len(px)} | NaN={px.isna().sum()} | first={px.index.min()} | last={px.index.max()}")

    print("\n--- Sample of last 10 equity points ---")
    print(equity.tail(10).to_string())
