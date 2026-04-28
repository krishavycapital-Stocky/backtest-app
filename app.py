import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
from strategy import load_data, run_backtest

st.set_page_config(page_title="NIFTY Backtest Dashboard", layout="wide")
st.title("📈 NIFTY Backtesting Dashboard")

with st.sidebar:
    st.header("⚙️ Settings")
    parquet_file = st.text_input("Parquet file path", value="data.parquet")

    st.subheader("Timeframe")
    tf_choice   = st.selectbox("Resample to", ["15min", "30min", "1h", "4h", "1D"])
    resample_tf = tf_choice

    strategy_name = st.selectbox("Strategy", ["EMA Crossover", "RSI", "ADX + EMA"])

    st.subheader("EMA Settings")
    fast_ema = st.slider("Fast EMA", min_value=3,  max_value=50,  value=9)
    slow_ema = st.slider("Slow EMA", min_value=10, max_value=200, value=21)

    if strategy_name == "RSI":
        st.subheader("RSI Settings")
        rsi_period = st.slider("RSI Period",          min_value=5,  max_value=30, value=14)
        rsi_buy    = st.slider("Buy when RSI above",  min_value=40, max_value=80, value=60)
        rsi_sell   = st.slider("Sell when RSI below", min_value=30, max_value=70, value=55)
    else:
        rsi_period, rsi_buy, rsi_sell = 14, 60, 55

    if strategy_name == "ADX + EMA":
        st.subheader("ADX Settings")
        adx_period    = st.slider("ADX Period",    min_value=5,  max_value=30, value=14)
        adx_threshold = st.slider("ADX Threshold", min_value=10, max_value=50, value=25)
    else:
        adx_period, adx_threshold = 14, 25

    st.subheader("Capital & Lots")
    capital  = st.number_input("Starting Capital (₹)", value=100000, step=10000)
    lot_size = st.number_input("Lot Size", value=15, step=1,
                                help="Auto-filled if found in data. NIFTY=75, BNF=15, FINNIFTY=40")
    num_lots = st.number_input("Number of Lots", value=1, step=1)
    position_pct = st.slider("Position size %", 10, 100, 95) / 100

    run_btn = st.button("▶ Run Backtest", use_container_width=True, type="primary")

if run_btn:
    with st.spinner("Running backtest..."):
        try:
            df, detected_lot = load_data(parquet_file, resample_tf=resample_tf)
            st.caption(f"Loaded {len(df)} candles on {tf_choice} timeframe")
        except Exception as e:
            st.error(f"Could not load data: {e}")
            st.stop()

        # Use detected lot size if found in data
        if detected_lot:
            st.sidebar.success(f"Lot size detected from data: {detected_lot}")
            lot_size = detected_lot

        try:
            stats, bt = run_backtest(
                df, strategy_name=strategy_name,
                fast_ema=fast_ema, slow_ema=slow_ema,
                rsi_buy=rsi_buy, rsi_sell=rsi_sell, rsi_period=rsi_period,
                adx_period=adx_period, adx_threshold=adx_threshold,
                starting_capital=capital, position_pct=position_pct)
        except Exception as e:
            st.error(f"Backtest error: {e}")
            st.stop()

    # Key metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Return",  f"{stats.get('Return [%]', 0):.2f}%")
    col2.metric("Max Drawdown",  f"{stats.get('Max. Drawdown [%]', 0):.2f}%")
    col3.metric("Sharpe Ratio",  f"{stats.get('Sharpe Ratio', 0):.2f}")
    col4.metric("Win Rate",      f"{stats.get('Win Rate [%]', 0):.2f}%")
    col5.metric("Total Trades",  str(stats.get('# Trades', 0)))

    st.divider()

    # Equity curve
    st.subheader("Equity Curve")
    equity = stats.get('_equity_curve')
    if equity is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity['Equity'],
                                  mode='lines', name='Portfolio',
                                  line=dict(color='#2563eb', width=1.5)))
        fig.add_hline(y=capital, line_dash="dash", line_color="gray",
                      annotation_text="Starting capital")
        fig.update_layout(height=350, xaxis_title="Date",
                          yaxis_title="Portfolio Value (₹)",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Trade log
    st.subheader("Trade Log")
    trades = stats.get('_trades')
    if trades is not None and not trades.empty:
        t = trades.copy()

        t['Entry Date']  = pd.to_datetime(t['EntryTime']).dt.strftime('%d-%b-%Y')
        t['Entry Time']  = pd.to_datetime(t['EntryTime']).dt.strftime('%H:%M')
        t['Exit Date']   = pd.to_datetime(t['ExitTime']).dt.strftime('%d-%b-%Y')
        t['Exit Time']   = pd.to_datetime(t['ExitTime']).dt.strftime('%H:%M')
        t['Entry Price'] = t['EntryPrice'].round(2)
        t['Exit Price']  = t['ExitPrice'].round(2)
        t['PnL (pts)']   = t['PnL'].round(2)
        t['PnL (₹)']     = (t['PnL'] * lot_size * num_lots).round(0).astype(int)
        t['Return %']    = t['ReturnPct'].round(2)

        display = t[['Entry Date', 'Entry Time', 'Exit Date', 'Exit Time',
                      'Entry Price', 'Exit Price',
                      'PnL (pts)', 'PnL (₹)', 'Return %']]

        def color_pnl(val):
            return 'color: green' if val > 0 else 'color: red'

        st.dataframe(
            display.style.map(color_pnl, subset=['PnL (pts)', 'PnL (₹)']),
            use_container_width=True, height=400
        )

        # P&L Summary
        total_rs   = int(t['PnL (₹)'].sum())
        winning    = int((t['PnL (₹)'] > 0).sum())
        losing     = int((t['PnL (₹)'] < 0).sum())
        avg_trade  = int(total_rs / max(len(t), 1))

        st.divider()
        st.subheader("P&L Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total P&L",       f"₹{total_rs:,}")
        s2.metric("Winning Trades",  str(winning))
        s3.metric("Losing Trades",   str(losing))
        s4.metric("Avg per Trade",   f"₹{avg_trade:,}")

    else:
        st.info("No trades with these parameters.")

    with st.expander("Full stats"):
        clean = {k: str(v) for k, v in stats.items() if not str(k).startswith('_')}
        st.json(clean)

else:
    st.info("👈 Pick a timeframe + strategy and click ▶ Run Backtest")