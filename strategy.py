import pandas as pd
import numpy as np

def load_data_from_df(raw_df, option_type="CALL", interval=1, resample_tf="15min"):
    df = raw_df.copy()
    df = df[df['option_type'] == option_type].copy()
    df = df[df['interval_min'] == interval].copy()

    detected_lot = None
    if 'symbol' in df.columns:
        sym = str(df['symbol'].iloc[0]).upper()
        lot_map = {'BANKNIFTY': 15, 'FINNIFTY': 40,
                   'MIDCPNIFTY': 50, 'NIFTY': 65, 'SENSEX': 10}
        for name, size in lot_map.items():
            if name in sym:
                detected_lot = size
                break

    df['datetime'] = pd.to_datetime(df['datetime'], utc=False)
    df = df.set_index('datetime')
    df.index = df.index.tz_localize(None)

    df = df[['spot']].copy()
    df.columns = ['Close']
    df['Open']   = df['Close']
    df['High']   = df['Close']
    df['Low']    = df['Close']
    df['Volume'] = 1

    df = df.resample(resample_tf).agg({
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum'
    }).dropna()

    return df.sort_index(), detected_lot

def load_data(parquet_path, option_type="CALL", interval=1, resample_tf="15min"):
    raw_df = pd.read_parquet(parquet_path)
    return load_data_from_df(raw_df, option_type, interval, resample_tf)

def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_adx(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = low.diff().abs().clip(lower=0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr      = tr.ewm(span=period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=period,  adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(span=period, adjust=False).mean(), plus_di, minus_di

def run_backtest(df, strategy_name="EMA Crossover",
                 fast_ema=9, slow_ema=21,
                 rsi_buy=60, rsi_sell=55, rsi_period=14,
                 adx_period=14, adx_threshold=25,
                 starting_capital=100000, position_pct=0.95):

    from backtesting import Backtest, Strategy

    if strategy_name == "EMA Crossover":
        fast = compute_ema(df['Close'], fast_ema).values
        slow = compute_ema(df['Close'], slow_ema).values

        class SelectedStrategy(Strategy):
            def init(self):
                self.fast = self.I(lambda: fast, name='Fast EMA')
                self.slow = self.I(lambda: slow, name='Slow EMA')
            def next(self):
                if not self.position:
                    if self.fast[-2] < self.slow[-2] and self.fast[-1] > self.slow[-1]:
                        self.buy(size=position_pct)
                else:
                    if self.fast[-2] > self.slow[-2] and self.fast[-1] < self.slow[-1]:
                        self.position.close()

    elif strategy_name == "RSI":
        rsi_vals = compute_rsi(df['Close'], rsi_period).values

        class SelectedStrategy(Strategy):
            def init(self):
                self.rsi = self.I(lambda: rsi_vals, name='RSI')
            def next(self):
                if not self.position:
                    if self.rsi[-2] < rsi_buy and self.rsi[-1] >= rsi_buy:
                        self.buy(size=position_pct)
                else:
                    if self.rsi[-1] < rsi_sell:
                        self.position.close()

    elif strategy_name == "ADX + EMA":
        fast = compute_ema(df['Close'], fast_ema).values
        slow = compute_ema(df['Close'], slow_ema).values
        adx, pdi, mdi = compute_adx(df, adx_period)

        class SelectedStrategy(Strategy):
            def init(self):
                self.fast = self.I(lambda: fast,        name='Fast EMA')
                self.slow = self.I(lambda: slow,        name='Slow EMA')
                self.adx  = self.I(lambda: adx.values,  name='ADX')
                self.pdi  = self.I(lambda: pdi.values,  name='+DI')
                self.mdi  = self.I(lambda: mdi.values,  name='-DI')
            def next(self):
                if not self.position:
                    if (self.adx[-1] > adx_threshold and
                            self.fast[-1] > self.slow[-1] and
                            self.pdi[-1]  > self.mdi[-1]):
                        self.buy(size=position_pct)
                else:
                    if self.fast[-1] < self.slow[-1] or self.pdi[-1] < self.mdi[-1]:
                        self.position.close()

    bt = Backtest(df, SelectedStrategy,
                  cash=starting_capital,
                  commission=0.0005,
                  exclusive_orders=True)
    return bt.run(), bt
