import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
from tabulate import tabulate  # For better table formatting
import logging

# Configure logging to print only errors and critical information
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(message)s"
)

# Parameters
STOCK_LIST = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV',
    'BPCL', 'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 'GRASIM',
    'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK',
    'INFY', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC', 'POWERGRID',
    'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA', 'TCS', 'TATACONSUM' , 'TATAMOTORS', 'TATASTEEL', 'TECHM',
    'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO'
]  # Only ticker names
START_DATE = '2021-01-01'
END_DATE = '2025-04-18'
CASH_AT_HAND = 100000
STOP_LOSS_RANGE = [0.02, 0.05, 0.1]  # 2% to 10%
TARGET_RANGE = [0.02, 0.05, 0.1, 0.15, 0.2]  # 2% to 20%
COMMISSION_PERCENT = 0.2  # Commission as a percentage (e.g., 0.1% = 0.001)

# Strategy class
class Strategy:
    @staticmethod
    def moving_average_crossover(data, short_window=50, long_window=200):
        if len(data) < long_window:
            data['Signal'] = np.nan
            return data

        data['SMA50'] = data['Close'].rolling(window=short_window).mean()
        data['SMA200'] = data['Close'].rolling(window=long_window).mean()
        data['Signal'] = 0
        data.loc[data['SMA50'] > data['SMA200'], 'Signal'] = 1
        data.loc[data['SMA50'] <= data['SMA200'], 'Signal'] = -1
        return data

    @staticmethod
    def rsi_strategy(data, rsi_period=14, overbought=70, oversold=30):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['Signal'] = 0
        data.loc[data['RSI'] < oversold, 'Signal'] = 1  # Buy signal
        data.loc[data['RSI'] > overbought, 'Signal'] = -1  # Sell signal
        return data

    @staticmethod
    def bollinger_bands(data, window=20, num_std_dev=2):
        try:
            data['SMA'] = data['Close'].rolling(window=window).mean()
            data['Upper Band'] = data['SMA'] + (num_std_dev * data['Close'].rolling(window=window).std())
            data['Lower Band'] = data['SMA'] - (num_std_dev * data['Close'].rolling(window=window).std())
            data = data.dropna()  # Ensure all columns are aligned
            data['Signal'] = 0
            data.loc[data['Close'] < data['Lower Band'], 'Signal'] = 1  # Buy signal
            data.loc[data['Close'] > data['Upper Band'], 'Signal'] = -1  # Sell signal
            print(f"Bollinger Bands calculated successfully for data:\n{data[['Close', 'SMA', 'Upper Band', 'Lower Band']].tail()}")
            print(f"Data alignment check for Bollinger Bands:\n{data[['Close', 'SMA', 'Upper Band', 'Lower Band']].tail()}")
            return data
        except Exception as e:
            print(f"Error in Bollinger Bands strategy: {e}")
            raise

    @staticmethod
    def macd_strategy(data, short_window=12, long_window=26, signal_window=9):
        try:
            data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
            data = data.dropna()  # Ensure all columns are aligned
            data['Signal'] = 0
            data.loc[data['MACD'] > data['Signal Line'], 'Signal'] = 1
            data.loc[data['MACD'] <= data['Signal Line'], 'Signal'] = -1
            print(f"MACD calculated successfully for data:\n{data[['Close', 'MACD', 'Signal Line']].tail()}")
            return data
        except Exception as e:
            print(f"Error in MACD strategy: {e}")
            raise

    @staticmethod
    def stochastic_oscillator(data, k_period=14, d_period=3, overbought=80, oversold=20):
        data['L14'] = data['Low'].rolling(window=k_period).min()
        data['H14'] = data['High'].rolling(window=k_period).max()
        data['%K'] = 100 * (data['Close'] - data['L14']) / (data['H14'] - data['L14'])
        data['%D'] = data['%K'].rolling(window=d_period).mean()
        data['Signal'] = 0
        data.loc[data['%K'] < oversold, 'Signal'] = 1  # Buy signal
        data.loc[data['%K'] > overbought, 'Signal'] = -1  # Sell signal
        return data

    @staticmethod
    def momentum_strategy(data, window=10):
        data['Momentum'] = data['Close'] - data['Close'].shift(window)
        data['Signal'] = 0
        data.loc[data['Momentum'] > 0, 'Signal'] = 1
        data.loc[data['Momentum'] <= 0, 'Signal'] = -1
        return data

    @staticmethod
    def moving_average_envelope(data, short_window=20, envelope_pct=0.02):
        data['SMA'] = data['Close'].rolling(window=short_window).mean()
        data['Upper Envelope'] = data['SMA'] * (1 + envelope_pct)
        data['Lower Envelope'] = data['SMA'] * (1 - envelope_pct)
        data['Signal'] = 0
        data.loc[data['Close'] > data['Upper Envelope'], 'Signal'] = -1  # Sell signal
        data.loc[data['Close'] < data['Lower Envelope'], 'Signal'] = 1  # Buy signal
        return data

    @staticmethod
    def average_true_range(data, atr_period=14):
        data['TR'] = np.maximum((data['High'] - data['Low']),
                                np.maximum(abs(data['High'] - data['Close'].shift(1)),
                                           abs(data['Low'] - data['Close'].shift(1))))
        data['ATR'] = data['TR'].rolling(window=atr_period).mean()
        data['Signal'] = 0
        data.loc[data['Close'] > data['Close'].shift(1) + data['ATR'], 'Signal'] = 1
        data.loc[data['Close'] < data['Close'].shift(1) - data['ATR'], 'Signal'] = -1
        return data

    @staticmethod
    def parabolic_sar(data, af=0.02, max_af=0.2):
        data['PSAR'] = data['Close']  # Placeholder for PSAR calculation
        data['Signal'] = 0
        # Implement PSAR logic here
        return data

    @staticmethod
    def ichimoku_cloud(data):
        data['Tenkan-sen'] = (data['High'].rolling(window=9).max() + data['Low'].rolling(window=9).min()) / 2
        data['Kijun-sen'] = (data['High'].rolling(window=26).max() + data['Low'].rolling(window=26).min()) / 2
        data['Signal'] = 0
        data.loc[data['Close'] > data['Kijun-sen'], 'Signal'] = 1
        data.loc[data['Close'] <= data['Kijun-sen'], 'Signal'] = -1
        return data

    @staticmethod
    def donchian_channel(data, window=20):
        data['Upper Channel'] = data['High'].rolling(window=window).max()
        data['Lower Channel'] = data['Low'].rolling(window=window).min()
        data['Signal'] = 0
        data.loc[data['Close'] > data['Upper Channel'], 'Signal'] = 1
        data.loc[data['Close'] < data['Lower Channel'], 'Signal'] = -1
        return data

# Backtesting function
def backtest(data, stop_loss, target, strategy_name):
    cash = CASH_AT_HAND
    position = 0
    entry_price = 0
    trades = []

    for i in range(len(data)):
        try:
            if data['Signal'].iloc[i] == 1 and position == 0:  # Buy signal
                trade_value = cash
                commission = trade_value * (COMMISSION_PERCENT / 100)  # Calculate commission
                position = (cash - commission) / data['Close'].iloc[i].item()
                entry_price = data['Close'].iloc[i].item()
                cash = 0
                trades.append(('Buy', data.index[i], entry_price, commission))
                print(f"Buy executed: Entry price = {entry_price}, Commission = {commission}")

            elif position > 0:
                trade_value = position * data['Close'].iloc[i].item()
                change = (data['Close'].iloc[i].item() - entry_price) / entry_price
                if change >= target or change <= -stop_loss:  # Target or Stop Loss hit
                    commission = trade_value * (COMMISSION_PERCENT / 100)  # Calculate commission
                    cash = trade_value - commission
                    position = 0
                    trades.append(('Sell', data.index[i], data['Close'].iloc[i].item(), commission))
                    print(f"Sell executed: Exit price = {data['Close'].iloc[i].item()}, Commission = {commission}")

        except Exception as e:
            print(f"Error during backtest loop: {e}")
            raise

    final_value = cash + (position * data['Close'].iloc[-1].item() if position > 0 else 0)
    profit_or_loss_percent = ((final_value - CASH_AT_HAND) / CASH_AT_HAND) * 100
    print(f"Backtest completed: Final value = {final_value}, Profit/Loss (%) = {profit_or_loss_percent}")
    return final_value, trades, profit_or_loss_percent, strategy_name

# Main function
def main():
    results = []
    seen_combinations = set()  # Track unique combinations of stock, strategy, stop loss, and target
    strategies = [
        ("Moving Average Crossover", Strategy.moving_average_crossover),
        ("RSI Strategy", Strategy.rsi_strategy),
        ("Bollinger Bands", Strategy.bollinger_bands),
        ("MACD Strategy", Strategy.macd_strategy),
        ("Stochastic Oscillator", Strategy.stochastic_oscillator),
        ("Momentum Strategy", Strategy.momentum_strategy),
        ("Moving Average Envelope", Strategy.moving_average_envelope),
        ("Average True Range", Strategy.average_true_range),
        ("Parabolic SAR", Strategy.parabolic_sar),
        ("Ichimoku Cloud", Strategy.ichimoku_cloud),
        ("Donchian Channel", Strategy.donchian_channel)
    ]

    for stop_loss, target, (strategy_name, strategy_function) in product(STOP_LOSS_RANGE, TARGET_RANGE, strategies):
        for stock in STOCK_LIST:
            stock_name = f"{stock}.NS"
            combination = (stock_name, strategy_name, stop_loss, target)
            if combination in seen_combinations:
                print(f"Skipping already processed combination: {combination}")
                continue
            seen_combinations.add(combination)

            try:
                data = yf.download(stock_name, start=START_DATE, end=END_DATE)
                data = data[~data.index.duplicated(keep='first')]  # Remove duplicate rows
                if data.empty or 'Close' not in data.columns:
                    print(f"Data for {stock_name} is empty or missing 'Close' column. Skipping.")
                    continue

                if data.isnull().any().any():
                    print(f"Data for {stock_name} contains NaN values. Filling missing values.")
                    data = data.fillna(method='ffill').fillna(method='bfill').dropna()

                strategy_data = strategy_function(data.copy())
                if 'Signal' not in strategy_data.columns or strategy_data['Signal'].isna().all():
                    print(f"No valid signals generated by {strategy_name} for {stock_name}. Skipping.")
                    continue

                final_value, trades, profit_or_loss_percent, strategy_name = backtest(
                    strategy_data, stop_loss, target, strategy_name
                )

                if not trades:
                    continue

                # Append the result
                results.append({
                    'Stock': stock_name,
                    'Strategy': strategy_name,
                    'Stop Loss': stop_loss,
                    'Target': target,
                    'Final Portfolio Value': final_value,
                    'Profit/Loss (%)': profit_or_loss_percent
                })

            except Exception as e:
                logging.error(f"Error processing {stock_name}: {e}")
                continue

    if not results:
        print("No backtest results to display. All stocks were skipped.")
        return

    # Convert results to DataFrame and drop duplicates
    results_df = pd.DataFrame(results)
    results_df = results_df.drop_duplicates(subset=['Stock', 'Strategy', 'Stop Loss', 'Target'])
    results_df = results_df.sort_values(by=['Stock', 'Strategy'])

    print("\nBacktest Results (Grouped by Stock and Strategy):")
    print(tabulate(results_df, headers='keys', tablefmt='grid'))

    results_df.to_csv("backtest_results.csv", index=False)

    print("\nSummary Report:")
    strategy_avg_returns = results_df.groupby('Strategy')['Profit/Loss (%)'].mean()
    best_strategy = strategy_avg_returns.idxmax()
    best_strategy_avg_return = strategy_avg_returns.max()

    stop_loss_target_avg_returns = results_df.groupby(['Stop Loss', 'Target'])['Profit/Loss (%)'].mean()
    best_stop_loss_target = stop_loss_target_avg_returns.idxmax()
    best_stop_loss_target_avg_return = stop_loss_target_avg_returns.max()

    print(f"\nBest Overall Strategy: {best_strategy} (Average Return: {best_strategy_avg_return:.2f}%)")
    print(f"Best Stop Loss and Target Combination: Stop Loss = {best_stop_loss_target[0]}, Target = {best_stop_loss_target[1]} (Average Return: {best_stop_loss_target_avg_return:.2f}%)")

if __name__ == "__main__":
    main()
