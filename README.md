# IndianStocksBacktest
TraderBot: Backtesting NIFTY 50 Strategies
TraderBot is a Python-based backtesting tool designed to evaluate various trading strategies on NIFTY 50 stocks. It uses historical stock data from Yahoo Finance to simulate trades and calculate performance metrics such as profit/loss percentages, best strategies, and optimal stop-loss/target combinations.

## Features
- Supports NIFTY 50 Stocks: Includes all NIFTY 50 stocks for backtesting.
- Multiple Trading Strategies:Moving Average Crossover, RSI Strategy, MACD Strategy You can add you own strategy
### Customizable Parameters:
- Stop-loss and target ranges
- Commission percentage
- Start and end dates for backtesting
Detailed Summary:
-  Best overall strategy
- Best stop-loss and target combination
- Highest return achieved with stock and strategy details
Output:
- Results displayed in a tabular format.
- Results saved to a CSV file (backtest_results.csv).
### Requirements
Python 3.8 or higher
#### Required Python libraries:
- yfinance
- pandas
- numpy
- tabulate

Usage
Open the script IndianStockBacktest.py and configure the following parameters:

- START_DATE: Start date for backtesting (e.g., '2021-01-01').
- END_DATE: End date for backtesting (e.g., '2025-04-18').
- CASH_AT_HAND: Initial cash for backtesting (e.g., 100000).
- STOP_LOSS_RANGE: List of stop-loss percentages (e.g., [0.02, 0.05, 0.1]).
- TARGET_RANGE: List of target percentages (e.g., [0.02, 0.05, 0.1, 0.15, 0.2]).
- COMMISSION_PERCENT: Commission percentage for trades (e.g., 0.2).
Run the script:

View the results:

The backtest results will be displayed in the terminal in a tabular format.
A summary report will show:
- Best overall strategy.
- Best stop-loss and target combination.
- Highest return achieved with stock and strategy details.
- Results will also be saved to backtest_results.csv.
Example Output

Stock	Strategy	Stop Loss	Target	Final Portfolio Value	Profit/Loss (%)
RELIANCE.NS	Moving Average Crossover	0.05	0.10	120000.00	20.00
TCS.NS	RSI Strategy	0.02	0.15	115000.00	15.00
INFY.NS	Bollinger Bands	0.10	0.20	130000.00	30.00
Summary Report
Best Overall Strategy: Bollinger Bands (Average Return: 25.00%)
Best Stop Loss and Target Combination: Stop Loss = 0.05, Target = 0.10 (Average Return: 20.00%)
Highest Return: 30.00% achieved by Bollinger Bands on INFY.NS
## Customization
Adding New Strategies
To add a new strategy:

Define a new static method in the Strategy class.
Implement the logic for generating buy/sell signals.
Add the strategy to the strategies list in the main() function.
Modifying Parameters
Update the STOP_LOSS_RANGE and TARGET_RANGE lists to test different stop-loss and target combinations.
Adjust the START_DATE and END_DATE to backtest over different time periods.
Limitations
The tool assumes perfect execution of trades at the closing price of the signal day.
Does not account for slippage or market impact.
Limited to historical data available on Yahoo Finance.
