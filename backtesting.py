import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import timedelta




def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-8)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

def delta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes delta.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-8)
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")




class BacktestDeltaNeutral:
    def __init__(self, ticker, start_date, end_date, option_type='call', strike_offset=0.0, risk_free_rate=0.01, hedge_threshold=0.1, transaction_cost=0.0):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.option_type = option_type
        self.strike_offset = strike_offset  # Offset from current price to determine strike price
        self.risk_free_rate = risk_free_rate
        self.hedge_threshold = hedge_threshold  # Threshold for re-hedging (e.g., when delta deviates by 0.1)
        self.transaction_cost = transaction_cost  # Cost per trade (as a percentage of trade value)
        self.data = None
        self.results = None

    def get_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.data['Returns'] = self.data['Adj Close'].pct_change()
        self.data['LogReturns'] = np.log(self.data['Adj Close'] / self.data['Adj Close'].shift(1))
        self.data.dropna(inplace=True)

    def estimate_volatility(self, window=20):
        # Use rolling standard deviation of log returns
        self.data['Volatility'] = self.data['LogReturns'].rolling(window).std() * np.sqrt(252)
        self.data['Volatility'].fillna(method='bfill', inplace=True)

    def run_backtest(self):
        # Initialize variables
        cash = 0.0
        stock_position = 0.0
        option_position = -1.0  # Sell one option contract
        portfolio_value = []
        dates = []
        deltas = []
        stock_positions = []
        option_positions = []
        cash_history = []
        
        # Initial date
        initial_date = self.data.index[0]
        expiry_date = initial_date + timedelta(days=30)  # Option expires in 30 days

        for idx in range(len(self.data)):
            date = self.data.index[idx]
            S = self.data['Adj Close'].iloc[idx]
            sigma = self.data['Volatility'].iloc[idx]
            T = (expiry_date - date).days / 252  # Time to maturity
            if T <= 0:
                # Option has expired, sell a new option
                expiry_date = date + timedelta(days=30)
                T = (expiry_date - date).days / 252
                K = S * (1 + self.strike_offset)
                # Calculate option price and delta for the new option
                opt_price = black_scholes(S, K, T, self.risk_free_rate, sigma, self.option_type)
                opt_delta = delta(S, K, T, self.risk_free_rate, sigma, self.option_type)
                # Update cash position from selling new option
                cash += opt_price * option_position  # Selling option generates cash
            else:
                K = S * (1 + self.strike_offset)

            # Calculate option price and delta
            opt_price = black_scholes(S, K, T, self.risk_free_rate, sigma, self.option_type)
            opt_delta = delta(S, K, T, self.risk_free_rate, sigma, self.option_type)

            # Calculate total delta
            total_delta = stock_position * 1 + option_position * opt_delta

            # Re-hedge if delta deviates beyond threshold
            if abs(total_delta) > self.hedge_threshold:
                desired_stock_position = -option_position * opt_delta
                delta_stock_trade = desired_stock_position - stock_position

                # Update cash and positions with transaction costs
                trade_cost = delta_stock_trade * S * self.transaction_cost
                cash -= delta_stock_trade * S + trade_cost  # Buy/sell stock
                stock_position += delta_stock_trade  # Update stock position

            # Record keeping
            portfolio_val = cash + stock_position * S + option_position * opt_price
            portfolio_value.append(portfolio_val)
            dates.append(date)
            deltas.append(total_delta)
            stock_positions.append(stock_position)
            option_positions.append(option_position)
            cash_history.append(cash)

        # Save results
        self.results = pd.DataFrame({
            'Date': dates,
            'PortfolioValue': portfolio_value,
            'Delta': deltas,
            'StockPosition': stock_positions,
            'OptionPosition': option_positions,
            'Cash': cash_history
        })
        self.results.set_index('Date', inplace=True)

    def plot_results(self):
        if self.results is None:
            print("Run the backtest first.")
            return
        plt.figure(figsize=(12, 6))
        plt.plot(self.results['PortfolioValue'], label='Portfolio Value')
        plt.title(f'Delta-Neutral Strategy Backtest for {self.ticker}')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.show()
