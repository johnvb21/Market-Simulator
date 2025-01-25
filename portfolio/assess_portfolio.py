import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_data import *
pd.set_option('display.float_format', '{:.4f}'.format)  # Set pandas to display float values with 4 decimal places


def get_portfolio_value(prices, allocations, start_val=1000000):
    spy_df = prices[['SPY']].copy()  # Copy the SPY data for later use in comparisons
    prices.drop('SPY', axis=1, inplace=True)  # Drop the SPY column from the prices dataframe
    start_prices = prices.iloc[0]  # Get the starting prices for each symbol
    for column in prices.columns:  # Loop through each symbol in the dataframe
        prices[column] = prices[column] / start_prices[column]  # Normalize each symbol's price to the starting price
    prices = prices * allocations * start_val  # Multiply by allocations and starting value to get portfolio values
    prices['Portfolio Value'] = prices.sum(axis=1)  # Sum across the rows to get the total portfolio value

    return prices[['Portfolio Value']], spy_df  # Return the portfolio values and the SPY data

def get_portfolio_stats(port_val, prices_SPY, rfr=0.0, sf=252, calc_optional=False):
    cum_return = ((port_val.iloc[-1] - port_val.iloc[0]) / port_val.iloc[0]).loc['Portfolio Value']  # Calculate cumulative return
    avg_daily = port_val['Portfolio Value'].pct_change().mean()  # Calculate the average daily return
    volatility = port_val['Portfolio Value'].pct_change().std()  # Calculate the volatility (standard deviation of daily returns)
    sharpe_ratio = ((avg_daily - rfr) / volatility) * math.sqrt(sf)  # Calculate the Sharpe ratio
    start_value = port_val.iloc[0]['Portfolio Value']  # Get the starting portfolio value
    end_value = port_val.iloc[-1]['Portfolio Value']  # Get the ending portfolio value
    beta, alpha = np.polyfit(prices_SPY['SPY'].pct_change().dropna(), port_val['Portfolio Value'].pct_change().dropna(), 1)
    # Calculate beta and alpha using linear regression
    return cum_return, avg_daily, volatility, sharpe_ratio, beta, alpha, start_value, end_value  # Return all calculated statistics

def print_portfolio_stats(cum_return, avg_daily, volatility, sharpe_ratio, beta, alpha, start_value, end_value):
    # Print the portfolio statistics with appropriate formatting
    print(f"Cumulative Return: {cum_return: .4f}")
    print(f"Avg Daily Return: {avg_daily: .4f}")
    print(f"Volatility (Std Dev): {volatility: .4f}")
    print(f"Sharpe Ratio: {sharpe_ratio: .4f}")
    print(f"Beta: {beta: .4f}")
    print(f"Alpha: {alpha: .4f}")
    print(f"Start Portfolio Value: {start_value: .2f}")
    print(f"End Portfolio Value: {end_value: .2f}")

def assess_portfolio(start_date, end_date, syms, allocs, sv, rfr, sf, gen_plot=True, calc_optional=False):
    assessed_dates = pd.date_range(start_date, end_date)  # Generate a date range for the assessment period
    prices = get_data(syms, assessed_dates)  # Get the price data for the symbols over the assessment period

    port_value, spy_prices = get_portfolio_value(prices, allocs, sv)  # Calculate the portfolio value and get the SPY prices
    spy_prices_daily = spy_prices[['SPY']].pct_change().dropna()  # Calculate daily returns for SPY
    port_value_daily = port_value['Portfolio Value'].pct_change().dropna()  # Calculate daily returns for the portfolio
    correlation = spy_prices_daily['SPY'].corr(port_value_daily)  # Calculate the correlation between SPY and portfolio returns
    print("Correlation between SPY and Portfolio Value:")
    print(f"{correlation:.4f}")

    if gen_plot:  # If plotting is enabled
        # Normalize the prices to start at 1
        normed_port_value = port_value / port_value.iloc[0]
        normed_spy_prices = spy_prices / spy_prices.iloc[0]

        plt.plot(normed_port_value.index, normed_port_value, label='Portfolio')  # Plot normalized portfolio value
        plt.plot(normed_spy_prices.index, normed_spy_prices, label='SPY')  # Plot normalized SPY value
        plt.title('Portfolio vs SPY (Normalized)')  # Set the plot title
        plt.xlabel('Date')  # Set the x-axis label
        plt.ylabel('Normalized Value')  # Set the y-axis label
        plt.legend()  # Show the legend
        plt.show()  # Display the plot

    return correlation  # Return the correlation value
if __name__ == "__main__":
    # Example for testing the function
    dates = pd.date_range('2011-01-01', '2011-12-31')
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'SPY']  # Include SPY as a benchmark

    # Get the stock data for the given symbols
    df_prices = get_data(symbols, dates, path="../data")  # Modify the path to match your directory structure

    # Print first few rows to verify
    if not df_prices.empty:
        print(df_prices.head())
        plot_normalized_data(df_prices, title="Normalized prices", xlabel="Date", ylabel="Price")
    else:
        print("No data loaded.")
    allocations = [0.25, 0.25, 0.25, 0.25]
    port_val, spy_val = get_portfolio_value(df_prices, allocations, start_val=1000000)
    print(port_val.head())
    cum_return, avg_daily, volatility, sharpe_ratio, beta, alpha, start_value, end_value = get_portfolio_stats(port_val, spy_val, rfr=0.0, sf=252)
    print_portfolio_stats(cum_return, avg_daily, volatility, sharpe_ratio, beta, alpha, start_value, end_value)