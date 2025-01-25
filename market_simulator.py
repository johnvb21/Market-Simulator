import pandas as pd
from datetime import datetime
import argparse
import importlib
import sys
import re

# Insert 'portfolio' directory to the Python path to allow importing custom modules
sys.path.insert(0, 'portfolio')
from portfolio.get_data import get_data, plot_data  # Import data fetching and plotting functions from portfolio module
from portfolio.assess_portfolio import get_portfolio_value, get_portfolio_stats, \
    plot_normalized_data  # Import portfolio assessment functions


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, args=None):
    """
    Placeholder for compute_portvals. This function should compute daily portfolio value
    given a sequence of orders in a CSV file. Currently, it just returns a placeholder value.

    Parameters
    ----------
    orders_file : str
        CSV file to read orders from.
    start_val : float
        Starting cash value.

    Returns
    -------
    portvals : pd.Series
        Placeholder for daily portfolio value for each trading day.
    """

    # Reading the order file containing trades
    commission = args.transaction_fee if args else 9.95  # Transaction fee for each trade
    market_impact_factor = args.market_impact_factor if args else 0.005  # Market impact as a factor of price
    verbose = args.do_verbose if args else False  # Verbose mode for debugging or detailed output

    # Load orders data from CSV, sorting by date
    orders = pd.read_csv(orders_file, index_col=0)
    orders.sort_index(inplace=True)

    # Get the date range from the order file
    first_date = orders.index[0]
    last_date = orders.index[-1]
    dates = pd.date_range(first_date, last_date)

    print('ORDERS:')
    print(orders.head())  # Print first few rows of the order data

    # Extract unique symbols from the orders and include SPY for reference
    symbols = list(set(orders['Symbol'].tolist()))
    symbols.append('SPY')  # Add SPY as a benchmark

    # Fetch price data for all relevant symbols over the date range
    price_df = get_data(symbols, dates)

    # Add a 'Cash' column to track cash movements separately
    price_df['Cash'] = 1.0
    spy_df = price_df[['SPY']].copy()  # Keep SPY data separately for later analysis
    price_df.drop('SPY', axis=1, inplace=True)  # Drop SPY from the price dataframe

    print("RELEVANT PRICES:")
    print(price_df)  # Print the price data for the symbols

    # Remove SPY from the symbols list since it's not part of the portfolio
    symbols.remove('SPY')

    # Initialize a dataframe to track changes in the portfolio (shares bought/sold) and cash
    changes = pd.DataFrame(0, columns=symbols, index=dates)
    changes['Cash'] = 0.00  # Set cash column to zero initially

    # Iterate through each order and apply changes to the 'changes' dataframe
    for index, row in orders.iterrows():
        date_temp = row.name  # Order date
        symbol_temp = row['Symbol']  # Stock symbol
        shares_temp = row['Shares']  # Number of shares
        order_type_temp = row['Order']  # BUY or SELL order

        # Process a SELL order
        if order_type_temp == 'SELL':
            shares_temp = shares_temp * -1  # Negative value for sold shares
            order_price = price_df.loc[date_temp, symbol_temp]
            order_price = (order_price * shares_temp * -1) * (
                        1 - market_impact_factor)  # Adjust price for market impact
            changes.loc[date_temp, symbol_temp] += shares_temp  # Subtract shares
            changes.loc[date_temp, 'Cash'] += order_price - commission  # Add cash to account, deduct commission

        # Process a BUY order
        elif order_type_temp == 'BUY':
            order_price = price_df.loc[date_temp, symbol_temp]
            order_price = (order_price * shares_temp) * (1 + market_impact_factor)  # Adjust price for market impact
            order_price = order_price * -1  # Subtract from cash (buying cost)
            changes.loc[date_temp, symbol_temp] += shares_temp  # Add shares to holdings
            changes.loc[date_temp, 'Cash'] += order_price - commission  # Deduct cash and transaction fee

    print("PORTFOLIO CHANGES")
    print(changes)  # Show changes in portfolio after orders

    # Calculate cumulative portfolio values over time
    cumulative_port = changes.cumsum()

    print("DAILY HOLDINGS:")
    print(cumulative_port)  # Show cumulative holdings over time

    # Initialize the portfolio value series with zero values
    portvals = pd.Series(0, index=dates, name='Portfolio Value')  # Placeholder portfolio values

    # Multiply cumulative holdings by the price of corresponding assets and compute total portfolio value
    mult_df = cumulative_port * price_df
    mult_df = mult_df.ffill()  # Forward fill missing prices
    mult_df['Cash'] = mult_df['Cash'] + start_val  # Add initial cash to the cash column

    print("ASSETS")
    print(mult_df)  # Show portfolio asset values over time

    # Calculate the final portfolio values by summing across all asset columns
    portvals = portvals + mult_df.sum(axis=1)

    print(portvals.reindex(index=price_df.index))  # Show portfolio value per day
    portvals = portvals.reindex(index=price_df.index)  # Reindex to match price data dates

    if verbose:
        print("\nFinal Portfolio Value Per Day (Stub Output):\n")
        print(portvals.tail())  # Output the last few days' portfolio values

    # Plot the portfolio value compared to SPY for reference
    portval_df = portvals.to_frame(name='Portfolio Value')
    concat_df = pd.concat([portval_df, spy_df], axis=1)
    print(concat_df)  # Show portfolio value and SPY values together
    plot_normalized_data(concat_df)  # Plot the data for comparison

    return portvals, spy_df  # Return portfolio values and SPY data


def test_code(args=None):
    # Set default arguments if none are provided
    if args is None:
        args = argparse.Namespace(
            orderfile='./orders/orders0.csv',  # Order file to use
            start_value=1000000,  # Initial cash value
            transaction_fee=9.95,  # Default transaction fee
            market_impact_factor=0.005,  # Default market impact
            do_verbose=True,  # Verbose mode enabled
            test_student=0,
            do_plot=True  # Enable plotting
        )

    print("Running Market Simulator ...\n")
    print(f"Orders file: {args.orderfile}")  # Show which orders file is being used

    # Compute portfolio values based on the orders file and other parameters
    portvals, spy_df = compute_portvals(orders_file=args.orderfile, start_val=args.start_value, args=args)

    # Get statistics about the portfolio's performance
    start_date = portvals.index.min()
    end_date = portvals.index.max()
    portval_df = portvals.to_frame(name='Portfolio Value')
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, beta, alpha, start_value, end_value = get_portfolio_stats(
        portval_df, spy_df)
    spy_df_temp = spy_df.rename(columns={spy_df.columns[0]: 'Portfolio Value'},
                                inplace=False)  # Rename SPY column for comparison
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY, beta_, alpha_, start_value_spy, end_value_spy = get_portfolio_stats(
        spy_df_temp, spy_df)

    # Print Portfolio vs SPY statistics
    print()
    print("--- begin statistics ------------------------------------------------- ")
    print(f"Date Range: {start_date.date()} to {end_date.date()} (portfolio)")

    print(f"Number of Trading Days:             {len(portvals):14d}")  # Total number of trading days
    print(f"Sharpe Ratio of Fund:               {sharpe_ratio:+.11f}")  # Sharpe ratio of the portfolio
    print(f"Sharpe Ratio of SPY:                {sharpe_ratio_SPY:+.11f}")  # Sharpe ratio of SPY

    print(f"Cumulative Return of Fund:          {cum_ret:+.11f}")  # Portfolio cumulative return
    print(f"Cumulative Return of SPY:           {cum_ret_SPY:+.11f}")  # SPY cumulative return

    print(f"Standard Deviation of Fund:         {std_daily_ret:+.11f}")  # Portfolio standard deviation
    print(f"Standard Deviation of SPY:          {std_daily_ret_SPY:+.11f}")  # SPY standard deviation

    print(f"Average Daily Return of Fund:       {avg_daily_ret:+.11f}")  # Portfolio average daily return
    print(f"Average Daily Return of SPY:        {avg_daily_ret_SPY:+.11f}")  # SPY average daily return

    final_portval = portvals.iloc[-1]  # Final portfolio value
    print(f"\nFinal Portfolio Value:        {final_portval:+,.11f}")  # Print final portfolio value

    if args.do_plot:
        # Stub for plot logic
        print("\nPlotting is enabled (Stub)...")  # Indicate that plotting is enabled


if __name__ == "__main__":
    # Command-line argument parser setup
    parser = argparse.ArgumentParser(description="Market Simulator")
    parser.add_argument('-f', '--file', type=str, dest="orderfile", default='./orders/covid19orders.csv',
                        help='Path to the order file')  # Orders file path argument
    parser.add_argument('-c', '--cash', type=float, dest="start_value", default=1000000,
                        help='Starting cash')  # Starting cash
    parser.add_argument('-t', '--transaction_fee', type=float, dest="transaction_fee", default=9.95,
                        help='Transaction fee')  # Transaction fee argument
    parser.add_argument('-m', '--market_impact', type=float, dest="market_impact_factor", default=0.005,
                        help='Market impact factor')  # Market impact factor argument
    parser.add_argument('-v', '--verbose', action='store_true', dest="do_verbose",
                        help='Verbose mode')  # Verbose mode argument
    parser.add_argument('-p', '--plot', action='store_true', dest="do_plot",
                        help='Plot results')  # Plot results argument
    args = parser.parse_args()  # Parse command-line arguments

    # List of order files to test
    order_files = [
        './orders/orders-short.csv',
        './orders/orders.csv',
        './orders/orders1.csv',
        './orders/orders2.csv',
        './orders/orders3.csv',
        './orders/covid19orders.csv',
        './orders/post-covid-order.csv',
    ]

    # Loop over each order file and run the simulation
    for order_file in order_files:
        print(f"Running test for {order_file}")  # Print current order file being tested
        args.orderfile = order_file  # Update the orderfile argument for each run
        test_code(args)  # Call the function with the updated args
