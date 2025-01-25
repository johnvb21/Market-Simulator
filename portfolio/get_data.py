import os
import pandas as pd


def get_data(symbols, dates, path="data"):
    """Read stock data (adjusted close) for given symbols from CSV files in a 'data' directory."""

    # Check if the path to the data directory exists
    if not os.path.exists(path):
        print(f"Error: The data directory '{path}' does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if the path does not exist

    # Initialize an empty DataFrame to hold the stock prices for all symbols
    df_final = pd.DataFrame(index=dates)

    for symbol in symbols:
        # Construct the file path for the stock symbol CSV
        file_path = os.path.join(path, f"{symbol}.csv")

        # Print the file path to ensure it's correct
        print(f"Looking for file: {file_path}")

        # Check if the file for the symbol exists
        if not os.path.exists(file_path):
            print(f"Warning: The file for symbol '{symbol}' does not exist at '{file_path}'. Skipping...")
            continue  # Skip to the next symbol if the file is missing

        try:
            # Read the CSV file for the given symbol
            df_temp = pd.read_csv(file_path,
                                  index_col='Date',
                                  parse_dates=True,
                                  usecols=['Date', 'Adj Close'],
                                  na_values='NaN')

            # Rename the 'Adj Close' column to the stock symbol for easier access later
            df_temp = df_temp.rename(columns={'Adj Close': symbol})

            # Join the stock data with the main DataFrame (df_final)
            df_final = df_final.join(df_temp, how='inner')  # Only join the dates that exist in both
            print(f"Loaded data for {symbol} from {file_path}")
        except Exception as e:
            print(f"Error: Could not load data for symbol '{symbol}' from {file_path}. Reason: {e}")

    if df_final.empty:
        print("Warning: No valid data was loaded. Please check the file paths and symbols.")
    else:
        print(f"Successfully loaded data for symbols: {', '.join(df_final.columns)}")

    return df_final


def add_symbol_to_dataframe(df, symbol, dates, path="data"):
    """Add a symbol (like SPY) to an existing DataFrame."""

    # Construct the file path for the stock symbol CSV
    file_path = os.path.join(path, f"{symbol}.csv")

    print(f"Looking for file: {file_path}")

    try:
        # Read the CSV file for the given symbol
        df_temp = pd.read_csv(file_path,
                              index_col='Date',
                              parse_dates=True,
                              usecols=['Date', 'Adj Close'],
                              na_values='NaN')

        # Rename the 'Adj Close' column to the stock symbol
        df_temp = df_temp.rename(columns={'Adj Close': symbol})

        # Join the symbol's data to the existing DataFrame
        df = df.join(df_temp, how='inner')  # Inner join to ensure dates align

        print(f"Added {symbol} to the DataFrame.")
    except FileNotFoundError:
        print(f"Warning: Data for symbol {symbol} not found at {file_path}.")
    except Exception as e:
        print(f"Error: Failed to add {symbol}. Reason: {e}")

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    import matplotlib.pyplot as plt
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.grid(color='gray', linestyle='-', linewidth=0.2)

    plt.show()


def plot_normalized_data(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison.
    Parameters
    ----------
        df: DataFrame containing stock prices to plot (non-normalized)
        title: plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        """
    df_normed = df / df.iloc[0, :]
    plot_data(df_normed, title=title, xlabel=xlabel, ylabel=ylabel)


# Example usage inside your main script
