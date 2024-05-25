# 2_download_data.py
import yfinance as yf
import os
import pandas as pd
import pandas_market_calendars as mcal

# Define the list of tickers and the data folder
stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'GS', 'C',
                    'IBM', 'INTC', 'AMD', 'QCOM', 'NFLX', 'DIS', 'SBUX', 'KO', 'PEP', 'MCD']
data_folder = 'data/2_raw/'

# Start and end date for data download, based on data ground truth
ground_truth_file = 'data/1_ground_truth_dates.csv'
ground_truth_df = pd.read_csv(ground_truth_file, parse_dates=['Date'])

start_date = ground_truth_df['Date'].min().strftime('%Y-%m-%d')
end_date = ground_truth_df['Date'].max().strftime('%Y-%m-%d')

print(f"Start date: {start_date}, End date: {end_date}")

# Create the folder if it doesn't exist
os.makedirs(data_folder, exist_ok=True)

# Get the NYSE calendar and valid trading days within the specified date range
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
valid_trading_days = schedule.index

# Function to check if the data is up-to-date
def is_data_up_to_date(file_path, valid_trading_days):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df_trading_days = df['Date'].isin(valid_trading_days)
    if not df_trading_days.any():
        return False
    file_start_date = df['Date'][df_trading_days].min()
    file_end_date = df['Date'][df_trading_days].max()
    print(f"Checking dates for {file_path}: file start date = {file_start_date}, file end date = {file_end_date}")
    # Check if the file covers the full date range within a tolerance for the last few trading days
    tolerance_days = 2
    return file_start_date <= valid_trading_days.min() and file_end_date >= valid_trading_days[-tolerance_days]

# Check and download data if necessary
for ticker in stock_tickers:
    file_path = os.path.join(data_folder, f'{ticker}.csv')
    if os.path.exists(file_path):
        if is_data_up_to_date(file_path, valid_trading_days):
            print(f"Data for {ticker} already exists and is up-to-date.")
            continue
        else:
            print(f"Updating data for {ticker}.")
    else:
        print(f"Downloading data for {ticker}.")

    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(file_path)

print("Data download complete.")
