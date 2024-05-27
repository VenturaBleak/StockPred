# 2_download_data.py
import os
import pandas as pd
import requests
import yfinance as yf
import io

# Define the list of tickers and the data folder
stock_tickers = ['IBM']
data_folder = 'data/2_raw/'
metadata_folder = 'data/2_metadata/'

# Create the folders if they don't exist
os.makedirs(data_folder, exist_ok=True)
os.makedirs(metadata_folder, exist_ok=True)

# Get the Alpha Vantage API key from environment variables
api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

class APIClient:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def download_data(self, symbol):
        raise NotImplementedError("This method should be implemented by subclasses.")

class YFinanceClient(APIClient):
    def download_data(self, symbol):
        data = yf.download(symbol, start="2000-01-01")
        data.reset_index(inplace=True)
        return data

class AlphaVantageClient(APIClient):
    def download_data(self, symbol, function):
        base_url = 'https://www.alphavantage.co/query'
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'datatype': 'json'
        }
        response = requests.get(base_url, params=params)

        # Print response content for debugging
        print(f"Response content for {symbol} with function {function}:")
        print(response.text)

        # Check if response is successful and parse JSON
        if response.status_code == 200:
            data = response.json()
            if "Error Message" in data or "Note" in data:
                raise ValueError(f"Error retrieving data for {symbol}: {data}")

            if function == 'OVERVIEW':
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame(data['annualReports'])
            return df
        else:
            raise ValueError(f"Error retrieving data: {response.content}")

class APIManager:
    def __init__(self, yfinance_client, alpha_vantage_client):
        self.yfinance_client = yfinance_client
        self.alpha_vantage_client = alpha_vantage_client

    def download_yfinance_data(self, symbol):
        return self.yfinance_client.download_data(symbol)

    def download_alpha_vantage_data(self, symbol, function):
        return self.alpha_vantage_client.download_data(symbol, function)

    def generate_metadata(self, df, source, data_type):
        metadata = {
            "feature_name": df.columns.tolist(),
            "data_type": [str(dtype) for dtype in df.dtypes],
            "source": [source] * len(df.columns),
            "data_type_category": [data_type] * len(df.columns),
            "description": [""] * len(df.columns)  # Placeholder for descriptions
        }
        return metadata

    def save_metadata(self, metadata, file_name):
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(metadata_folder, file_name), index=False)

    def save_data(self, df, file_path):
        df.to_csv(file_path, index=False)

# Initialize API clients and manager
yfinance_client = YFinanceClient()
alpha_vantage_client = AlphaVantageClient(api_key=api_key)
api_manager = APIManager(yfinance_client, alpha_vantage_client)

# Download and save the data for each ticker
for ticker in stock_tickers:
    # Download daily adjusted data using yfinance
    file_path = os.path.join(data_folder, f'{ticker}_daily_adjusted.csv')
    if not os.path.exists(file_path):
        print(f"Downloading daily adjusted data for {ticker} from yfinance...")
        df = api_manager.download_yfinance_data(ticker)
        api_manager.save_data(df, file_path)
        print(f"Daily adjusted data for {ticker} saved.")

        # Generate and save metadata
        metadata = api_manager.generate_metadata(df, source='yfinance', data_type='daily_adjusted')
        api_manager.save_metadata(metadata, 'yfinance_daily_adjusted_metadata.csv')
        print(f"Metadata for {ticker} daily adjusted data saved.")
    else:
        print(f"Daily adjusted data for {ticker} already exists.")

    # Download other data using Alpha Vantage
    functions = ['OVERVIEW', 'INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW', 'EARNINGS']
    for function in functions:
        file_path = os.path.join(data_folder, f'{ticker}_{function.lower()}.csv')
        if not os.path.exists(file_path):
            print(f"Downloading {function.lower()} data for {ticker} from Alpha Vantage...")
            try:
                df = api_manager.download_alpha_vantage_data(ticker, function=function)
                api_manager.save_data(df, file_path)
                print(f"{function.lower()} data for {ticker} saved.")

                # Generate and save metadata
                metadata = api_manager.generate_metadata(df, source='Alpha Vantage', data_type=function.lower())
                api_manager.save_metadata(metadata, f'alpha_vantage_{function.lower()}_metadata.csv')
                print(f"Metadata for {ticker} {function.lower()} data saved.")
            except ValueError as e:
                print(f"Error downloading {function.lower()} data for {ticker}: {e}")
        else:
            print(f"{function.lower()} data for {ticker} already exists.")

print("Data download and metadata creation complete.")
