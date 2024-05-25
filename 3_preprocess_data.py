# 3_preprocess_data.py
import pandas as pd
import os
from glob import glob

# Define folders
raw_data_folder = 'data/2_raw'
preprocessed_data_folder = 'data/3_preprocessed'
os.makedirs(preprocessed_data_folder, exist_ok=True)

# Load ground truth dates
ground_truth_dates = pd.read_csv('data/1_ground_truth_dates.csv')
ground_truth_dates['Date'] = pd.to_datetime(ground_truth_dates['Date'])

# Preprocess each file
for file in glob(os.path.join(raw_data_folder, '*.csv')):
    ticker = os.path.basename(file).split('.')[0]
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').reindex(ground_truth_dates['Date']).reset_index()
    df.rename(columns={'index': 'Date'}, inplace=True)
    df['Ticker'] = ticker
    df['Active'] = df['Close'].notna().astype(int)

    # Drop rows where 'Adj Close' is missing
    df = df.dropna(subset=['Adj Close'])

    # Calculate daily returns
    df['Daily_Return'] = df['Adj Close'].pct_change()

    # Merge with ground truth dates to include holiday and time information
    df = pd.merge(df, ground_truth_dates, on='Date', how='left')

    # Add previous day's time features
    for col in ground_truth_dates.columns:
        if col != 'Date':
            df[f'Prev_{col}'] = df[col].shift(1)

    # Sort columns as specified
    time_features = [col for col in ground_truth_dates.columns if col != 'Date']
    time_features_prev = [f'Prev_{col}' for col in time_features]
    df = df[['Date', 'Ticker', 'Daily_Return'] + time_features + time_features_prev + [col for col in df.columns if col not in ['Date', 'Ticker', 'Daily_Return'] + time_features + time_features_prev]]

    # Save preprocessed file
    df.to_csv(os.path.join(preprocessed_data_folder, f'{ticker}.csv'), index=False)

print("Preprocessing complete.")