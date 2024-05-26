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
for ticker in ['IBM']:
    # Initialize an empty DataFrame for the combined data
    combined_df = pd.DataFrame()

    # Process each raw data file for the ticker
    for file in glob(os.path.join(raw_data_folder, f'{ticker}_*.csv')):
        print(f"Preprocessing {file}...")
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').reindex(ground_truth_dates['Date']).reset_index()
        df['Ticker'] = ticker

        # Drop rows where 'Adj Close' is missing for daily adjusted data
        if 'Adj Close' in df.columns:
            df = df.dropna(subset=['Adj Close'])

        # Calculate daily returns for daily adjusted data
        if 'Adj Close' in df.columns:
            df['Daily_Return'] = df['Adj Close'].pct_change()

        # Merge with ground truth dates to include holiday and time information
        df = pd.merge(df, ground_truth_dates, on='Date', how='left')

        # Add previous day's time features
        for col in ground_truth_dates.columns:
            if col != 'Date':
                df[f'Prev_{col}'] = df[col].shift(1)

        # Append the preprocessed DataFrame to the combined DataFrame
        combined_df = pd.concat([combined_df, df], axis=1)

    # Drop duplicate columns (keep only the first occurrence)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # Ensure consistent output format
    time_features = [col for col in ground_truth_dates.columns if col != 'Date']
    time_features_prev = [f'Prev_{col}' for col in time_features]
    combined_df = combined_df[['Date', 'Ticker'] + time_features + time_features_prev +
                              [col for col in combined_df.columns if
                               col not in ['Date', 'Ticker'] + time_features + time_features_prev]]

    # Save preprocessed file
    combined_df.to_csv(os.path.join(preprocessed_data_folder, f'{ticker}.csv'), index=False)

print("Preprocessing complete.")
