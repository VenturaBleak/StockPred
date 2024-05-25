# 4_feature_engineering.py
import pandas as pd
import numpy as np
import os
from glob import glob

# Define folders
preprocessed_data_folder = 'data/3_preprocessed'
features_data_folder = 'data/4_features'
os.makedirs(features_data_folder, exist_ok=True)

# Specify features for lagged calculations
percentage_features = ['Daily_Return']
absolute_features = ['Volume']
lag_periods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
aggregation_periods = [5, 10, 20, 30, 60, 200, 500]
annualization_factor = np.sqrt(252)

# Function to calculate simple lagged features
def calculate_simple_lagged_features(df, features, lags):
    lagged_features = pd.DataFrame(index=df.index)
    for feature in features:
        for lag in lags:
            lagged_features[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    return lagged_features

# Function to calculate aggregated features
def calculate_aggregated_features(df, features, periods):
    aggregated_features = pd.DataFrame(index=df.index)
    for feature in features:
        for period in periods:
            if feature == 'Daily_Return':
                aggregated_features[f'{feature}_mean_{period}d'] = (df['Adj Close'].shift(1) / df['Adj Close'].shift(
                    period + 1)) ** (1 / period) - 1
                aggregated_features[f'{feature}_std_{period}d'] = df['Daily_Return'].rolling(window=period).std().shift(
                    1) * annualization_factor
            else:
                aggregated_features[f'{feature}_mean_{period}d'] = df[feature].rolling(window=period).mean().shift(1)
                aggregated_features[f'{feature}_std_{period}d'] = df[feature].rolling(window=period).std().shift(1)
    return aggregated_features

# Function to calculate additional features from OHLC data
def calculate_ohlc_features(df):
    ohlc_features = pd.DataFrame(index=df.index)
    ohlc_features['Return_Open_High'] = df['High'] / df['Open'] - 1
    ohlc_features['Return_Open_Low'] = df['Low'] / df['Open'] - 1
    ohlc_features['Return_High_Close'] = df['Close'] / df['High'] - 1
    ohlc_features['Return_Low_Close'] = df['Close'] / df['Low'] - 1
    ohlc_features['Range_High_Low'] = (df['High'] - df['Low']) / df['Close']
    for feature in ohlc_features.columns:
        ohlc_features[f'{feature}_lag1'] = ohlc_features[feature].shift(1)
    return ohlc_features

# Function to calculate high/low indicators within given aggregation periods
def calculate_high_low_indicators(df, periods):
    high_low_indicators = pd.DataFrame(index=df.index)
    for period in periods:
        high_low_indicators[f'Max_High_{period}d'] = df['High'].rolling(window=period).max().shift(1)
        high_low_indicators[f'Min_Low_{period}d'] = df['Low'].rolling(window=period).min().shift(1)
    return high_low_indicators

# Process each file
for file in glob(os.path.join(preprocessed_data_folder, '*.csv')):
    df = pd.read_csv(file)
    ticker = os.path.basename(file).split('.')[0]

    # Drop rows with missing target values
    df = df.dropna(subset=['Daily_Return'])

    # Calculate additional features from OHLC data
    ohlc_features = calculate_ohlc_features(df)

    # Calculate high/low indicators within given aggregation periods
    high_low_indicators = calculate_high_low_indicators(df, aggregation_periods)

    # Calculate simple lagged features for OHLC and volume
    lagged_features = calculate_simple_lagged_features(df, ['Open', 'High', 'Low', 'Close'], [1])

    # Calculate simple lagged features for percentage and absolute features
    lagged_percentage_absolute_features = calculate_simple_lagged_features(df, percentage_features + absolute_features, lag_periods)

    # Calculate aggregated features
    aggregated_percentage_features = calculate_aggregated_features(df, percentage_features, aggregation_periods)
    aggregated_absolute_features = calculate_aggregated_features(df, absolute_features, aggregation_periods)

    # Define time features, independent of capitalization
    time_feature_cols = [col for col in df.columns if any(term in col.lower() for term in ['numbered', 'time', 'day', 'month', 'year', 'week', 'quarter'])]
    time_features = df[time_feature_cols]

    # Combine all features into one DataFrame
    features_df = pd.concat([df[['Date', 'Ticker', 'Daily_Return']], time_features,
                             ohlc_features, high_low_indicators, lagged_features,
                             lagged_percentage_absolute_features, aggregated_percentage_features,
                             aggregated_absolute_features], axis=1)

    # Remove unnecessary columns
    features_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Active',
                              'Return_Open_High', 'Return_Open_Low', 'Return_High_Close',
                              'Return_Low_Close', 'Range_High_Low'], inplace=True, errors='ignore')

    # Indicate whether any row has features with no values
    feature_cols = [col for col in features_df.columns if any(term in col for term in ['lag', 'mean', 'std'])]

    # Drop rows with missing values in the feature columns, if more than 2 missing features
    features_df = features_df[features_df.isnull().sum(axis=1) <= 2]

    # Check for 'inf' values in the numeric columns and raise exception if found
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    if np.isinf(features_df[numeric_cols]).sum().sum() > 0:
        raise ValueError(f"Invalid 'inf' values found in {ticker} data after feature engineering")

    # Check for 'NA' values in the generated features and fill with dummy value: -999
    features_df.fillna(-999, inplace=True)

    # Save the cleaned DataFrame
    features_df.to_csv(os.path.join(features_data_folder, f'{ticker}.csv'), index=False)

print("Feature engineering complete.")