import os
import pandas as pd
import numpy as np
from glob import glob

# Define folders
preprocessed_data_folder = os.path.join('data', '3_preprocessed')
preprocessed_dummy_data_folder = os.path.join('data', '3_preprocessed_dummy')
features_data_folder = os.path.join('data', '4_features')
features_dummy_data_folder = os.path.join('data', '4_features_dummy')
os.makedirs(features_data_folder, exist_ok=True)
os.makedirs(features_dummy_data_folder, exist_ok=True)

# Specify features for lagged calculations
real_absolute_features = []
dummy_absolute_features = ['Target', 'Feature1', 'Feature2', 'Feature3', 'Feature4']

lag_periods = [1, 2, 3, 4, 5]
aggregation_periods = [5, 10]
annualization_factor = np.sqrt(252)

# Function to calculate simple lagged features
def calculate_simple_lagged_features(df, features, lags):
    lagged_features = pd.DataFrame(index=df.index)
    for feature in features:
        for lag in lags:
            lagged_features[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    return lagged_features

# Function to calculate aggregated features
def calculate_aggregated_features(df, features, periods, percentage=False):
    aggregated_features = pd.DataFrame(index=df.index)
    for feature in features:
        for period in periods:
            if percentage:
                aggregated_features[f'{feature}_mean_{period}d'] = (df[feature].shift(1) / df[feature].shift(
                    period + 1)) ** (1 / period) - 1
                aggregated_features[f'{feature}_std_{period}d'] = df[feature].rolling(window=period).std().shift(
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

# Function to calculate look-ahead targets for multiple target columns
def calculate_look_ahead_targets(df, periods, target_columns, use_dummy_data):
    look_ahead_targets = pd.DataFrame(index=df.index)
    for target_column in target_columns:
        for period in periods:
            if use_dummy_data and period == 0:
                look_ahead_targets[f'{target_column}_T{period}'] = df[target_column]
            else:
                look_ahead_targets[f'{target_column}_T{period}'] = df[target_column].shift(-period)
    return look_ahead_targets

def main(use_dummy_data=False, look_ahead_periods=range(0, 21)):
    # Determine the folder to use for preprocessed data
    preprocessed_folder = preprocessed_dummy_data_folder if use_dummy_data else preprocessed_data_folder
    features_folder = features_dummy_data_folder if use_dummy_data else features_data_folder

    if use_dummy_data:
        print("Using dummy data for feature engineering...")

        # delete files in the folder
        for file in os.listdir(features_folder):
            os.remove(os.path.join(features_folder, file))

    # Determine the features to use based on the data type
    absolute_features = dummy_absolute_features if use_dummy_data else real_absolute_features

    # Process each file
    for file in glob(os.path.join(preprocessed_folder, '*.csv')):
        df = pd.read_csv(file)
        ticker = os.path.basename(file).split('.')[0]

        # Identify target columns
        target_columns = [col for col in df.columns if 'Target' in col]

        # Drop rows with missing target values
        df = df.dropna(subset=target_columns)

        # Calculate OHLC features only for real data
        ohlc_features = calculate_ohlc_features(df) if not use_dummy_data else pd.DataFrame(index=df.index)

        # Calculate high/low indicators only for real data
        high_low_indicators = calculate_high_low_indicators(df, aggregation_periods) if not use_dummy_data else pd.DataFrame(index=df.index)

        # Calculate simple lagged features for absolute features
        lagged_features = calculate_simple_lagged_features(df, absolute_features, lag_periods)

        # Calculate aggregated features for absolute features
        aggregated_absolute_features = calculate_aggregated_features(df, absolute_features, aggregation_periods, percentage=False)

        # Define time features, independent of capitalization
        time_feature_cols = [col for col in df.columns if any(term in col.lower() for term in ['numbered', 'time', 'day', 'month', 'year', 'week', 'quarter'])]
        time_features = df[time_feature_cols]

        # Calculate look-ahead targets for all target columns
        look_ahead_targets = calculate_look_ahead_targets(df, look_ahead_periods, target_columns, use_dummy_data)

        # Combine all features into one DataFrame
        features_df = pd.concat([df[['Date', 'Ticker']], time_features,
                                 ohlc_features, high_low_indicators,
                                 lagged_features, aggregated_absolute_features,
                                 look_ahead_targets, df[['Feature1', 'Feature2', 'Feature3', 'Feature4']]], axis=1)

        # Drop rows with missing values in the target columns
        target_columns_lagged = [f'{col}_T{period}' for col in target_columns for period in look_ahead_periods]
        features_df.dropna(subset=target_columns_lagged, inplace=True)

        # Remove unnecessary columns and original target columns
        features_df.drop(columns=target_columns, inplace=True, errors='ignore')

        # Indicate whether any row has features with no values
        feature_cols = [col for col in features_df.columns if any(term in col for term in ['lag', 'mean', 'std', 'T'])]

        # Drop rows with missing values in the feature columns, if more than 2 missing features
        features_df = features_df[features_df.isnull().sum(axis=1) <= 2]

        # Check for 'inf' values in the numeric columns and raise exception if found
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if np.isinf(features_df[numeric_cols]).sum().sum() > 0:
            raise ValueError(f"Invalid 'inf' values found in {ticker} data after feature engineering")

        # Check for 'NA' values in the generated features and fill with dummy value: -999
        features_df.fillna(-999, inplace=True)


        # Save the cleaned DataFrame
        features_df.to_csv(os.path.join(features_folder, f'{ticker}.csv'), index=False)

    print("Feature engineering complete.")

if __name__ == "__main__":
    # Lookahead should be a range of t+x to predict. This means, at time t, we want to predict the target at t+0 as well as t+1, t+2, t+3, etc.
    look_ahead_periods = range(0, 1)
    main(use_dummy_data=True, look_ahead_periods=look_ahead_periods)  # Set to False to use real data
