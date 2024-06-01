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
dummy_absolute_features = ['Target_Absolute', 'Feature1', 'Feature2', 'Feature3', 'Feature4']

# Specify periods for lagged and aggregated features
lag_periods = [1, 2, 3, 4, 5]
look_ahead_periods = range(0, 1)

# Function to calculate simple lagged features
def calculate_simple_lagged_features(df, features, lags):
    lagged_features = pd.DataFrame(index=df.index)
    for feature in features:
        for lag in lags:
            lagged_features[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    return lagged_features

# Function to calculate logarithmic look-ahead targets for multiple target columns
def calculate_log_look_ahead_targets(df, periods, target_columns):
    log_look_ahead_targets = pd.DataFrame(index=df.index)
    for target_column in target_columns:
        for period in periods:
            if period == 0:
                log_look_ahead_targets[f'{target_column}_T{period}'] = df[target_column]
                numerator = df[target_column]
                denominator = df[target_column].shift(1)
                log_return = np.log(numerator / denominator)
                invalid_log_mask = np.isnan(log_return)
                if invalid_log_mask.any():
                    invalid_numerator = numerator[invalid_log_mask]
                    invalid_denominator = denominator[invalid_log_mask]
                    invalid_result = log_return[invalid_log_mask]
                    print(f"Invalid log values encountered in {target_column}_T{period}:")
                    for num, denom, result in zip(invalid_numerator, invalid_denominator, invalid_result):
                        print(f"{50 * '-'}")
                        print(f"Formula: log({num} / {denom})")
                        print(f"Numerator: {num}")
                        print(f"Denominator: {denom}")
                        print(f"Result: {result}")

                log_look_ahead_targets[f'Target_log_return_T{period}'] = log_return
            else:
                log_look_ahead_targets[f'{target_column}_T{period}'] = df[target_column].shift(-period)
                numerator = df[target_column].shift(-period)
                denominator = df[target_column].shift(1)
                log_return = np.log(numerator / denominator)
                invalid_log_mask = np.isnan(log_return)
                if invalid_log_mask.any():
                    invalid_numerator = numerator[invalid_log_mask]
                    invalid_denominator = denominator[invalid_log_mask]
                    invalid_result = log_return[invalid_log_mask]
                    print(f"Invalid log values encountered in {target_column}_T{period}:")
                    for num, denom, result in zip(invalid_numerator, invalid_denominator, invalid_result):
                        print(f"{50 * '-'}")
                        print(f"Formula: log({num} / {denom})")
                        print(f"Numerator: {num}")
                        print(f"Denominator: {denom}")
                        print(f"Result: {result}")
                log_look_ahead_targets[f'Target_log_return_T{period}'] = log_return
    return log_look_ahead_targets

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

        # Calculate simple lagged features for absolute features
        lagged_features = calculate_simple_lagged_features(df, absolute_features, lag_periods)

        # Define time features, independent of capitalization
        time_feature_cols = [col for col in df.columns if any(term in col.lower() for term in ['numbered', 'time', 'day', 'month', 'year', 'week', 'quarter'])]
        time_features = df[time_feature_cols]

        # Calculate look-ahead targets for all target columns
        log_look_ahead_targets = calculate_log_look_ahead_targets(df, look_ahead_periods, target_columns)

        # Combine all features into one DataFrame
        features_df = pd.concat([df[['Date', 'Ticker']], time_features,
                                 lagged_features, log_look_ahead_targets,
                                 df[['Feature1', 'Feature2', 'Feature3', 'Feature4']]], axis=1)

        # Drop rows with missing values in the aboslute and log target columns
        target_columns_lagged = [f'Target_log_return_T{period}' for col in target_columns for period in look_ahead_periods]
        features_df.dropna(subset=target_columns_lagged, inplace=True)
        features_df.drop(columns=target_columns, inplace=True, errors='ignore')

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
    main(use_dummy_data=True, look_ahead_periods=look_ahead_periods)  # Set to False to use real data
