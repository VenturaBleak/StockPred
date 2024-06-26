import os
import pandas as pd
import numpy as np
from glob import glob

# Define folders
input_folder = os.path.join('data', '2_raw')
dummy_data_folder = os.path.join('data', '2_raw_dummy')
output_folder = os.path.join('data', '3_preprocessed')
dummy_output_folder = os.path.join('data', '3_preprocessed_dummy')
additional_features_folder = os.path.join('data', 'additional_features')

os.makedirs(output_folder, exist_ok=True)
os.makedirs(dummy_output_folder, exist_ok=True)

def load_ground_truth_dates(file_path):
    """Loads the ground truth dates from a CSV file."""
    ground_truth_dates = pd.read_csv(file_path)
    ground_truth_dates['Date'] = pd.to_datetime(ground_truth_dates['Date'])
    return ground_truth_dates

def load_additional_features(folder_path):
    """Loads additional feature datasets from a specified folder."""
    feature_files = glob(os.path.join(folder_path, '*.csv'))
    additional_dfs = []
    for file in feature_files:
        df = pd.read_csv(file)
        if 'OVERVIEW' in file:
            join_columns = ['Ticker']
        elif any(x in file for x in ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW']):
            join_columns = ['Ticker', 'Year']
        elif 'EARNINGS' in file:
            join_columns = ['Ticker', 'Year', 'Quarter']
        else:
            join_columns = ['Ticker', 'Date']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        additional_dfs.append((df, join_columns))
    return additional_dfs

def merge_datasets(main_df, additional_dfs):
    """Merges additional data sources into the main DataFrame based on specified join columns."""
    for df, join_columns in additional_dfs:
        if 'Date' in join_columns:
            main_df = main_df.merge(df, on=join_columns, how='left')
        else:
            main_df['Year'] = main_df['Date'].dt.year
            main_df = main_df.merge(df, left_on=['Ticker', 'Year'], right_on=join_columns, how='left')
            main_df.drop(columns=['Year'], inplace=True)
    return main_df

def preprocess_file(file, ground_truth_dates, additional_data_sources):
    """Preprocesses a single raw data file, aligning it with ground truth dates, and merging additional data sources."""
    print(f"Preprocessing {file}...")
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = pd.merge(ground_truth_dates, df, on='Date', how='left')
    df['Ticker'] = os.path.basename(file).split('.')[0]

    # Merge additional data sources
    df = merge_datasets(df, additional_data_sources)

    return df

def save_preprocessed_data(df, ticker, output_folder):
    """Saves the preprocessed DataFrame as a CSV file."""
    file_path = os.path.join(output_folder, f'{ticker}.csv')
    df.to_csv(file_path, index=False)
    print(f"Preprocessed data saved to {file_path}")

def main(use_dummy_data=False):
    # Load ground truth dates
    ground_truth_dates = load_ground_truth_dates('data/1_ground_truth_dates.csv')

    # Load additional feature datasets
    additional_features = load_additional_features(additional_features_folder)

    # Determine the folder to use for raw data
    data_folder = dummy_data_folder if use_dummy_data else input_folder
    output_folder_to_use = dummy_output_folder if use_dummy_data else output_folder

    # Delete files in the folder, if using dummy data
    if use_dummy_data:
        for file in os.listdir(output_folder_to_use):
            os.remove(os.path.join(output_folder_to_use, file))

    # Process each raw data file
    for file in glob(os.path.join(data_folder, '*.csv')):
        ticker = os.path.basename(file).split('.')[0]
        preprocessed_df = preprocess_file(file, ground_truth_dates, additional_features)

        # Ensure consistent output format
        # columns_to_keep = ['Date', 'Ticker'] + \
        #                   [col for col in preprocessed_df.columns if col not in ['Date', 'Ticker'] and not col.startswith('Prev_')]
        # preprocessed_df = preprocessed_df[columns_to_keep]

        # Drop prev_ columns
        # preprocessed_df = preprocessed_df.drop(columns=[col for col in preprocessed_df.columns if col.startswith('Prev_')])

        # Save preprocessed data
        save_preprocessed_data(preprocessed_df, ticker, output_folder_to_use)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main(use_dummy_data=True)  # Set to False to use real data