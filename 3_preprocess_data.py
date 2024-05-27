"""
3_preprocess_data.py

This script preprocesses raw time series data files, aligning them with ground truth dates,
and generating additional features based on the previous day's data. It ensures the data
is consistent and ready for subsequent analysis.
"""

import os
import pandas as pd
from glob import glob

# Define folders
input_folder = os.path.join('data', '2_raw')
dummy_data_folder = os.path.join('data', '2_raw_dummy')
output_folder = os.path.join('data', '3_preprocessed')
dummy_output_folder = os.path.join('data', '3_preprocessed_dummy')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(dummy_output_folder, exist_ok=True)

def load_ground_truth_dates(file_path):
    """
    Loads the ground truth dates from a CSV file.

    Args:
        file_path (str): Path to the ground truth dates file.

    Returns:
        pd.DataFrame: DataFrame containing the ground truth dates.
    """
    ground_truth_dates = pd.read_csv(file_path)
    ground_truth_dates['Date'] = pd.to_datetime(ground_truth_dates['Date'])
    return ground_truth_dates

def preprocess_file(file, ground_truth_dates):
    """
    Preprocesses a single raw data file, aligning it with ground truth dates,
    and generating features based on the previous day's data.

    Args:
        file (str): Path to the raw data file.
        ground_truth_dates (pd.DataFrame): DataFrame containing the ground truth dates.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    print(f"Preprocessing {file}...")
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = pd.merge(ground_truth_dates, df, on='Date', how='left')
    df['Ticker'] = os.path.basename(file).split('.')[0]

    # Add previous day's features for 'DateNumbered'
    df['Prev_DateNumbered'] = df['DateNumbered'].shift(1)

    # Calculate the difference between today's and previous day's 'DateNumbered'
    df['Diff_DateNumbered'] = df['DateNumbered'] - df['Prev_DateNumbered']

    return df

def save_preprocessed_data(df, ticker, output_folder):
    """
    Saves the preprocessed DataFrame as a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing the preprocessed data.
        ticker (str): Stock ticker symbol.
        output_folder (str): Folder to save the preprocessed data.
    """
    file_path = os.path.join(output_folder, f'{ticker}.csv')

    df.to_csv(file_path, index=False)
    print(f"Preprocessed data saved to {file_path}")

def main(use_dummy_data=False):
    # Load ground truth dates
    ground_truth_dates = load_ground_truth_dates('data/1_ground_truth_dates.csv')

    # Determine the folder to use for raw data
    data_folder = dummy_data_folder if use_dummy_data else input_folder
    output_folder_to_use = dummy_output_folder if use_dummy_data else output_folder

    # delete files in the folder, if using dummy data
    if use_dummy_data:
        for file in os.listdir(output_folder_to_use):
            os.remove(os.path.join(output_folder_to_use, file))

    # Process each raw data file
    for file in glob(os.path.join(data_folder, '*.csv')):
        ticker = os.path.basename(file).split('.')[0]  # Corrected to handle DUMMY1.csv case
        preprocessed_df = preprocess_file(file, ground_truth_dates)

        # Ensure consistent output format
        columns_to_keep = ['Date', 'Ticker'] + \
                          [col for col in preprocessed_df.columns if col not in ['Date', 'Ticker'] and not col.startswith('Prev_')]
        preprocessed_df = preprocessed_df[columns_to_keep]

        # drop prev_ columns
        preprocessed_df = preprocessed_df.drop(columns=[col for col in preprocessed_df.columns if col.startswith('Prev_')])

        # Save preprocessed data
        save_preprocessed_data(preprocessed_df, ticker, output_folder_to_use)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main(use_dummy_data=True)  # Set to False to use real data