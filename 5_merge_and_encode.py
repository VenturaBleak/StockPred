import os
import pandas as pd
import numpy as np
from glob import glob
import category_encoders as ce

# Define folders
features_data_folder = os.path.join('data', '4_features')
features_dummy_data_folder = os.path.join('data', '4_features_dummy')
final_data_folder = os.path.join('data', '5_final')
final_dummy_data_folder = os.path.join('data', '5_final_dummy')
os.makedirs(final_data_folder, exist_ok=True)
os.makedirs(final_dummy_data_folder, exist_ok=True)

# Function to load and concatenate all feature-engineered files
def load_and_concatenate_files(folder):
    all_files = glob(os.path.join(folder, '*.csv'))
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

# Function to encode the 'Ticker' column
def encode_ticker_column(df):
    original_ticker = df['Ticker']
    encoder = ce.BinaryEncoder(cols=['Ticker'])
    df = encoder.fit_transform(df)
    df['Ticker'] = original_ticker
    return df

# Main function to merge, encode, and split the dataset
def main(use_dummy_data=False):
    features_folder = features_dummy_data_folder if use_dummy_data else features_data_folder
    final_folder = final_dummy_data_folder if use_dummy_data else final_data_folder

    # Load and concatenate all feature-engineered files
    combined_df = load_and_concatenate_files(features_folder)

    # Add date numbered column
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])

    # Encode the 'Ticker' column
    combined_df = encode_ticker_column(combined_df)

    # Reorder columns to have Date and Ticker first
    combined_df = combined_df[['Date', 'Ticker'] + [col for col in combined_df.columns if col not in ['Date', 'Ticker']]]

    # Set index to be a combination of Date and Ticker
    combined_df.set_index(['Date', 'Ticker'], inplace=True)

    # Check for 'inf' or 'NA' values in the final dataset and raise exception if found
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    if combined_df[numeric_cols].isna().sum().sum() > 0 or np.isinf(combined_df[numeric_cols]).sum().sum() > 0:
        raise ValueError("Invalid values found in the final dataset")

    # Determine the split date based on the specified number of most recent dates
    num_test_days = 500  # Number of most recent days to be used as holdout set
    unique_dates = combined_df.index.get_level_values('Date').unique()
    split_date = unique_dates[-num_test_days]

    # Split the dataset into training and holdout test sets
    train_df = combined_df[combined_df.index.get_level_values('Date') < split_date]
    holdout_df = combined_df[combined_df.index.get_level_values('Date') >= split_date]

    # Save the training and holdout datasets
    train_df.to_csv(os.path.join(final_folder, 'train_dataset.csv'))
    holdout_df.to_csv(os.path.join(final_folder, 'holdout_dataset.csv'))

    print("Merging, encoding, and splitting complete.")

if __name__ == "__main__":
    main(use_dummy_data=True)  # Set to False to use real data