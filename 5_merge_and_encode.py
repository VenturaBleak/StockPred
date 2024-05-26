# 5_merge_and_encode.py
import pandas as pd
import numpy as np
import os
from glob import glob
import category_encoders as ce

# Define folders
features_data_folder = 'data/4_features'
final_data_folder = 'data/5_final'
os.makedirs(final_data_folder, exist_ok=True)

# Load and concatenate all feature-engineered files
all_files = glob(os.path.join(features_data_folder, '*.csv'))
df_list = [pd.read_csv(file) for file in all_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Add date numbered column
combined_df['Date'] = pd.to_datetime(combined_df['Date'])


# Save original 'Ticker' column before encoding
original_ticker = combined_df['Ticker']

# Binary encode the ticker symbol using category_encoders
encoder = ce.BinaryEncoder(cols=['Ticker'])
combined_df = encoder.fit_transform(combined_df)

# Add the original 'Ticker' column back to the DataFrame for reference
combined_df['Ticker'] = original_ticker

# Reorder columns to have Date and Ticker first
combined_df = combined_df[['Date', 'Ticker'] + [col for col in combined_df.columns if col not in ['Date', 'Ticker']]]

# Set index to be a combination of Date and Ticker
combined_df.set_index(['Date', 'Ticker'], inplace=True)

# Check for 'inf' or 'NA' values in the final dataset and raise exception if found
numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
if combined_df[numeric_cols].isna().sum().sum() > 0 or np.isinf(combined_df[numeric_cols]).sum().sum() > 0:
    raise ValueError("Invalid values found in the final dataset")

# Determine the split date based on the specified number of most recent dates
num_test_days = 600  # Number of most recent days to be used as test set
split_date = combined_df.index.get_level_values('Date').unique()[-num_test_days]

# Split the dataset into training and test sets
train_df = combined_df[combined_df.index.get_level_values('Date') < split_date]
test_df = combined_df[combined_df.index.get_level_values('Date') >= split_date]

# Save the training and test datasets
train_df.to_csv(os.path.join(final_data_folder, 'train_dataset.csv'))
test_df.to_csv(os.path.join(final_data_folder, 'test_dataset.csv'))

print("Merging, encoding, and splitting complete.")