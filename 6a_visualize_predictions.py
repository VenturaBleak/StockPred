import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define paths
results_folder = os.path.join('data', '6_results_dummy')
predictions_file = os.path.join(results_folder, 'all_predictions.csv')
ground_truth_file = os.path.join('data', '5_final_dummy', 'train_dataset.csv')  # Adjust the path as necessary

# Load predictions
predictions_df = pd.read_csv(predictions_file, parse_dates=['Date'])
ground_truth_df = pd.read_csv(ground_truth_file, parse_dates=['Date'])


# Function to convert log returns to absolute values
def convert_log_to_absolute(df, ground_truth_df):
    df = df.sort_values(by=['Date', 'Ticker'])
    ground_truth_df = ground_truth_df[['Date', 'Ticker', 'Target_Absolute_T0']]
    ground_truth_df = ground_truth_df.rename(columns={'Target_Absolute_T0': 'Prev_Absolute'})

    # Merge with ground truth to get the previous day's absolute values
    df = df.merge(ground_truth_df, on=['Date', 'Ticker'], how='left')
    df['Prev_Absolute'] = df['Prev_Absolute'].shift(1)

    # Calculate absolute values from log returns
    df['Actual_Absolute'] = df['Prev_Absolute'] * np.exp(df['Actual'])
    df['Predicted_Absolute'] = df['Prev_Absolute'] * np.exp(df['Predicted'])

    return df


# Function to calculate MAPE
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# Plot actual vs. predicted values for the first up to five tickers
tickers = predictions_df['Ticker'].unique()[:5]
mape_list = []

for ticker in tickers:
    ticker_data = predictions_df[predictions_df['Ticker'] == ticker]

    plt.figure(figsize=(14, 7))
    for target_col in ticker_data['Target'].unique():
        target_data = ticker_data[ticker_data['Target'] == target_col]

        # If the target column is in log return format, convert to absolute
        if 'log_return' in target_col:
            target_data = convert_log_to_absolute(target_data, ground_truth_df)
            actual_values = target_data['Actual_Absolute']
            predicted_values = target_data['Predicted_Absolute']
        else:
            actual_values = target_data['Actual']
            predicted_values = target_data['Predicted']

        # Calculate MAPE for the ticker
        mape = calculate_mape(actual_values, predicted_values)
        mape_list.append(mape)

        # Plot only the dates present in the dataset
        plt.plot(target_data['Date'], actual_values.values, label=f'Actual {target_col}')
        plt.plot(target_data['Date'], predicted_values.values, label=f'Predicted {target_col}', linestyle='--')

    plt.title(f'Actual vs Predicted for Ticker: {ticker} (MAPE: {mape:.2f}%)')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
    plt.show()

# Calculate and display overall MAPE
overall_mape = np.mean(mape_list)
print(f"Overall MAPE across all tickers: {overall_mape:.2f}%")
