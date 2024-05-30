import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
results_folder = os.path.join('data', '6_results_dummy')
predictions_file = os.path.join(results_folder, 'all_predictions.csv')

# Load predictions
predictions_df = pd.read_csv(predictions_file, parse_dates=['Date'])

# Plot actual vs. predicted values for the first up to five tickers
tickers = predictions_df['Ticker'].unique()[:5]
for ticker in tickers:
    ticker_data = predictions_df[predictions_df['Ticker'] == ticker]

    plt.figure(figsize=(14, 7))
    for target_col in ticker_data['Target'].unique():
        target_data = ticker_data[ticker_data['Target'] == target_col]

        # Extract values
        actual_values = target_data['Actual']
        predicted_values = target_data['Predicted']

        # Plot only the dates present in the dataset
        plt.plot(actual_values.values, label=f'Actual {target_col}')
        plt.plot(predicted_values.values, label=f'Predicted {target_col}', linestyle='--')

    plt.title(f'Actual vs Predicted for Ticker: {ticker}')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
    plt.show()
