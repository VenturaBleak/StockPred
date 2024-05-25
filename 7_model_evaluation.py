# 7_model_evaluation.py
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Ensure the necessary directories exist
os.makedirs('data/7_results/tickers', exist_ok=True)

# Load the ensemble models
model_file = os.path.join('data/6_models', 'ensemble_models.pkl')
ensemble_models = joblib.load(model_file)

# Define folder and load data
test_data_file = os.path.join('data/5_final', 'test_dataset.csv')
test_df = pd.read_csv(test_data_file, parse_dates=['Date'], index_col=['Date', 'Ticker'])

print("Data loaded successfully.")
# Print date range
print(f"Data date range: {test_df.index.get_level_values('Date').min()} to {test_df.index.get_level_values('Date').max()}")
print(f"----------------------------------------")

# Sort the MultiIndex
test_df = test_df.sort_index()

# Prepare the test set
X_test_final = test_df.drop(columns=['Daily_Return'])
y_test_final = test_df['Daily_Return']

# Use the ensemble of models to predict on the test set
start_time = time.time()
test_preds = np.mean([model.predict(X_test_final) for model in ensemble_models], axis=0)
ensemble_inference_time = time.time() - start_time

# Calculate ensemble model metrics
ensemble_mse = mean_squared_error(y_test_final, test_preds)
ensemble_mae = mean_absolute_error(y_test_final, test_preds)
ensemble_r2 = r2_score(y_test_final, test_preds)

print(f"Ensemble Model - Test MSE: {ensemble_mse}, MAE: {ensemble_mae}, R²: {ensemble_r2}, Inference Time: {ensemble_inference_time}")

# Save the final test predictions
test_results_df = pd.DataFrame({
    'True Values': y_test_final,
    'Predictions': test_preds
}, index=y_test_final.index)
test_results_df.to_csv(os.path.join('data/7_results', 'final_test_predictions.csv'))

# Baseline models for comparison using the aggregated features
baseline_features = [col for col in X_test_final.columns if col.startswith('Daily_Return_mean')]
baseline_errors = {}
baseline_maes = {}
baseline_r2s = {}
baseline_times = {}
baseline_preds = {}

for feature in baseline_features:
    start_time = time.time()
    baseline_pred = X_test_final[feature]
    inference_time = time.time() - start_time

    mse = mean_squared_error(y_test_final, baseline_pred)
    mae = mean_absolute_error(y_test_final, baseline_pred)
    r2 = r2_score(y_test_final, baseline_pred)

    baseline_preds[feature] = baseline_pred
    baseline_errors[feature] = mse
    baseline_maes[feature] = mae
    baseline_r2s[feature] = r2
    baseline_times[feature] = inference_time

    print()

# Prepare the data for plotting
plot_data = test_results_df.reset_index()
plot_data['Date'] = plot_data['Date'].dt.strftime('%Y-%m-%d')

# Plot the overall results
plt.figure(figsize=(14, 7))

# Plot true values
sns.lineplot(data=plot_data, x='Date', y='True Values', label='True Values')

# Plot ensemble model predictions
sns.lineplot(data=plot_data, x='Date', y='Predictions', label='Ensemble Predictions')

# Plot baseline model predictions
for feature, pred in baseline_preds.items():
    pred_data = pd.DataFrame({
        'Date': plot_data['Date'],
        feature: pred.values
    })
    sns.lineplot(data=pred_data, x='Date', y=feature, label=f'Baseline ({feature})')

plt.legend()
plt.title(f'Model Predictions vs True Values (Overall)\nMAE: {ensemble_mae}, R²: {ensemble_r2}')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.xticks(rotation=45)
plt.savefig(os.path.join('data/7_results', 'model_vs_baseline_comparison_overall.png'))
plt.show()

# Plot results per ticker
tickers = test_df.index.get_level_values('Ticker').unique()
for ticker in tickers:
    ticker_data = plot_data[plot_data['Ticker'] == ticker]
    plt.figure(figsize=(14, 7))

    # Plot true values
    sns.lineplot(data=ticker_data, x='Date', y='True Values', label='True Values')

    # Plot ensemble model predictions
    sns.lineplot(data=ticker_data, x='Date', y='Predictions', label='Ensemble Predictions')

    # Plot baseline model predictions
    for feature, pred in baseline_preds.items():
        pred_data = pd.DataFrame({
            'Date': ticker_data['Date'],
            feature: pred.loc[(slice(None), ticker)].values
        })
        sns.lineplot(data=pred_data, x='Date', y=feature, label=f'Baseline ({feature})')

    plt.legend()
    plt.title(f'Model Predictions vs True Values ({ticker})\nMAE: {ensemble_mae}, R²: {ensemble_r2}')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.xticks(rotation=45)
    ticker_dir = 'data/7_results/tickers'
    os.makedirs(ticker_dir, exist_ok=True)
    plt.savefig(os.path.join(ticker_dir, f'model_vs_baseline_comparison_{ticker}.png'))
    plt.show()

# Create a comparison table
comparison_table = pd.DataFrame({
    'Model': ['Ensemble'] + list(baseline_features),
    'MSE': [ensemble_mse] + [baseline_errors[feature] for feature in baseline_features],
    'MAE': [ensemble_mae] + [baseline_maes[feature] for feature in baseline_features],
    'R²': [ensemble_r2] + [baseline_r2s[feature] for feature in baseline_features],
    'Inference Time': [ensemble_inference_time] + [baseline_times[feature] for feature in baseline_features]
})

comparison_table.to_csv(os.path.join('data/7_results', 'model_comparison.csv'), index=False)

# Print baseline errors
print("Baseline Errors:")
for feature, error in baseline_errors.items():
    print(f"{feature} Baseline MSE: {error}, MAE: {baseline_maes[feature]}, R²: {baseline_r2s[feature]}, Inference Time: {baseline_times[feature]}")

print("Evaluation complete.")