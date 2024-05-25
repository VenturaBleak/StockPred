# 6_rolling_window_cv.py
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import tqdm
import logging
import joblib
from sklearn.base import clone

# Ensure the necessary directories exist
os.makedirs('data/6_models', exist_ok=True)
os.makedirs('data/6_results', exist_ok=True)

# Setup logging
logging.basicConfig(filename='data/6_models/model_training.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Define folder and load data
train_data_file = os.path.join('data/5_final', 'train_dataset.csv')
train_df = pd.read_csv(train_data_file, parse_dates=['Date'], index_col=['Date', 'Ticker'])

# Sort the MultiIndex
train_df = train_df.sort_index()

# Prepare the data for model training
X_train = train_df.drop(columns=['Daily_Return'])
y_train = train_df['Daily_Return']

# Print the features and target variable names and their corresponding dtypes
print("Features and Target Variable Information:")
print("Feature_Name".ljust(30), "Data_Type".ljust(20))
print("-" * 50)
for col in X_train.columns:
    print(col.ljust(30), str(X_train[col].dtype).ljust(20))

print("-" * 50)
# Print the target variable's name and its corresponding dtype
print("Target Variable Information:")
print(y_train.name.ljust(30), str(y_train.dtype).ljust(20))
print("-" * 50)

# Define rolling window parameters
train_window = 1200  # Number of days in the training window
val_window = 250  # Number of days in the validation window
step_size = 100  # Number of days to move the window forward each iteration

# Calculate the total number of rolling window folds
unique_dates = sorted(X_train.index.get_level_values('Date').unique())
total_periods = len(unique_dates)
n_splits = (total_periods - train_window - val_window) // step_size + 1

print(f"Total number of dates in training set: {total_periods}")
print(f"Total number of CV splits: {n_splits}")

# Initialize lists to store results
train_mse_scores = []
val_mse_scores = []
train_mae_scores = []
val_mae_scores = []

# Ensemble model using RandomForestRegressor
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
ensemble_models = []

# Perform rolling window cross-validation
for fold in tqdm.tqdm(range(n_splits)):

    split_start = fold * step_size
    train_start = split_start
    train_end = train_start + train_window
    val_start = train_end
    val_end = val_start + val_window

    if val_end > total_periods:
        break

    train_date_range = unique_dates[train_start:train_end]
    val_date_range = unique_dates[val_start:val_end]

    X_fold_train = X_train.loc[(train_date_range[0]):(train_date_range[-1])]
    X_fold_val = X_train.loc[(val_date_range[0]):(val_date_range[-1])]
    y_fold_train = y_train.loc[(train_date_range[0]):(train_date_range[-1])]
    y_fold_val = y_train.loc[(val_date_range[0]):(val_date_range[-1])]

    # Verification step
    train_dates = X_fold_train.index.get_level_values('Date')
    val_dates = X_fold_val.index.get_level_values('Date')
    train_tickers = X_fold_train.index.get_level_values('Ticker').unique()
    val_tickers = X_fold_val.index.get_level_values('Ticker').unique()

    # Ensure all tickers are included in the training set
    if not set(val_tickers).issubset(set(train_tickers)):
        print(f"cv fold: {fold}, train_tickers: {train_tickers}, val_tickers: {val_tickers}")
        raise ValueError("Not all tickers are included in the training set")

    # Ensure no time spillover
    if train_dates.max() >= val_dates.min():
        print(f"cv fold: {fold}, train_dates max: {train_dates.max()}, val_dates min: {val_dates.min()}")
        raise ValueError("Time spillover detected between training and validation sets")

    # Clone the base model to ensure each fold has a fresh model
    model = clone(base_model)

    # Train the model
    model.fit(X_fold_train, y_fold_train)  # Optimization based on MSE

    # Predict and evaluate
    y_train_pred = model.predict(X_fold_train)
    y_val_pred = model.predict(X_fold_val)

    # Calculate metrics
    train_mse = mean_squared_error(y_fold_train, y_train_pred)
    val_mse = mean_squared_error(y_fold_val, y_val_pred)
    train_mae = mean_absolute_error(y_fold_train, y_train_pred)
    val_mae = mean_absolute_error(y_fold_val, y_val_pred)

    # Store metrics
    train_mse_scores.append(train_mse)
    val_mse_scores.append(val_mse)
    train_mae_scores.append(train_mae)
    val_mae_scores.append(val_mae)

    # Append the model to the ensemble
    ensemble_models.append(model)

    # Log the results for each fold
    logging.info(f"Fold {fold + 1}/{n_splits} - Train MSE: {train_mse}, Val MSE: {val_mse}, Train MAE: {train_mae}, Val MAE: {val_mae}")

    if fold == 0 or fold == n_splits - 1:
        print(f"{50*'*'}")
        print(f"cv fold: {fold}, train_dates max: {train_dates.max()}, val_dates min: {val_dates.min()}")
        print(f"train_dates: {train_dates.unique()}, val_dates: {val_dates.unique()}")

# Save the ensemble models
model_file = os.path.join('data/6_models', 'ensemble_models.pkl')
joblib.dump(ensemble_models, model_file)

# Save the results
results_df = pd.DataFrame({
    'Train MSE': train_mse_scores,
    'Val MSE': val_mse_scores,
    'Train MAE': train_mae_scores,
    'Val MAE': val_mae_scores
})
results_df.to_csv(os.path.join('data/6_results', 'cv_results.csv'), index=False)

print("Rolling window cross-validation and model training complete.")