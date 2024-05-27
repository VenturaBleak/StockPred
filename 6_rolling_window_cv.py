import pandas as pd
import os
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import tqdm
import logging
import joblib
from sklearn.base import clone

# Define folders
def get_data_paths(use_dummy_data):
    if use_dummy_data:
        final_data_folder = os.path.join('data', '5_final_dummy')
        results_folder = os.path.join('data', '6_results_dummy')
    else:
        final_data_folder = os.path.join('data', '5_final')
        results_folder = os.path.join('data', '6_results')
    return final_data_folder, results_folder

# Ensure the necessary directories exist
final_data_folder, results_folder = get_data_paths(use_dummy_data=True)
os.makedirs(results_folder, exist_ok=True)

# Setup logging
logging.basicConfig(filename=os.path.join(results_folder, 'model_training.log'), level=logging.INFO, format='%(asctime)s %(message)s')

# Define folder and load data
train_data_file = os.path.join(final_data_folder, 'train_dataset.csv')
print(f"Loading data from: {train_data_file}")
train_df = pd.read_csv(train_data_file, parse_dates=['Date'], index_col=['Date', 'Ticker'])
print("Data loaded successfully.")

# Sort the MultiIndex
train_df = train_df.sort_index()
print("Data sorted by MultiIndex.")

# Identify target columns dynamically
target_columns = [col for col in train_df.columns if col.startswith('Target_T')]
look_ahead_periods = sorted([int(col.replace('Target_T', '')) for col in target_columns])
print(f"Identified target columns: {target_columns}")

# Prepare the data for model training
X_train = train_df.drop(columns=target_columns)
y_train_dict = {f'Target_T{period}': train_df[f'Target_T{period}'] for period in look_ahead_periods}

# Print features and corresponding dtypes
print("-" * 50)
print("Feature Information:")
print("Feature_Name".ljust(30), "Data_Type".ljust(20))
print("-" * 50)
for col in X_train.columns:
    print(col.ljust(30), str(X_train[col].dtype).ljust(20))

print("-" * 50)
print("Target Variable Information:")
for period in look_ahead_periods:
    target_col = f'Target_T{period}'
    print(target_col.ljust(30), str(y_train_dict[target_col].dtype).ljust(20))
print("-" * 50)

# Define rolling window parameters
train_window = 2500  # Number of days in the training window
val_window = 1  # Number of days in the validation window
step_size = 1  # Number of days to move the window forward each iteration

# Calculate the total number of rolling window folds
unique_dates = sorted(X_train.index.get_level_values('Date').unique())
total_periods = len(unique_dates)
n_splits = (total_periods - train_window - val_window) // step_size + 1

print(f"Total number of dates in training set: {total_periods}")
print(f"Total number of CV splits: {n_splits}")
print(f"lookahead periods: {look_ahead_periods}")

# Initialize lists to store results
train_mse_scores = {f'Target_T{period}': [] for period in look_ahead_periods}
val_mse_scores = {f'Target_T{period}': [] for period in look_ahead_periods}

# Use LightGBM Regressor
base_model = LGBMRegressor(random_state=42)

# Use RandomForest Regressor
# base_model = RandomForestRegressor(random_state=42)

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

    X_fold_train = X_train.loc[train_date_range[0]:train_date_range[-1]].copy()
    X_fold_val = X_train.loc[val_date_range[0]:val_date_range[-1]].copy()

    # Process each target period iteratively
    for period in look_ahead_periods:
        target_col = f'Target_T{period}'

        y_fold_train = y_train_dict[target_col].loc[train_date_range[0]:train_date_range[-1]]
        y_fold_val = y_train_dict[target_col].loc[val_date_range[0]:val_date_range[-1]]

        model = clone(base_model)
        model.fit(X_fold_train, y_fold_train)

        y_train_pred = model.predict(X_fold_train)
        y_val_pred = model.predict(X_fold_val)

        train_mse = mean_squared_error(y_fold_train, y_train_pred)
        val_mse = mean_squared_error(y_fold_val, y_val_pred)

        train_mse_scores[target_col].append(train_mse)
        val_mse_scores[target_col].append(val_mse)
        #
        # # Add predictions to the training and validation sets for subsequent periods
        # X_fold_train.loc[:, f'Pred_{target_col}'] = y_train_pred
        # X_fold_val.loc[:, f'Pred_{target_col}'] = y_val_pred
        #
        # # Update X_train with new prediction features for next iterations
        # X_train.loc[train_date_range[0]:train_date_range[-1], f'Pred_{target_col}'] = y_train_pred
        # X_train.loc[val_date_range[0]:val_date_range[-1], f'Pred_{target_col}'] = y_val_pred
        #
        # # Log the results for each period and fold
        # logging.info(f"Fold {fold + 1}/{n_splits} - Train MSE {target_col}: {train_mse}, Val MSE {target_col}: {val_mse}")

# Save the results
print("Saving results...")
results_dict = {f'Train MSE {target_col}': train_mse_scores[target_col] for target_col in target_columns}
results_dict.update({f'Val MSE {target_col}': val_mse_scores[target_col] for target_col in target_columns})

results_df = pd.DataFrame(results_dict)
results_df.to_csv(os.path.join(results_folder, 'cv_results.csv'), index=False)

print("Rolling window cross-validation and model training complete.")