import pandas as pd
import os
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import tqdm
import logging
from sklearn.base import clone
import optuna

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
logging.basicConfig(filename=os.path.join(results_folder, 'model_training.log'), level=logging.INFO,
                    format='%(asctime)s %(message)s')

# Define folder and load data
train_data_file = os.path.join(final_data_folder, 'train_dataset.csv')
print(f"Loading data from: {train_data_file}")
train_df = pd.read_csv(train_data_file, parse_dates=['Date'], index_col=['Date', 'Ticker'])
print("Data loaded successfully.")

# Sort the MultiIndex
train_df = train_df.sort_index()
print("Data sorted by MultiIndex.")

# Identify target columns dynamically
target_columns_absolute = [col for col in train_df.columns if col.startswith('Target_Absolute_T')]
target_columns_log = [col for col in train_df.columns if col.startswith('Target_log_return_T')]
look_ahead_periods = sorted(set(int(col.split('_')[-1][1:]) for col in target_columns_absolute + target_columns_log))
print(f"Identified target columns: {target_columns_absolute + target_columns_log}")

# Prompt user whether absolute or log return target
target_type = input("Enter the target type (absolute/log): ").strip().lower()
if target_type == 'absolute':
    target_columns = target_columns_absolute
    drop_columns = target_columns_log
elif target_type == 'log':
    target_columns = target_columns_log
    drop_columns = target_columns_absolute
else:
    raise ValueError("Invalid target type. Choose 'absolute' or 'log'.")

# Check if the target columns exist
if not target_columns:
    raise KeyError(f"No columns found for the specified target type: {target_type}")

# Prepare the data for model training
X_train = train_df.drop(columns=target_columns_absolute + target_columns_log)
y_train_dict = {col: train_df[col] for col in target_columns}

# Print features and corresponding dtypes
print("-" * 50)
print("Feature Information:")
print("Feature_Name".ljust(30), "Data_Type".ljust(20))
print("-" * 50)
for col in X_train.columns:
    print(col.ljust(30), str(X_train[col].dtype).ljust(20))

print("-" * 50)
print("Target Variable Information:")
for target_col in target_columns:
    print(target_col.ljust(30), str(y_train_dict[target_col].dtype).ljust(20))
print("-" * 50)

# Define rolling window parameters
OPTIMIZATION_TIME = 3600 / 15 # 3600 seconds = 1 hour
train_window = 2500  # Number of days in the training window
step_size = 100  # Number of days to move the window forward each iteration
val_window = step_size  # Number of days in the validation window, equal to step size

# Calculate the total number of rolling window folds
unique_dates = sorted(X_train.index.get_level_values('Date').unique())
total_periods = len(unique_dates)
n_splits = (total_periods - train_window - val_window) // step_size + 1

print(f"Total number of dates in training set: {total_periods}")
print(f"Total number of CV splits: {n_splits}")
print(f"lookahead periods: {look_ahead_periods}")

# Initialize lists to store results and predictions
train_scores = {col: [] for col in target_columns}
val_scores = {col: [] for col in target_columns}
all_predictions = []

# Prompt user for the objective function
objective_func_name = input("Enter the objective function (mse/mae/rmse/mape): ").strip().lower()
if objective_func_name == 'mse':
    objective_func = mean_squared_error
    metric = 'l2'
elif objective_func_name == 'mae':
    objective_func = mean_absolute_error
    metric = 'l1'
elif objective_func_name == 'rmse':
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    objective_func = rmse
    metric = 'rmse'
elif objective_func_name == 'mape':
    objective_func = mean_absolute_percentage_error
    metric = 'mape'
else:
    raise ValueError("Invalid objective function. Choose 'mse', 'mae', 'rmse', or 'mape'.")

# Define the objective function for Optuna
def objective(trial):
    param = {
        # Core parameters
        'boosting_type': 'gbdt',  # Default boosting type
        'objective': 'regression',  # Loss function to be minimized
        'metric': metric,  # Evaluation metric
        'verbosity': -1,  # Suppress output

        # Learning control parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.25),  # Learning rate
        'early_stopping_round': 100,  # Early stopping rounds

        # Tree parameters
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000), # 100 to 5000
        'num_leaves': trial.suggest_int('num_leaves', 5, 2000),  # Max number of leaves per tree
        'max_depth': trial.suggest_int('max_depth', 5, 2000),  # Max depth of tree
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 3, 100),  # Min data points in a leaf
        'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 0.001, 50.0),
        # Min sum of instance weight in a leaf
        'max_bin': trial.suggest_int('max_bin', 10, 255),  # Max number of bins for discretizing continuous features

        # Regularization parameters
        'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 30.0),  # L1 regularization term on weights
        'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 30.0),  # L2 regularization term on weights
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.001, 1.0),  # Min gain to make a split
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 30.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 30.0),


        # Other parameters
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),  # Fraction of features to consider
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Fraction of data to use for each iteration
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),  # Frequency of subsample
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 300),  # Min data points in a child
        'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 40.0),  # Min sum of instance weight
        'min_split_gain': trial.suggest_float('min_split_gain', 0.001, 1.0),  # Min loss reduction to make a split
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.01, 40.0),
        # Balancing of positive and negative weights
        'max_delta_step': trial.suggest_float('max_delta_step', 0.01, 40.0),
        # Max delta step for each tree's weight estimation


        # Feature parameters
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        # Fraction of features to consider at each split
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        # Fraction of data to use for each iteration
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),  # Frequency of bagging
        'max_bin': trial.suggest_int('max_bin', 10, 255),  # Max number of bins for discretizing continuous features
        'cat_smooth': trial.suggest_int('cat_smooth', 10, 100),  # Smoothing factor for categorical feature splits
        'cat_l2': trial.suggest_float('cat_l2', 1, 40),  # L2 regularization for categorical features
        'min_data_per_group': trial.suggest_int('min_data_per_group', 5, 500),  # Min data points per categorical group
    }

    base_model = LGBMRegressor(**param)
    val_scores_trial = {col: [] for col in target_columns}

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
        for target_col in target_columns:
            y_fold_train = y_train_dict[target_col].loc[train_date_range[0]:train_date_range[-1]]
            y_fold_val = y_train_dict[target_col].loc[val_date_range[0]:val_date_range[-1]]

            model = clone(base_model)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                eval_metric=metric
            )

            y_val_pred = model.predict(X_fold_val)

            val_score = objective_func(y_fold_val, y_val_pred)
            val_scores_trial[target_col].append(val_score)

    avg_val_score = np.mean([np.mean(val_scores_trial[target_col]) for target_col in val_scores_trial])
    return avg_val_score

# optuna sampler
sampler = optuna.samplers.TPESampler(
    seed=42,
    n_startup_trials=30
)

# Optimize hyperparameters with Optuna
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, timeout=OPTIMIZATION_TIME)

# Output the best hyperparameters
best_params = study.best_params
best_params['verbosity'] = -1  # Add verbosity parameter
print("Best hyperparameters:", best_params)

# Perform rolling window cross-validation with the best hyperparameters
final_model = LGBMRegressor(**best_params)

fold_val_scores = []

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
    for target_col in target_columns:
        y_fold_train = y_train_dict[target_col].loc[train_date_range[0]:train_date_range[-1]]
        y_fold_val = y_train_dict[target_col].loc[val_date_range[0]:val_date_range[-1]]

        final_model.fit(X_fold_train, y_fold_train)

        y_train_pred = final_model.predict(X_fold_train)
        y_val_pred = final_model.predict(X_fold_val)

        train_score = objective_func(y_fold_train, y_train_pred)
        val_score = objective_func(y_fold_val, y_val_pred)

        train_scores[target_col].append(train_score)
        val_scores[target_col].append(val_score)
        fold_val_scores.append(val_score)

        # Log predictions for each fold
        val_predictions = pd.DataFrame({
            'Date': X_fold_val.index.get_level_values('Date'),
            'Ticker': X_fold_val.index.get_level_values('Ticker'),
            'Actual': y_fold_val,
            'Predicted': y_val_pred,
            'Target': target_col,
            'Fold': fold
        })
        all_predictions.append(val_predictions)

    tqdm.tqdm.write(f"Fold {fold + 1}/{n_splits} - Val {objective_func_name.upper()}: {np.mean(fold_val_scores)}")

# print average scores
print("-" * 50)
print("Average Scores:")
for target_col in target_columns:
    print(f"Train {objective_func_name.upper()} {target_col}: {np.mean(train_scores[target_col])}")
    print(f"Val {objective_func_name.upper()} {target_col}: {np.mean(val_scores[target_col])}")

# Save the results
print("Saving results...")
results_dict = {f'Train {objective_func_name.upper()} {target_col}': train_scores[target_col] for target_col in
                target_columns}
results_dict.update(
    {f'Val {objective_func_name.upper()} {target_col}': val_scores[target_col] for target_col in target_columns})

results_df = pd.DataFrame(results_dict)
results_df.to_csv(os.path.join(results_folder, 'cv_results.csv'), index=False)

# Save all predictions
predictions_df = pd.concat(all_predictions)
predictions_file = os.path.join(results_folder, 'all_predictions.csv')
predictions_df.to_csv(predictions_file, index=False)
print(f"Predictions saved to {predictions_file}")

print("Rolling window cross-validation and model training complete.")