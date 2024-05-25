# run_all_steps.py

# Description: This script is the main script that will be used to run the entire project.

import os

# Step 1: Create time ground truth
print("Creating time ground truth...")
exec(open('1_create_date_ground_truth.py').read())

# Step 2: Download Data
print("Downloading data...")
exec(open('2_download_data.py').read())

# Step 3: Preprocess Data
print("Preprocessing data...")
exec(open('3_preprocess_data.py').read())

# Step 4: Feature Engineering
print("Engineering features...")
exec(open('4_feature_engineering.py').read())

# Step 5: Merge and Encode
print("Merging and encoding data...")
exec(open('5_merge_and_encode.py').read())

# Prompt user: do you want to train the model
train_model = input("Do you want to train the model? (yes/no): ").strip().lower()

if train_model == 'yes':
    print("Training the model...")
    # Step 6: Model Training based on Rolling Window Cross-Validation
    print("Performing rolling window cross-validation...")
    exec(open('6_rolling_window_cv.py').read())

    # Step 7: Model Evaluation
    print("Evaluating the model...")
    exec(open('7_model_evaluation.py').read())
else:
    print("Skipping model training.")

print("Done!")
