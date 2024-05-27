# run_all_steps.py

# Description: This script is the main script that will be used to run the entire project.

import os

# user input: data preparation?
prepare_data = input("Do you want to prepare data? (yes/no): ").strip().lower()
if prepare_data == 'yes':
    # Step 1: Create time ground truth
    print("-" * 50)
    print("Creating time ground truth...")
    exec(open('1_create_date_ground_truth.py').read())

    # if dummy data is used, generate dummy data, else download real data
    use_dummy_data = input("Do you want to use dummy data? (yes/no): ").strip().lower()
    if use_dummy_data == 'yes':
        # Step 2b: Generate Dummy Data
        print("-" * 50)
        print("Generating dummy data...")
        exec(open('2b_generate_dummy_data.py').read())
    else:
        # Step 2a: Download Data
        print("-" * 50)
        print("Downloading data...")
        exec(open('2a_download_data.py').read())

    # Step 3: Preprocess Data
    print("-" * 50)
    print("Preprocessing data...")
    exec(open('3_preprocess_data.py').read())

    # Step 4: Feature Engineering
    print("-" * 50)
    print("Engineering features...")
    exec(open('4_feature_engineering.py').read())

    # Step 5: Merge and Encode
    print("-" * 50)
    print("Merging and encoding data...")
    exec(open('5_merge_and_encode.py').read())
else:
    print("Skipping data preparation.")

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