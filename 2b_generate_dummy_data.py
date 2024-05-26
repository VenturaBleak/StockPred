"""
generate_dummy_data.py

This script generates sophisticated synthetic time series data for testing and validation purposes.
The data includes multiple features that influence the target variable. The data range is based
on the ground truth dates.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DummyDataGenerator:
    def __init__(self, data_folder, ground_truth_file):
        self.data_folder = data_folder
        os.makedirs(self.data_folder, exist_ok=True)
        self.ground_truth_dates = pd.read_csv(ground_truth_file)
        self.ground_truth_dates['Date'] = pd.to_datetime(self.ground_truth_dates['Date'])

    def generate_features(self, n, random_seed=None):
        """
        Generates multiple random features over time.

        Args:
            n (int): The number of timestamps.
            random_seed (int): Random seed for reproducibility.

        Returns:
            pd.DataFrame: DataFrame containing the generated features.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate random features with different characteristics
        feature_1 = np.random.randn(n) * 10
        feature_2 = np.cumsum(np.random.randn(n))
        feature_3 = np.sin(np.linspace(0, 3 * np.pi, n)) * 20 + np.random.randn(n)
        feature_4 = np.linspace(0, 100, n) + np.random.randn(n) * 5

        features = pd.DataFrame({
            'Feature1': feature_1,
            'Feature2': feature_2,
            'Feature3': feature_3,
            'Feature4': feature_4
        })

        return features

    def generate_target(self, features, random_seed=None, noise_level=2):
        """
        Constructs the target variable using the generated features.

        Args:
            features (pd.DataFrame): DataFrame containing the generated features.
            random_seed (int): Random seed for reproducibility.
            noise_level (float): The standard deviation of the noise.

        Returns:
            pd.Series: The constructed target variable.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n = len(features)
        # Introduce variability in the trend
        trend_start = np.random.uniform(-100, 100)
        trend_direction = np.random.choice([-1, 1])
        trend_magnitude = np.random.uniform(0.5, 1.5)
        trend = trend_start + trend_direction * trend_magnitude * np.linspace(0, 50, n)

        # Introduce variability in the seasonal component
        seasonal_period = np.random.randint(50, 3000)
        seasonal_magnitude = np.random.uniform(5, 20)
        seasonal = seasonal_magnitude * np.sin(np.linspace(0, 2 * np.pi * (n / seasonal_period), n))

        noise = np.random.randn(n) * noise_level

        target = features['Feature1'] * 0.5 + features['Feature2'] * 0.2 + \
                 features['Feature3'] * 0.1 + features['Feature4'] * 0.2 + \
                 trend + seasonal + noise

        return target

    def generate_data(self, random_seed=None, noise_level=2):
        """
        Generates synthetic time series data including features and target variable.

        Args:
            random_seed (int): Random seed for reproducibility.
            noise_level (float): The standard deviation of the noise.

        Returns:
            pd.DataFrame: DataFrame containing the synthetic data.
        """
        n = len(self.ground_truth_dates)
        features = self.generate_features(n, random_seed)
        target = self.generate_target(features, random_seed, noise_level)

        df = self.ground_truth_dates.copy()
        df = pd.concat([df, features], axis=1)
        df['Target'] = target

        return df

    def save_data(self, df, filename='dummy_data.csv'):
        """
        Saves the DataFrame as a CSV file.

        Args:
            df (pd.DataFrame): DataFrame containing the synthetic data.
            filename (str): The name of the output CSV file.
        """
        file_path = os.path.join(self.data_folder, filename)
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def visualize_data(self, df):
        """
        Visualizes the target variable over time.

        Args:
            df (pd.DataFrame): DataFrame containing the synthetic data.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(df['Date'], df['Target'], label='Target')
        plt.title('Target Variable Over Time')
        plt.xlabel('Date')
        plt.ylabel('Target')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    data_folder = 'data/2_raw/'
    ground_truth_file = 'data/1_ground_truth_dates.csv'
    generator = DummyDataGenerator(data_folder, ground_truth_file)
    df_dummy = generator.generate_data(random_seed=1111, noise_level=0.05)
    generator.save_data(df_dummy)
    generator.visualize_data(df_dummy)
