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

        # Delete files in the folder
        for file in os.listdir(self.data_folder):
            os.remove(os.path.join(self.data_folder, file))

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

    def generate_target(self, features, random_seed=None, noise_level=2, mode='feature_dependent', start_value=0, sine_range=6):
        """
        Constructs the target variable using the generated features.

        Args:
            features (pd.DataFrame): DataFrame containing the generated features.
            random_seed (int): Random seed for reproducibility.
            noise_level (float): The standard deviation of the noise.
            mode (str): Mode of target generation - 'feature_dependent', 'feature_independent', 'sine_feature'.
            start_value (float): Starting value for the trend.
            sine_range (float): Range for sine cycles.

        Returns:
            pd.Series: The constructed target variable.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n = len(features)
        # Generate trend and seasonal components
        trend = start_value + np.linspace(np.random.uniform(-100, 100), np.random.uniform(-100, 100), n)
        seasonal = 10 * np.sin(np.linspace(0, sine_range * np.pi, n))
        noise = noise_level

        if mode == 'feature_dependent':
            target = features['Feature1'] * np.random.uniform(0.1, 0.5) + \
                        features['Feature2'] * np.random.uniform(0.1, 0.5) + \
                        features['Feature3'] * np.random.uniform(0.1, 0.5) + \
                        features['Feature4'] * np.random.uniform(0.1, 0.5) + \
                     trend + seasonal + noise
        elif mode == 'feature_independent':
            target = trend + seasonal + noise
        else:
            raise ValueError("Mode must be 'feature_dependent' or 'feature_independent'")

        return target

    def generate_data(self, ticker, random_seed=None, noise_level=2, mode='feature_dependent', start_value=0, sine_range=6):
        """
        Generates synthetic time series data including features and target variable.

        Args:
            ticker (str): The ticker symbol for the generated data.
            random_seed (int): Random seed for reproducibility.
            noise_level (float): The standard deviation of the noise.
            mode (str): Mode of target generation - 'feature_dependent', 'feature_independent', 'sine_feature'.
            start_value (float): Starting value for the trend.
            sine_range (float): Range for sine cycles.

        Returns:
            pd.DataFrame: DataFrame containing the synthetic data.
        """
        n = len(self.ground_truth_dates)
        features = self.generate_features(n, random_seed)
        target = self.generate_target(features, random_seed, noise_level, mode, start_value, sine_range)

        df = self.ground_truth_dates[['Date']].copy()
        df = pd.concat([df, features], axis=1)
        df['Target'] = target
        df['Ticker'] = ticker

        return df

    def save_data(self, df, filename):
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
    data_folder = os.path.join('data', '2_raw_dummy')
    ground_truth_file = os.path.join('data', '1_ground_truth_dates.csv')
    tickers = ['DUMMY1', 'DUMMY2', 'DUMMY3']
    random_seeds = [42, 123, 456]

    generator = DummyDataGenerator(data_folder, ground_truth_file)

    mode = 'feature_dependent'  # Change mode as needed: 'feature_dependent', 'feature_independent'
    for ticker, seed in zip(tickers, random_seeds):
        sine_range = np.random.uniform(20, 100)  # Randomize sine range for each ticker
        start_value = np.random.uniform(-100, 100)  # Randomize start value for each ticker
        df_dummy = generator.generate_data(ticker, random_seed=seed, noise_level=0, mode=mode, start_value=start_value, sine_range=sine_range)
        filename = f'{ticker}_{mode}.csv'
        generator.save_data(df_dummy, filename)
        generator.visualize_data(df_dummy)
