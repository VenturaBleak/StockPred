# 1_create_date_ground_truth.py
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import os

# Define the date range
start_date = '1990-01-01'
end_date = '2024-05-01'

# Get the NYSE calendar
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
date_range = schedule.index

# Create a DataFrame for the date range
date_df = pd.DataFrame(date_range, columns=['Date'])

# Add DateNumbered feature
date_df['DateNumbered'] = date_df['Date'].apply(lambda x: (x - pd.Timestamp('1900-01-01')).days + 1)

# Add additional time features
date_df['DayOfYear'] = date_df['Date'].dt.dayofyear
date_df['Month'] = date_df['Date'].dt.month
date_df['Year'] = date_df['Date'].dt.year
date_df['Weekday'] = date_df['Date'].dt.weekday + 1
date_df['WeekOfYear'] = date_df['Date'].dt.isocalendar().week
date_df['WeekOfMonth'] = date_df['Date'].apply(lambda x: (x.day - 1) // 7 + 1)
date_df['Quarter'] = date_df['Date'].dt.quarter

# is ultimo, i.e. last trading day of the month -> that is the day before the first day of the next month


# Add cyclical features
date_df['Sin_DayOfYear'] = np.sin(2 * np.pi * date_df['DayOfYear'] / 365)
date_df['Cos_DayOfYear'] = np.cos(2 * np.pi * date_df['DayOfYear'] / 365)
date_df['Sin_Weekday'] = np.sin(2 * np.pi * date_df['Weekday'] / 5)
date_df['Cos_Weekday'] = np.cos(2 * np.pi * date_df['Weekday'] / 5)
date_df['Sin_WeekOfYear'] = np.sin(2 * np.pi * date_df['WeekOfYear'] / 52)
date_df['Cos_WeekOfYear'] = np.cos(2 * np.pi * date_df['WeekOfYear'] / 52)
date_df['Sin_Month'] = np.sin(2 * np.pi * date_df['Month'] / 12)
date_df['Cos_Month'] = np.cos(2 * np.pi * date_df['Month'] / 12)
date_df['Sin_WeekOfMonth'] = np.sin(2 * np.pi * date_df['WeekOfMonth'] / 4.45)
date_df['Cos_WeekOfMonth'] = np.cos(2 * np.pi * date_df['WeekOfMonth'] / 4.45)
date_df['Sin_Quarter'] = np.sin(2 * np.pi * date_df['Quarter'] / 4)
date_df['Cos_Quarter'] = np.cos(2 * np.pi * date_df['Quarter'] / 4)

# Save the ground truth date array
output_folder = 'data'
os.makedirs(output_folder, exist_ok=True)
date_df.to_csv(os.path.join(output_folder, '1_ground_truth_dates.csv'), index=False)

print("Ground truth dates created.")