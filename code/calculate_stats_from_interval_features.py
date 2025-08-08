import pandas as pd
import numpy as np

# Replace 'your_file.csv' with your actual CSV file path
df = pd.read_csv('/home/lucky/PycharmProjects/Damona/data/biotrial_results/features/neural_features_w_intervals_biotrial.csv')


# Columns to sum
sum_columns = ['sleep', 'BAD', 'wake', 'sleep_wake_transitions']

# Replace empty strings in all columns with 0
df = df.replace('', 0)

# Convert all columns to float type
df = df.astype(float, errors='ignore')

# Compute the sum for specified columns and mean for the others,
# for each unique combination of condition, rat_id, and stage
grouped = df.groupby(['condition', 'rat_id', 'stage'])

sum_df = grouped[sum_columns].sum()
mean_df = grouped[df.columns.difference(['condition', 'rat_id', 'stage'] + sum_columns)].mean()

# Combine the results
result = pd.concat([sum_df, mean_df], axis=1).reset_index()
result.to_csv('result.csv', index=False)