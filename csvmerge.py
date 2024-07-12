import pandas as pd
import os

# List of input CSV file names
file_names = [f'./Anvith/csv/June_{i}.csv' for i in range(1, 18)]

# Initialize an empty DataFrame to hold the combined data
combined_df = pd.DataFrame()

# Iterate over each file
for file_name in file_names:
    # Read the current file into a DataFrame
    df = pd.read_csv(file_name)

    # Rename the 'rainfall' column to the file name (without the .csv extension)
    df.rename(columns={'Rainfall': file_name[:-4]}, inplace=True)

    if combined_df.empty:
        # For the first file, initialize the combined_df with the current df
        combined_df = df
    else:
        # Merge the current df with the combined_df on 'latitude' and 'longitude'
        combined_df = pd.merge(combined_df, df, on=['Longitude', 'Latitude'], how='outer')

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('June_1-17.csv', index=False)

print("Files combined successfully into combined_rainfall.csv")
