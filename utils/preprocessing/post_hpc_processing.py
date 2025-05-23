#### This script combines the cropped healthy csvs into one csv, and then identifies the minimum size of image needed

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Paths to the cropped healthy csvs
csvs_path = '/path/to/cropped/csvs'  # <-- CHANGE THIS TO YOUR DIRECTORY

# Output path for the combined csv
output_path = '/path/to/output/combined_crops.csv'  # <-- CHANGE THIS TO YOUR OUTPUT FILE

# Get all the csvs
csvs = os.listdir(csvs_path)

# Initialize the dataframe
combined_df = None

# Iterate through all the csvs
for csv in tqdm(csvs, desc='Combining CSVs'):
    # Load the csv
    df = pd.read_csv(os.path.join(csvs_path, csv))
    
    if combined_df is None:
        combined_df = df
        continue

    # Append to the combined dataframe
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Save the combined dataframe
combined_df.to_csv(output_path, index=False)

# Get the minimum size of the image
min_x = combined_df['end_x'].max() - combined_df['start_x'].min() 
min_y = combined_df['end_y'].max() - combined_df['start_y'].min()
min_z = combined_df['end_z'].max() - combined_df['start_z'].min()

print(f"Minimum size of image needed: {min_x} x {min_y} x {min_z}")

#write it out
with open('/path/to/output/min_size.txt', 'w') as f:  # <-- CHANGE THIS TO YOUR OUTPUT FILE
    f.write(f"Minimum size of image needed: {min_x} x {min_y} x {min_z}")