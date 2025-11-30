import pandas as pd
import numpy as np
import os

# --- Configuration ---
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
OUTPUT_FILE = 'taxi_nbody_input.csv'

# NYC Bounding Box (Remove outliers that distort physics calculations)
LAT_MIN, LAT_MAX = 40.50, 40.90
LON_MIN, LON_MAX = -74.25, -73.70

def clean_and_prepare():
    print("--- Phase 1: Data Loading & Cleaning ---")
    
    # 1. Load Data
    cols = ['id', 'pickup_datetime', 'pickup_latitude', 'pickup_longitude']
    
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    if os.path.exists(TRAIN_FILE):
        print(f"Loading {TRAIN_FILE}...")
        df_train = pd.read_csv(TRAIN_FILE, usecols=cols)
    
    if os.path.exists(TEST_FILE):
        print(f"Loading {TEST_FILE}...")
        df_test = pd.read_csv(TEST_FILE, usecols=cols)

    # 2. Merge
    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Total raw records: {len(df)}")

    # 3. Spatial Filtering
    # We remove trips outside NYC because they skew the "Avg Distance" metrics
    df = df[
        (df['pickup_latitude'].between(LAT_MIN, LAT_MAX)) & 
        (df['pickup_longitude'].between(LON_MIN, LON_MAX))
    ]
    
    # 4. Temporal Processing
    print("Converting timestamps to absolute minutes...")
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    min_time = df['pickup_datetime'].min()
    
    # Convert to minutes since start of dataset
    df['time_minutes'] = (df['pickup_datetime'] - min_time).dt.total_seconds() / 60.0
    df['time_minutes'] = df['time_minutes'].astype(int)

    # 5. Sorting (CRITICAL for Windowed N-Body)
    print("Sorting data by time (required for sliding window)...")
    df.sort_values('time_minutes', inplace=True)

    # 6. Export
    # We drop headers to make C parsing simpler and faster
    output_df = df[['id', 'pickup_latitude', 'pickup_longitude', 'time_minutes']]
    
    print(f"Writing {len(output_df)} particles to {OUTPUT_FILE}...")
    output_df.to_csv(OUTPUT_FILE, index=False, header=False)
    print("Done. Ready for C simulation.")

if __name__ == "__main__":
    clean_and_prepare()
