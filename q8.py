"""
Flight Delay Analysis - Question 8: Operational Efficiency & Turnaround Times

Research Question 8:
  How do taxi out and taxi in times affect overall flight duration and delays?

This script results in:
  1. Scatter plots of Taxi Out vs. Ground Time and Taxi In vs. Ground Time.
  2. Bar charts of top 10 origin airports by average Taxi Out and top 10 destination airports by average Taxi In.
  3. Correlation heatmap among Taxi Out, Taxi In, Ground Time, Departure Delay, and Arrival Delay.
  4. Summary statistics printed to console.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------
# 
# ---------------------------
def load_data(file_path, nrows=None):
    """
    Load the dataset from CSV.
    """
    df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
    logging.info(f"Loaded {len(df)} records from {file_path}")
    return df

def preprocess_turnaround(df):
    """
    Convert relevant columns to numeric and impute missing values with the column mean.
    Compute GROUND_TIME = ELAPSED_TIME - AIR_TIME.
    """
    cols = ['TAXI_OUT', 'TAXI_IN', 'ELAPSED_TIME', 'AIR_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())
    # Compute ground time (total taxi time)
    df['GROUND_TIME'] = df['ELAPSED_TIME'] - df['AIR_TIME']
    return df

# ---------------------------
# 
# ---------------------------
def analyze_turnaround(df):
    """
    Generate and save plots and summary statistics for turnaround analysis.
    """
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'q8')
    os.makedirs(results_dir, exist_ok=True)

    # 1) Scatter: Taxi Out vs. Ground Time
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='TAXI_OUT', y='GROUND_TIME', alpha=0.3)
    plt.title("Taxi Out vs. Ground Time")
    plt.xlabel("Taxi Out (min)")
    plt.ylabel("Ground Time (Elapsed - Air) (min)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'taxi_out_vs_ground_time.png'))
    plt.close()
    logging.info("Saved scatter: taxi_out_vs_ground_time.png")

    # 2) Scatter: Taxi In vs. Ground Time
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='TAXI_IN', y='GROUND_TIME', alpha=0.3)
    plt.title("Taxi In vs. Ground Time")
    plt.xlabel("Taxi In (min)")
    plt.ylabel("Ground Time (Elapsed - Air) (min)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'taxi_in_vs_ground_time.png'))
    plt.close()
    logging.info("Saved scatter: taxi_in_vs_ground_time.png")

    # 3) Top 10 Origin Airports by Avg Taxi Out
    orig_taxi = (
        df.groupby('ORIGIN_AIRPORT')['TAXI_OUT']
          .mean()
          .nlargest(10)
          .reset_index(name='Avg_Taxi_Out')
    )
    plt.figure(figsize=(10,6))
    sns.barplot(data=orig_taxi, x='ORIGIN_AIRPORT', y='Avg_Taxi_Out')
    plt.title("Top 10 Origin Airports by Avg Taxi Out")
    plt.xlabel("Origin Airport")
    plt.ylabel("Avg Taxi Out (min)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'top10_origin_taxi_out.png'))
    plt.close()
    logging.info("Saved bar: top10_origin_taxi_out.png")

    # 4) Top 10 Destination Airports by Avg Taxi In
    dest_taxi = (
        df.groupby('DESTINATION_AIRPORT')['TAXI_IN']
          .mean()
          .nlargest(10)
          .reset_index(name='Avg_Taxi_In')
    )
    plt.figure(figsize=(10,6))
    sns.barplot(data=dest_taxi, x='DESTINATION_AIRPORT', y='Avg_Taxi_In')
    plt.title("Top 10 Destination Airports by Avg Taxi In")
    plt.xlabel("Destination Airport")
    plt.ylabel("Avg Taxi In (min)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'top10_dest_taxi_in.png'))
    plt.close()
    logging.info("Saved bar: top10_dest_taxi_in.png")

    # 5) Correlation Heatmap
    corr_cols = ['TAXI_OUT','TAXI_IN','GROUND_TIME','DEPARTURE_DELAY','ARRIVAL_DELAY']
    corr = df[corr_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
    plt.title("Correlation: Taxi Times, Ground Time & Delays")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'corr_turnaround.png'))
    plt.close()
    logging.info("Saved heatmap: corr_turnaround.png")

    # 6) Summary Statistics
    print("\nSummary Statistics for Turnaround Metrics:")
    print(df[corr_cols].describe())

def main():
    # Path to dataset
    data_path = os.path.join(os.path.dirname(__file__), 'flights.csv')

    # Load, preprocess, analyze
    df = load_data(data_path)
    df = preprocess_turnaround(df)
    analyze_turnaround(df)

if __name__ == "__main__":
    main()
