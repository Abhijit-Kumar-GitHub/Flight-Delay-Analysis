"""
Flight Delay Analysis  Question 7: Cancellation & Diversion Analysis

Research Q7:
  What factors most frequently lead to flight cancellations or diversions,
  and how do these events correlate with delay patterns?

This script gives insights into:
  1. Bar chart of cancellation reasons (% of cancellations).
  2. Bar charts of average arrival delay for cancelled vs. non-cancelled and diverted vs. non-diverted flights.
  3. Heatmap of correlation between CANCELLED, DIVERTED, and delay columns.
  4. Console output of the underlying tables.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# ---------------------------
# 
# ---------------------------
def load_data(file_path, nrows=None):
    df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
    logging.info(f"Loaded {len(df)} records.")
    return df

# ---------------------------
# 
# ---------------------------
def preprocess_cancellation_diversion(df):
    """
    Convert CANCELLED and DIVERTED to numeric flags,
    and fill missing CANCELLATION_REASON with 'None'.
    """
    df['CANCELLED'] = pd.to_numeric(df['CANCELLED'], errors='coerce').fillna(0).astype(int)
    df['DIVERTED']  = pd.to_numeric(df['DIVERTED'],  errors='coerce').fillna(0).astype(int)
    df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].fillna('None')
    return df

# ---------------------------
# 
# ---------------------------
def analyze_cancellation_diversion(df):
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'q7')
    os.makedirs(results_dir, exist_ok=True)

    # 1) Cancellation Reasons Distribution
    reasons = (
        df[df['CANCELLED']==1]['CANCELLATION_REASON']
          .value_counts(normalize=True) * 100
    ).rename_axis('Reason').reset_index(name='Percent')
    plt.figure(figsize=(8, 6))
    sns.barplot(data=reasons, x='Reason', y='Percent')
    plt.title('Cancellation Reasons (% of Cancellations)')
    plt.xlabel('Reason (A=Airline, B=Weather, C=NAS, D=Security)')  # :contentReference[oaicite:1]{index=1}
    plt.ylabel('Percent of Cancellations')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cancellation_reasons.png'))
    plt.close()

    # 2) Avg Arrival Delay: Cancelled vs Non‑Cancelled
    canc_delay = (
        df.groupby('CANCELLED')[['ARRIVAL_DELAY']]
          .mean()
          .reset_index()
    )
    plt.figure(figsize=(6, 5))
    sns.barplot(data=canc_delay, x='CANCELLED', y='ARRIVAL_DELAY')
    plt.title('Avg Arrival Delay by Cancellation Status')
    plt.xlabel('Cancelled (0=No, 1=Yes)')
    plt.ylabel('Avg Arrival Delay (min)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'arrival_delay_by_cancellation.png'))
    plt.close()

    # 3) Avg Arrival Delay: Diverted vs Non‑Diverted
    div_delay = (
        df.groupby('DIVERTED')[['ARRIVAL_DELAY']]
          .mean()
          .reset_index()
    )
    plt.figure(figsize=(6, 5))
    sns.barplot(data=div_delay, x='DIVERTED', y='ARRIVAL_DELAY')
    plt.title('Avg Arrival Delay by Diversion Status')
    plt.xlabel('Diverted (0=No, 1=Yes)')
    plt.ylabel('Avg Arrival Delay (min)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'arrival_delay_by_diversion.png'))
    plt.close()

    # 4) Correlation Matrix
    corr = df[['CANCELLED', 'DIVERTED', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']].corr()
    print("\nCorrelation Matrix:")
    print(corr)
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title('Correlation: Cancellation/Diversion & Delays')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'corr_cancellation_diversion.png'))
    plt.close()

    # 5) Console Summaries
    print("\nCancellation Reasons (% of all cancellations):")
    print(reasons.to_string(index=False))
    print("\nAvg Arrival Delay by Cancellation Status:")
    print(canc_delay.to_string(index=False))
    print("\nAvg Arrival Delay by Diversion Status:")
    print(div_delay.to_string(index=False))

# ---------------------------
# 
# ---------------------------
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    data_path = os.path.join(os.path.dirname(__file__), 'flights.csv')
    df = load_data(data_path)
    df = preprocess_cancellation_diversion(df)
    analyze_cancellation_diversion(df)

if __name__ == "__main__":
    main()
