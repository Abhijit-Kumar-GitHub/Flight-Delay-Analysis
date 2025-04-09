
"""
question5_delay_attribution.py

Project: Flight Delay Analysis - Delay Attribution

This script answers:
  "What are the relative contributions of different delay causes 
   (air system, security, airline, late aircraft, weather) to the overall delay time?"
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path, nrows=None):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
    print(f"Loaded {len(df)} records.")
    return df

def preprocess_delay_causes(df):
    """
    Convert delay-cause columns to numeric and fill missing with zero.
    """
    cause_cols = [
        'AIR_SYSTEM_DELAY',
        'SECURITY_DELAY',
        'AIRLINE_DELAY',
        'LATE_AIRCRAFT_DELAY',
        'WEATHER_DELAY'
    ]
    for col in cause_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def analyze_delay_attribution(df):
    """
    Compute and visualize the relative contributions of each delay cause,
    and plot a boxplot of their distributions.
    """
    # Prepare results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results', 'q5')
    os.makedirs(results_dir, exist_ok=True)

    # Define cause columns
    cause_cols = [
        'AIR_SYSTEM_DELAY',
        'SECURITY_DELAY',
        'AIRLINE_DELAY',
        'LATE_AIRCRAFT_DELAY',
        'WEATHER_DELAY'
    ]

    # 1) Bar & Pie charts of percentage attribution
    total_by_cause = df[cause_cols].sum()
    total_delay = total_by_cause.sum()
    percent = (total_by_cause / total_delay * 100).sort_values(ascending=False)
    attribution = percent.reset_index()
    attribution.columns = ['Cause', 'Percentage']

    # Bar chart
    plt.figure(figsize=(8, 6))
    sns.barplot(data=attribution, x='Cause', y='Percentage', alpha=0.8)
    plt.title('Delay Attribution by Cause')
    plt.ylabel('Percentage of Total Delay (%)')
    plt.xlabel('Delay Cause')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'delay_attribution_bar.png'))
    plt.close()

    # Pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        attribution['Percentage'],
        labels=attribution['Cause'],
        autopct='%1.1f%%',
        startangle=140
    )
    plt.title('Delay Attribution by Cause')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'delay_attribution_pie.png'))
    plt.close()

    # 2) Boxplot of distributions for each cause
    # Melt DataFrame into long format for seaborn
    df_melted = df[cause_cols].melt(var_name='Cause', value_name='Delay')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cause', y='Delay', data=df_melted)  # :contentReference[oaicite:0]{index=0}
    plt.title('Distribution of Delay Causes (Boxplot)')
    plt.xlabel('Delay Cause')
    plt.ylabel('Delay Minutes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'delay_causes_boxplot.png'))
    plt.close()

    # Print results
    print("\nDelay Attribution (% of total delay):")
    print(attribution.to_string(index=False))
    print(f"\nTotal delay minutes across all flights: {total_delay:.0f}")

def main():
    # Path to your dataset
    data_path = os.path.join(os.path.dirname(__file__), 'flights.csv')

    # Load, preprocess, and analyze
    df = load_data(data_path)
    df = preprocess_delay_causes(df)
    analyze_delay_attribution(df)

if __name__ == "__main__":
    main()
