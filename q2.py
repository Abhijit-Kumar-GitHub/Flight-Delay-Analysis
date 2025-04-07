

"""
Project: Flight Delay Analysis – Airline Reliability & Performance

This script addresses the research question:
    "Which airlines consistently perform better in terms of on-time departures and arrivals?"
It analyzes airline performance by:
    - Calculating average departure and arrival delays.
    - Computing cancellation rates.
    - Visualizing these metrics to compare airlines.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# 
# ---------------------------
def load_data(file_path, nrows=None):
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        nrows (int, optional): Number of rows to read (for testing).
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        logging.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# ---------------------------
# 
# ---------------------------
def preprocess_data(df):
    """
    Preprocess the flight dataset:
      - Convert delay columns to numeric and impute missing values with column mean.
      - Convert CANCELLED to numeric (if needed).
      
    Args:
        df (pd.DataFrame): Raw dataset.
    
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Columns to process for delays
    delay_cols = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT', 'TAXI_IN']
    for col in delay_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
    
    # Ensure CANCELLED column is numeric (assuming 0 or 1)
    df['CANCELLED'] = pd.to_numeric(df['CANCELLED'], errors='coerce')
    # Impute CANCELLED missing values with 0 (assuming missing means not cancelled)
    df['CANCELLED'].fillna(0, inplace=True)
    
    return df

# ---------------------------
# 
# ---------------------------
def plot_bar(data, x, y, title, xlabel, ylabel, output_filename):
    """
    Create and save a bar chart.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        x (str): Column name for x-axis.
        y (str): Column name for y-axis.
        title (str): Plot title.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
        output_filename (str): File path to save the plot.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x=x, y=y, data=data, palette="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    logging.info(f"Bar chart saved: {output_filename}")

def analyze_airline_performance(df, results_subdir):
    """
    Analyze airline performance:
      - Compute average departure and arrival delays per airline.
      - Compute cancellation rates per airline.
      - Visualize these metrics with bar charts.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset.
        results_subdir (str): Directory to save the results.
    """
    os.makedirs(results_subdir, exist_ok=True)
    
    # Group data by AIRLINE and calculate performance metrics
    airline_perf = df.groupby('AIRLINE').agg({
        'DEPARTURE_DELAY': 'mean',
        'ARRIVAL_DELAY': 'mean',
        'CANCELLED': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    airline_perf.rename(columns={
        'DEPARTURE_DELAY': 'Avg_Departure_Delay',
        'ARRIVAL_DELAY': 'Avg_Arrival_Delay',
        'CANCELLED': 'Cancellation_Rate'
    }, inplace=True)
    
    logging.info("Airline performance metrics computed.")
    print("\nAirline Performance Metrics:")
    print(airline_perf)
    
    # Plot average departure delay per airline
    plot_bar(
        airline_perf, x='AIRLINE', y='Avg_Departure_Delay',
        title="Average Departure Delay by Airline",
        xlabel="Airline",
        ylabel="Average Departure Delay (min)",
        output_filename=os.path.join(results_subdir, 'avg_departure_delay_by_airline.png')
    )
    
    # Plot average arrival delay per airline
    plot_bar(
        airline_perf, x='AIRLINE', y='Avg_Arrival_Delay',
        title="Average Arrival Delay by Airline",
        xlabel="Airline",
        ylabel="Average Arrival Delay (min)",
        output_filename=os.path.join(results_subdir, 'avg_arrival_delay_by_airline.png')
    )
    
    # Plot cancellation rate per airline (as a percentage)
    airline_perf['Cancellation_Rate_Percent'] = airline_perf['Cancellation_Rate'] * 100
    plot_bar(
        airline_perf, x='AIRLINE', y='Cancellation_Rate_Percent',
        title="Cancellation Rate by Airline",
        xlabel="Airline",
        ylabel="Cancellation Rate (%)",
        output_filename=os.path.join(results_subdir, 'cancellation_rate_by_airline.png')
    )

# ---------------------------
# 
# ---------------------------
def perform_statistical_analysis_airlines(df):
    """
    Compute and print summary statistics for airline performance metrics.
    
    Args:
        df (pd.DataFrame): DataFrame containing airline performance metrics.
    """
    airline_stats = df.describe()
    logging.info("Airline Performance Summary Statistics:\n" + airline_stats.to_string())
    print("\nAirline Performance Summary Statistics:")
    print(airline_stats)

# ---------------------------
# Main Execution
# ---------------------------
def main():
    # Define directories
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'results', 'q2')
    os.makedirs(results_dir, exist_ok=True)
    
    # Path to the dataset
    data_file = os.path.join(base_dir, 'flights.csv')
    
    # Load data
    df = load_data(data_file)
    
    # Preprocess the data: convert numeric columns and impute missing values
    df = preprocess_data(df)
    
    # Analyze airline performance
    analyze_airline_performance(df, results_dir)
    
    # Pefrorming additional statistical analysis on airline metrics
    # First, computing the performance metrics per airline
    airline_perf = df.groupby('AIRLINE').agg({
        'DEPARTURE_DELAY': 'mean',
        'ARRIVAL_DELAY': 'mean',
        'CANCELLED': 'mean'
    }).reset_index()
    airline_perf.rename(columns={
        'DEPARTURE_DELAY': 'Avg_Departure_Delay',
        'ARRIVAL_DELAY': 'Avg_Arrival_Delay',
        'CANCELLED': 'Cancellation_Rate'
    }, inplace=True)
    
    perform_statistical_analysis_airlines(airline_perf)
    
    logging.info("Question 2 analysis complete. Check the 'results/q2' folder for all plots.")

if __name__ == "__main__":
    main()
