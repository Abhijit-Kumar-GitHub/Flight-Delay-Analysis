"""
Project: Flight Delay Analysis – Delay Propagation & Causality

This script answers the research question:
    "How do departure delays impact arrival delays?"
It performs analysis on the raw dataset and then repeats the analysis after removing outliers.

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#####################

def load_data(file_path, nrows=None):
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path (str): The file path to the CSV data.
        nrows (int, optional): Number of rows to read (for testing).
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        logging.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logging.error("Error loading data: " + str(e))
        raise

def convert_time_format(time_val):
    """
    Convert a time value in HHMM format into a 'HH:MM' string.
    
    Args:
        time_val (int or str): Time in HHMM format.
    
    Returns:
        str: Time formatted as 'HH:MM', or None if conversion fails.
    """
    try:
        if pd.isna(time_val):
            return None
        time_str = f"{int(time_val):04d}"  # Ensure 4-digit string with leading zeros
        return f"{time_str[:2]}:{time_str[2:]}"
    except Exception as e:
        logging.warning(f"Time conversion error for {time_val}: {e}")
        return None

# ---------------------------
#
# ---------------------------
def preprocess_delays(df):
    """
    Convert delay and taxi time columns to numeric values and impute missing values with the column mean.
    Also converts scheduled departure time to a readable string.
    
    Args:
        df (pd.DataFrame): Raw dataset.
        
    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    cols = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT', 'TAXI_IN']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
    df['SCHEDULED_DEPARTURE_STR'] = df['SCHEDULED_DEPARTURE'].apply(convert_time_format)
    return df

def remove_outliers(df, cols, factor=1.5):
    """
    Remove outliers from specified columns using the IQR method.
    
    Args:
        df (pd.DataFrame): DataFrame to process.
        cols (list): List of columns to remove outliers from.
        factor (float): IQR multiplier (default=1.5).
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        original_count = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        logging.info(f"Removed outliers from {col}: {original_count - len(df_clean)} rows dropped")
    return df_clean

# ---------------------------
# 
# ---------------------------
def plot_scatter(df, x_col, y_col, title, xlabel, ylabel, output_filename):
    """
    Create and save a scatter plot for the given columns.
    
    Args:
        df (pd.DataFrame): DataFrame with data.
        x_col (str): Column for the x-axis.
        y_col (str): Column for the y-axis.
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        output_filename (str): File path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    logging.info(f"Plot saved: {output_filename}")

def plot_boxplots(df, cols, output_filename):
    """
    Create and save boxplots for the specified columns to visualize outliers.
    
    Args:
        df (pd.DataFrame): DataFrame containing data.
        cols (list): List of column names for which to create boxplots.
        output_filename (str): File path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[cols])
    plt.title("Boxplots for Delay Variables")
    plt.xlabel("Delay Variables")
    plt.ylabel("Minutes")
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    logging.info(f"Boxplot saved: {output_filename}")

def analyze_delay_propagation(df, results_subdir):
    """
    Generate visualizations and statistics to analyze delay propagation.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset.
        results_subdir (str): Directory to save the resulting plots.
    """
    # Ensure results_subdir exists
    os.makedirs(results_subdir, exist_ok=True)
    
    # Scatter Plot: Departure Delay vs. Arrival Delay
    plot_scatter(
        df, 'DEPARTURE_DELAY', 'ARRIVAL_DELAY',
        title="Departure Delay vs. Arrival Delay",
        xlabel="Departure Delay (min)",
        ylabel="Arrival Delay (min)",
        output_filename=os.path.join(results_subdir, 'departure_vs_arrival_delay.png')
    )
    
    # Compute and log correlation between departure and arrival delays
    corr_delay = df['DEPARTURE_DELAY'].corr(df['ARRIVAL_DELAY'])
    logging.info(f"Correlation (Departure vs. Arrival Delay): {corr_delay:.3f}")
    
    # Scatter Plot: Taxi Out vs. Departure Delay
    plot_scatter(
        df, 'TAXI_OUT', 'DEPARTURE_DELAY',
        title="Taxi Out Time vs. Departure Delay",
        xlabel="Taxi Out (min)",
        ylabel="Departure Delay (min)",
        output_filename=os.path.join(results_subdir, 'taxi_out_vs_departure_delay.png')
    )
    
    # Scatter Plot: Taxi In vs. Arrival Delay
    plot_scatter(
        df, 'TAXI_IN', 'ARRIVAL_DELAY',
        title="Taxi In Time vs. Arrival Delay",
        xlabel="Taxi In (min)",
        ylabel="Arrival Delay (min)",
        output_filename=os.path.join(results_subdir, 'taxi_in_vs_arrival_delay.png')
    )
    
    # Boxplots for delay variables
    boxplot_file = os.path.join(results_subdir, 'boxplots_delay_variables.png')
    plot_boxplots(df, ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT', 'TAXI_IN'], boxplot_file)
    
    # Print summary statistics for key delay columns
    summary_stats = df[['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT', 'TAXI_IN']].describe()
    logging.info("Summary Statistics for Delay Variables:\n" + summary_stats.to_string())
    print("\nSummary Statistics for Delay Variables:")
    print(summary_stats)

# ---------------------------
# 
# ---------------------------
def perform_statistical_analysis(df):
    """
    Compute and print the correlation matrix for delay-related variables.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.
    """
    cols = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT', 'TAXI_IN']
    corr_matrix = df[cols].corr()
    logging.info("Correlation Matrix:\n" + corr_matrix.to_string())
    print("\nCorrelation Matrix:")
    print(corr_matrix)

# ---------------------------
# Main Execution
# ---------------------------
def main():
    # Define directories
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'results', 'q1')
    os.makedirs(results_dir, exist_ok=True)
    
    # Path to dataset (adjust as necessary)
    data_file = os.path.join(base_dir, 'flights.csv')
    
    # Load data     df = load_data(data_file)
    
    # Preprocess data: convert numeric columns and impute missing values
    df = preprocess_delays(df)
    
    # Analysis Including Outliers
    print("\n--- Analysis Including Outliers ---")
    analyze_delay_propagation(df, results_dir)
    perform_statistical_analysis(df)
    
    # Remove outliers based on DEPARTURE_DELAY and ARRIVAL_DELAY
    df_no_outliers = remove_outliers(df, ['DEPARTURE_DELAY', 'ARRIVAL_DELAY'])
    
    # Define subdirectory for no-outlier analysis
    no_outliers_dir = os.path.join(results_dir, 'no_outliers')
    os.makedirs(no_outliers_dir, exist_ok=True)
    
    # Analysis After Removing Outliers
    print("\n--- Analysis After Removing Outliers ---")
    analyze_delay_propagation(df_no_outliers, no_outliers_dir)
    perform_statistical_analysis(df_no_outliers)
    
    logging.info("Question 1 analysis complete. Check the 'results/q1' folder for all plots.")

if __name__ == "__main__":
    main()
