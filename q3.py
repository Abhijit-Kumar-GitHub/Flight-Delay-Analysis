"""
Project: Flight Delay Analysis : Temporal Patterns & Seasonality

This script addresses the research question:
    "How do flight delays vary by time of day, day of the week, and month?"
Additionally, it focuses on:
    - Comparing delays on Sundays vs. workdays.
    - Comparing weekdays vs. weekends.
    - Analyzing delays by season (winter, spring, summer, fall).
    - Examining the impact of major American festivals (e.g., New Year, Good Friday, Easter, 4th of July, Christmas).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

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
        nrows (int, optional): Number of rows to load (for testing).
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        logging.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_temporal_features(df):
    """
    Preprocess the dataset by converting necessary columns to numeric,
    imputing missing values, and extracting temporal features:
      - DEP_HOUR from SCHEDULED_DEPARTURE (assumed in HHMM format).
      - FLIGHT_DATE as a datetime from YEAR, MONTH, DAY.
      - Season based on MONTH.
      - Holiday flag based on known 2015 festival dates.
    
    Args:
        df (pd.DataFrame): Raw dataset.
        
    Returns:
        pd.DataFrame: DataFrame with added temporal features.
    """
    # Convert columns to numeric and impute missing delays with mean
    delay_cols = ['DEPARTURE_DELAY']
    for col in delay_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
    
    # Process SCHEDULED_DEPARTURE to extract hour
    df['SCHEDULED_DEPARTURE'] = pd.to_numeric(df['SCHEDULED_DEPARTURE'], errors='coerce')
    df['DEP_HOUR'] = df['SCHEDULED_DEPARTURE'].apply(lambda x: int(f"{int(x):04d}"[:2]) if pd.notna(x) else None)
    
    # Ensure DAY, MONTH, DAY_OF_WEEK are numeric
    df['DAY'] = pd.to_numeric(df['DAY'], errors='coerce')
    df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce')
    df['DAY_OF_WEEK'] = pd.to_numeric(df['DAY_OF_WEEK'], errors='coerce')
    
    # Create a FLIGHT_DATE column (assuming YEAR, MONTH, DAY exist)
    df['FLIGHT_DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    
    # Extract Season: Winter (Dec, Jan, Feb), Spring (Mar, Apr, May),
    # Summer (Jun, Jul, Aug), Fall (Sep, Oct, Nov)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Unknown'
    df['Season'] = df['MONTH'].apply(get_season)
    
    # Flag major American festivals (for 2015)
    # Define known festival dates in 2015 as strings in 'YYYY-MM-DD' format
    festival_dates = {
        '2015-01-01': 'New Year',
        '2015-04-03': 'Good Friday',
        '2015-04-05': 'Easter Sunday',
        '2015-07-04': '4th of July',
        '2015-11-26': 'Thanksgiving',
        '2015-12-25': 'Christmas'
    }
    df['Holiday'] = df['FLIGHT_DATE'].dt.strftime('%Y-%m-%d').map(festival_dates)
    # Mark non-festival days as "No"
    df['Holiday'].fillna('No', inplace=True)
    
    # Flag weekdays vs. weekends (assuming Monday=1, Sunday=7)
    df['Is_Weekend'] = df['DAY_OF_WEEK'].apply(lambda x: 'Weekend' if x in [6, 7] else 'Weekday')
    # Also create a flag for Sundays specifically (assuming Sunday is 7)
    df['Is_Sunday'] = df['DAY_OF_WEEK'].apply(lambda x: 'Sunday' if x == 7 else 'Not Sunday')
    
    return df

# ---------------------------
# 
# ---------------------------
def plot_temporal_trends(df, group_col, delay_col, title, xlabel, ylabel, output_filename, kind='line'):
    """
    Group the data by a temporal feature and plot the average delay.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        group_col (str): Column name to group by.
        delay_col (str): Column name for the delay metric.
        title (str): Plot title.
        xlabel (str): x-axis label.
        ylabel (str): y-axis label.
        output_filename (str): Path to save the plot.
        kind (str): Plot type ('line' or 'bar').
    """
    agg_df = df.groupby(group_col)[delay_col].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    if kind == 'line':
        sns.lineplot(data=agg_df, x=group_col, y=delay_col, marker='o')
    elif kind == 'bar':
        sns.barplot(data=agg_df, x=group_col, y=delay_col, palette="viridis")
    else:
        raise ValueError("Unsupported plot type. Use 'line' or 'bar'.")
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    logging.info(f"Plot saved: {output_filename}")

def analyze_temporal_patterns(df, results_subdir):
    """
    Analyze temporal patterns in flight delays, focusing on:
        - Average departure delay by departure hour.
        - Average departure delay by day of week (specifically Sunday vs. workdays, weekdays vs. weekends).
        - Average departure delay by season.
        - Average departure delay on festival days vs. non-festival days.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with temporal features.
        results_subdir (str): Directory to save plots.
    """
    os.makedirs(results_subdir, exist_ok=True)
    
    # Plot average departure delay by departure hour
    plot_temporal_trends(
        df, group_col='DEP_HOUR', delay_col='DEPARTURE_DELAY',
        title="Average Departure Delay by Hour",
        xlabel="Scheduled Departure Hour", ylabel="Avg Departure Delay (min)",
        output_filename=os.path.join(results_subdir, 'avg_departure_delay_by_hour.png'),
        kind='line'
    )
    
    # Plot average departure delay by day of week (Weekday vs Weekend)
    plot_temporal_trends(
        df, group_col='Is_Weekend', delay_col='DEPARTURE_DELAY',
        title="Average Departure Delay: Weekdays vs. Weekends",
        xlabel="Day Type", ylabel="Avg Departure Delay (min)",
        output_filename=os.path.join(results_subdir, 'avg_departure_delay_by_weekend.png'),
        kind='bar'
    )
    
    # Plot average departure delay for Sundays vs. other days
    plot_temporal_trends(
        df, group_col='Is_Sunday', delay_col='DEPARTURE_DELAY',
        title="Average Departure Delay: Sundays vs. Non-Sundays",
        xlabel="Sunday Status", ylabel="Avg Departure Delay (min)",
        output_filename=os.path.join(results_subdir, 'avg_departure_delay_by_sunday.png'),
        kind='bar'
    )
    
    # Plot average departure delay by season
    plot_temporal_trends(
        df, group_col='Season', delay_col='DEPARTURE_DELAY',
        title="Average Departure Delay by Season",
        xlabel="Season", ylabel="Avg Departure Delay (min)",
        output_filename=os.path.join(results_subdir, 'avg_departure_delay_by_season.png'),
        kind='bar'
    )
    
    # Plot average departure delay by holiday status (Festival vs. No)
    plot_temporal_trends(
        df, group_col='Holiday', delay_col='DEPARTURE_DELAY',
        title="Average Departure Delay on Festival Days vs. Non-Festival Days",
        xlabel="Holiday", ylabel="Avg Departure Delay (min)",
        output_filename=os.path.join(results_subdir, 'avg_departure_delay_by_holiday.png'),
        kind='bar'
    )
    
    # Print summary statistics for each temporal grouping (example for DEP_HOUR)
    for col in ['DEP_HOUR', 'DAY_OF_WEEK', 'MONTH', 'Season', 'Holiday', 'Is_Weekend', 'Is_Sunday']:
        summary = df.groupby(col)['DEPARTURE_DELAY'].describe()
        logging.info(f"Summary Statistics by {col}:\n{summary}")
        print(f"\nSummary Statistics by {col}:")
        print(summary)

# ---------------------------
# 
# ---------------------------
def perform_temporal_statistical_analysis(df):
    """
    Compute and print correlations between temporal features and departure delay.
    
    Args:
        df (pd.DataFrame): DataFrame with temporal features.
    """
    # Create a subset with relevant numerical features
    temp_features = ['DEP_HOUR', 'DAY_OF_WEEK', 'MONTH']
    corr = df[temp_features + ['DEPARTURE_DELAY']].corr()
    logging.info("Correlation Matrix (Temporal Features & Departure Delay):\n" + corr.to_string())
    print("\nCorrelation Matrix (Temporal Features & Departure Delay):")
    print(corr)

# ---------------------------
# 
# ---------------------------
def main():
    # Define directories
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'results', 'q3')
    os.makedirs(results_dir, exist_ok=True)
    
    # Specify dataset path (adjust as needed)
    data_file = os.path.join(base_dir, 'flights.csv')
    
    # Load data 
    df = load_data(data_file)
    
    # Preprocess data and extract temporal features
    df = preprocess_temporal_features(df)
    
    # Analyze temporal patterns
    analyze_temporal_patterns(df, results_dir)
    
    # Perform additional statistical analysis on temporal features
    perform_temporal_statistical_analysis(df)
    
    logging.info("Question 3 analysis complete. Check the 'results/q3' folder for plots and summary statistics.")

if __name__ == "__main__":
    main()
