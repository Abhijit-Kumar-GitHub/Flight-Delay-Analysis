# question1_delay_propagation.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path, nrows=None):
    """
    Load the dataset from a CSV file.
    You can limit the rows for quicker exploratory analysis.
    """
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        print(f"Data loaded successfully with {len(df)} records.")
        return df
    except Exception as e:
        print("Error loading data:", e)
        raise

def preprocess_delays(df):
    """
    Convert delay and taxi time columns to numeric values.
    This function also handles non-numeric issues by coercing errors.
    """
    cols = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT', 'TAXI_IN']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Optionally, drop rows with missing delay information
    df = df.dropna(subset=['DEPARTURE_DELAY', 'ARRIVAL_DELAY'])
    return df

def analyze_delay_propagation(df):
    """
    Generate visualizations and statistics to assess delay propagation.
    - Scatter plot of Departure Delay vs. Arrival Delay.
    - Scatter plots for Taxi Out vs. Departure Delay and Taxi In vs. Arrival Delay.
    - Calculate and print correlation coefficients.
    """
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    
    # Plot Departure Delay vs. Arrival Delay
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='DEPARTURE_DELAY', y='ARRIVAL_DELAY', alpha=0.5)
    plt.title("Departure Delay vs. Arrival Delay")
    plt.xlabel("Departure Delay (minutes)")
    plt.ylabel("Arrival Delay (minutes)")
    plt.savefig(os.path.join(results_dir, 'departure_vs_arrival_delay.png'))
    plt.close()
    
    # Calculate correlation between departure and arrival delay
    corr_delay = df['DEPARTURE_DELAY'].corr(df['ARRIVAL_DELAY'])
    print("Correlation between Departure and Arrival Delays:", corr_delay)
    
    # Analyze intermediate factor: Taxi Out vs. Departure Delay
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='TAXI_OUT', y='DEPARTURE_DELAY', alpha=0.5)
    plt.title("Taxi Out Time vs. Departure Delay")
    plt.xlabel("Taxi Out (minutes)")
    plt.ylabel("Departure Delay (minutes)")
    plt.savefig(os.path.join(results_dir, 'taxi_out_vs_departure_delay.png'))
    plt.close()
    
    # Analyze intermediate factor: Taxi In vs. Arrival Delay
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='TAXI_IN', y='ARRIVAL_DELAY', alpha=0.5)
    plt.title("Taxi In Time vs. Arrival Delay")
    plt.xlabel("Taxi In (minutes)")
    plt.ylabel("Arrival Delay (minutes)")
    plt.savefig(os.path.join(results_dir, 'taxi_in_vs_arrival_delay.png'))
    plt.close()
    
    # Optionally, print basic summary statistics for delay columns
    print("\nSummary Statistics for Delays:")
    print(df[['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT', 'TAXI_IN']].describe())

def main():
    # Ensure the 'results' folder exists to save plots
    os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
    
    # Update the file path to your dataset location
    data_path = os.path.join(os.path.dirname(__file__), 'flights.csv')
    
    # Load and preprocess the data
    df = load_data(data_path)
    df = preprocess_delays(df)
    analyze_delay_propagation(df)

if __name__ == "__main__":
    main()
