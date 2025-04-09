
"""
Flight Delay Analysis - Question 4: Airport & Route Analysis

Research Q4:
  Are there specific origin-destination pairs or hub airports that experience
  disproportionately high delays or cancellations?

This script produces:
  1. Top 10 origin airports by avg departure delay.
  2. Top 10 destination airports by avg arrival delay.
  3. Top 10 origin-destination routes by avg arrival delay (min. 500 flights).
  4. Scatter: route flight count vs. avg arrival delay.
  5. Top 10 busiest airports by flight volume (horizontal bar).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1) Load and Clean Data
def load_and_clean(path, nrows=None):
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    # ensure numeric
    for col in ['DEPARTURE_DELAY','ARRIVAL_DELAY','CANCELLED']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# 2) Plotting Functions
def plot_bar(df, x, y, title, xlabel, ylabel, filename, horizontal=False):
    plt.figure(figsize=(10,6))
    if horizontal:
        sns.barplot(data=df, y=x, x=y)
    else:
        sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved: {filename}")

def plot_scatter(df, x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved: {filename}")

# 3) Analysis Functions
def analyze_airports(df, outdir):
    os.makedirs(outdir, exist_ok=True)

    # a) Top 10 Origins by Avg Departure Delay
    orig = (df.groupby('ORIGIN_AIRPORT')
              .DEPARTURE_DELAY.mean()
              .sort_values(ascending=False)
              .head(10)
              .reset_index(name='Avg_Dep_Delay'))
    plot_bar(orig, 'ORIGIN_AIRPORT', 'Avg_Dep_Delay',
             "Top 10 Origin Airports by Avg Departure Delay",
             "Origin Airport", "Avg Departure Delay (min)",
             os.path.join(outdir,'top10_origin_dep_delay.png'))

    # b) Top 10 Destinations by Avg Arrival Delay
    dest = (df.groupby('DESTINATION_AIRPORT')
               .ARRIVAL_DELAY.mean()
               .sort_values(ascending=False)
               .head(10)
               .reset_index(name='Avg_Arr_Delay'))
    plot_bar(dest, 'DESTINATION_AIRPORT', 'Avg_Arr_Delay',
             "Top 10 Destination Airports by Avg Arrival Delay",
             "Destination Airport", "Avg Arrival Delay (min)",
             os.path.join(outdir,'top10_dest_arr_delay.png'))

    return orig, dest

def analyze_routes(df, outdir, min_flights=500):
    os.makedirs(outdir, exist_ok=True)

    # c) Route stats
    route = (df.groupby(['ORIGIN_AIRPORT','DESTINATION_AIRPORT'])
               .ARRIVAL_DELAY.agg(['mean','size'])
               .rename(columns={'mean':'Avg_Arr_Delay','size':'Flight_Count'})
               .reset_index())
    # filter low-volume
    route = route[route.Flight_Count >= min_flights]

    # Top 10 by Avg_Arr_Delay
    top_routes = route.sort_values('Avg_Arr_Delay', ascending=False).head(10)
    top_routes['Route'] = top_routes.ORIGIN_AIRPORT + "→" + top_routes.DESTINATION_AIRPORT
    plot_bar(top_routes, 'Route', 'Avg_Arr_Delay',
             f"Top 10 Routes (≥{min_flights} flights) by Avg Arrival Delay",
             "Route", "Avg Arrival Delay (min)",
             os.path.join(outdir,'top10_routes_arr_delay.png'))

    # d) Scatter: Flight_Count vs. Avg_Arr_Delay
    plot_scatter(route, 'Flight_Count','Avg_Arr_Delay',
                 "Route Volume vs. Avg Arrival Delay",
                 "Number of Flights","Avg Arrival Delay (min)",
                 os.path.join(outdir,'route_count_vs_delay.png'))

    return top_routes, route

def analyze_busiest_airports(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    # e) Top 10 busiest airports by flight count (origin + dest)
    counts = pd.concat([
        df.ORIGIN_AIRPORT.value_counts(),
        df.DESTINATION_AIRPORT.value_counts()
    ]).groupby(level=0).sum().sort_values(ascending=False).head(10)
    busy = counts.reset_index()
    busy.columns = ['Airport','Total_Flights']
    plot_bar(busy, 'Airport','Total_Flights',
             "Top 10 Busiest Airports by Flight Volume",
             "Airport","Total Flights",
             os.path.join(outdir,'top10_busiest_airports.png'),
             horizontal=True)
    return busy

# 4) Main Execution
def main():
    base = os.path.dirname(__file__)
    data_file = os.path.join(base,'flights.csv')
    results = os.path.join(base,'results','q4')

    df = load_and_clean(data_file) 

    orig, dest = analyze_airports(df, os.path.join(results,'airports'))
    top_routes, all_routes = analyze_routes(df, os.path.join(results,'routes'))
    busy = analyze_busiest_airports(df, os.path.join(results,'busiest'))

    # Print summaries
    logging.info("Origin Airport Summary:\n" + orig.describe().to_string())
    logging.info("Destination Airport Summary:\n" + dest.describe().to_string())
    logging.info("Route Summary (all):\n" + all_routes[['Avg_Arr_Delay','Flight_Count']].describe().to_string())
    logging.info("Busiest Airport Summary:\n" + busy.describe().to_string())

if __name__ == "__main__":
    main()
