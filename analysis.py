"""
flight_delay_analysis.py

Consolidated script for Questions 1–8 of the Flight Delay Analysis project,
enriched with airline, cancellation‐reason, and airport metadata,
and with a context‐aware missing‐value strategy.
"""
import os
import sys
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# Lookup maps
# ------------------------------------------------------------------------------
AIRLINE_MAP = {
    "UA": "United Air Lines Inc.",
    "AA": "American Airlines Inc.",
    "US": "US Airways Inc.",
    "F9": "Frontier Airlines Inc.",
    "B6": "JetBlue Airways",
    "OO": "Skywest Airlines Inc.",
    "AS": "Alaska Airlines Inc.",
    "NK": "Spirit Air Lines",
    "WN": "Southwest Airlines Co.",
    "DL": "Delta Air Lines Inc.",
    "EV": "Atlantic Southeast Airlines",
    "HA": "Hawaiian Airlines Inc.",
    "MQ": "American Eagle Airlines Inc.",
    "VX": "Virgin America"
}

CANC_REASON_MAP = {
    "A": "Airline/Carrier",
    "B": "Weather",
    "C": "National Air System",
    "D": "Security",
    "None": "Not Cancelled"
}

# ------------------------------------------------------------------------------
# Data Loading & Missing‐Value Helpers
# ------------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load flights.csv into a DataFrame."""
    try:
        df = pd.read_csv(path, low_memory=False)
        logging.info(f"Loaded {len(df)} flight records from {path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        sys.exit(1)

def load_airports(path: str) -> pd.DataFrame:
    """Load airports.csv into a DataFrame."""
    try:
        airports = pd.read_csv(path, dtype={'IATA_CODE': str}, low_memory=False)
        logging.info(f"Loaded {len(airports)} airport records from {path}")
        return airports
    except FileNotFoundError:
        logging.error(f"Airport file not found: {path}")
        sys.exit(1)

def summarize_missing(df: pd.DataFrame, name: str):
    """Log both isnull() and isna() summaries for DataFrame `df`."""
    nulls = df.isnull().sum().sort_values(ascending=False)
    nas   = df.isna().sum().sort_values(ascending=False)
    logging.info(f"=== Missing summary for {name} ===")
    if nulls.any():
        logging.info("isnull():\n%s", nulls[nulls>0].to_string())
    if nas.any():
        logging.info("isna():\n%s",   nas[nas>0].to_string())
    if not (nulls.any() or nas.any()):
        logging.info("No missing values in %s", name)

# ------------------------------------------------------------------------------
# Context‐Aware Imputation
# ------------------------------------------------------------------------------
def handle_missing_contextual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Context‑aware imputation without chained assignment:
      - Delays → median by airline
      - Taxi times → median by airport
      - Distance → median by route
      - Other numerics → global median
      - Flags → fill 0
      - Descriptions & metadata → 'Unknown'
    """
    # 1) Delays by airline
    for col in ['DEPARTURE_DELAY','ARRIVAL_DELAY']:
        df[col] = df.groupby('AIRLINE_FULL')[col]\
                    .transform(lambda x: x.fillna(x.median()))

    # 2) Taxi out by origin airport
    df['TAXI_OUT'] = df.groupby('ORIGIN_AIRPORT')['TAXI_OUT']\
                       .transform(lambda x: x.fillna(x.median()))

    # 3) Taxi in by destination airport
    df['TAXI_IN']  = df.groupby('DESTINATION_AIRPORT')['TAXI_IN']\
                       .transform(lambda x: x.fillna(x.median()))

    # 4) Distance by route
    df['DISTANCE'] = df.groupby(
        ['ORIGIN_AIRPORT','DESTINATION_AIRPORT']
    )['DISTANCE'].transform(lambda x: x.fillna(x.median()))

    # 5) Other numeric columns: global median
    other_nums = [
        'ELAPSED_TIME','AIR_TIME',
        'AIR_SYSTEM_DELAY','SECURITY_DELAY',
        'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY'
    ]
    for c in other_nums:
        if df[c].isna().any():
            med = df[c].median()
            df[c] = df[c].fillna(med)
            logging.info(f"Filled {c} with global median={med:.2f}")

    # 6) Cancellation flags
    df['CANCELLED'] = df['CANCELLED'].fillna(0)
    df['DIVERTED']  = df['DIVERTED'].fillna(0)

    # 7) Cancellation descriptions
    df['CANCELLATION_DESC'] = df['CANCELLATION_DESC'].fillna('Unknown')

    # 8) Airport metadata fields
    for col in ['ORIG_NAME','ORIG_CITY','ORIG_STATE',
                'DEST_NAME','DEST_CITY','DEST_STATE']:
        df[col] = df[col].fillna('Unknown')

    # 9) Any remaining object columns → 'Unknown'
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna('Unknown')

    return df


# ------------------------------------------------------------------------------
# Core Preprocessing
# ------------------------------------------------------------------------------
def preprocess_core(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    """
    1) Summarize missing in raw flights
    2) Map airline codes & cancellation reasons
    3) Convert flight‐level numerics to numeric dtype
    4) Derive FLIGHT_DATE & DEP_HOUR
    5) Merge airport metadata for origin & destination
    6) Impute all remaining missing values
    """
    summarize_missing(df, "raw flights")

    # Map airline codes → full names
    df['AIRLINE_FULL'] = df['AIRLINE'].map(AIRLINE_MAP).fillna("Unknown Airline")

    # Map cancellation reasons → descriptions
    df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].fillna('None')
    df['CANCELLATION_DESC']    = df['CANCELLATION_REASON'].map(CANC_REASON_MAP)

    # Convert key columns to numeric
    num_cols = [
        'DEPARTURE_DELAY','ARRIVAL_DELAY','TAXI_OUT','TAXI_IN',
        'DISTANCE','ELAPSED_TIME','AIR_TIME',
        'AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY',
        'LATE_AIRCRAFT_DELAY','WEATHER_DELAY'
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Flags
    df['CANCELLED'] = pd.to_numeric(df['CANCELLED'], errors='coerce')
    df['DIVERTED']  = pd.to_numeric(df['DIVERTED'],  errors='coerce')

    # Date/time features
    df['FLIGHT_DATE'] = pd.to_datetime(df[['YEAR','MONTH','DAY']])
    df['DEP_HOUR']     = (df['SCHEDULED_DEPARTURE'] // 100).astype(int)

    # Merge origin airport metadata
    orig_cols = {
        'IATA_CODE':'ORIGIN_AIRPORT',
        'AIRPORT':'ORIG_NAME',
        'CITY':'ORIG_CITY',
        'STATE':'ORIG_STATE',
        'COUNTRY':'ORIG_COUNTRY',
        'LATITUDE':'ORIG_LAT',
        'LONGITUDE':'ORIG_LON'
    }
    df = df.merge(
        airports.rename(columns=orig_cols),
        on='ORIGIN_AIRPORT',
        how='left'
    )

    # Merge destination airport metadata
    dest_cols = {
        'IATA_CODE':'DESTINATION_AIRPORT',
        'AIRPORT':'DEST_NAME',
        'CITY':'DEST_CITY',
        'STATE':'DEST_STATE',
        'COUNTRY':'DEST_COUNTRY',
        'LATITUDE':'DEST_LAT',
        'LONGITUDE':'DEST_LON'
    }
    df = df.merge(
        airports.rename(columns=dest_cols),
        on='DESTINATION_AIRPORT',
        how='left'
    )

    # Final missing‐value imputation
    df = handle_missing_contextual(df)
    summarize_missing(df, "enriched flights (post‐impute)")

    return df

# ------------------------------------------------------------------------------
# Q1: Delay Propagation & Causality
# ------------------------------------------------------------------------------
def analyze_q1(df: pd.DataFrame):
    out = os.path.join('results','q1'); os.makedirs(out, exist_ok=True)
    # Scatter: Departure vs Arrival Delay
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='DEPARTURE_DELAY', y='ARRIVAL_DELAY', alpha=0.3)
    plt.title('Departure Delay vs Arrival Delay')
    plt.xlabel('Departure Delay (min)'); plt.ylabel('Arrival Delay (min)')
    plt.savefig(os.path.join(out,'dep_vs_arr.png')); plt.close()
    # Boxplots of delays & taxi times
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df[['DEPARTURE_DELAY','ARRIVAL_DELAY','TAXI_OUT','TAXI_IN']])
    plt.title('Boxplot: Delay & Taxi Times'); plt.xticks(rotation=45)
    plt.savefig(os.path.join(out,'boxplots_delays.png')); plt.close()
    # Correlation
    corr = df[['DEPARTURE_DELAY','ARRIVAL_DELAY','TAXI_OUT','TAXI_IN']].corr()
    logging.info("Q1 correlation:\n%s", corr)

# ------------------------------------------------------------------------------
# Q2: Airline Reliability & Performance
# ------------------------------------------------------------------------------
def analyze_q2(df: pd.DataFrame):
    out = os.path.join('results','q2'); os.makedirs(out, exist_ok=True)
    grp = df.groupby('AIRLINE_FULL').agg(
        Avg_Dep_Delay=('DEPARTURE_DELAY','mean'),
        Avg_Arr_Delay=('ARRIVAL_DELAY','mean'),
        Cancel_Rate=('CANCELLED','mean')
    ).reset_index().sort_values('Avg_Dep_Delay')
    # Departure delay bar
    plt.figure(figsize=(12,6))
    sns.barplot(data=grp, y='AIRLINE_FULL', x='Avg_Dep_Delay')
    plt.title('Avg Departure Delay by Airline'); plt.xlabel('Min'); plt.ylabel('')
    plt.savefig(os.path.join(out,'avg_dep_by_airline.png')); plt.close()
    # Arrival delay bar
    plt.figure(figsize=(12,6))
    sns.barplot(data=grp, y='AIRLINE_FULL', x='Avg_Arr_Delay')
    plt.title('Avg Arrival Delay by Airline'); plt.xlabel('Min'); plt.ylabel('')
    plt.savefig(os.path.join(out,'avg_arr_by_airline.png')); plt.close()
    # Cancellation rate bar
    grp['Cancel_%'] = grp['Cancel_Rate']*100
    plt.figure(figsize=(12,6))
    sns.barplot(data=grp, y='AIRLINE_FULL', x='Cancel_%')
    plt.title('Cancellation Rate by Airline (%)'); plt.xlabel('%'); plt.ylabel('')
    plt.savefig(os.path.join(out,'cancel_rate_by_airline.png')); plt.close()
    logging.info("Q2 airline performance:\n%s", grp)

# ------------------------------------------------------------------------------
# Q3: Temporal Patterns & Seasonality
# ------------------------------------------------------------------------------
def analyze_q3(df: pd.DataFrame):
    out = os.path.join('results','q3'); os.makedirs(out, exist_ok=True)
    # Season mapping
    df['Season'] = pd.cut(
        df['MONTH'],
        bins=[0,2,5,8,11,12],
        labels=['Late Winter','Spring','Summer','Fall','Early Winter'],
        right=True
    )
    # By hour
    hr = df.groupby('DEP_HOUR')['DEPARTURE_DELAY'].mean().reset_index()
    plt.figure(figsize=(10,6))
    sns.lineplot(data=hr, x='DEP_HOUR', y='DEPARTURE_DELAY', marker='o')
    plt.title('Avg Departure Delay by Hour'); plt.xlabel('Hour'); plt.ylabel('Min')
    plt.savefig(os.path.join(out,'dep_delay_by_hour.png')); plt.close()
    # By season
    # ss = df.groupby('Season')['DEPARTURE_DELAY'].mean().reset_index()         -->> warining dera koi
    ss = df.groupby('Season', observed=True)['DEPARTURE_DELAY'].mean().reset_index()
    plt.figure(figsize=(8,6))
    sns.barplot(data=ss, x='Season', y='DEPARTURE_DELAY')
    plt.title('Avg Departure Delay by Season'); plt.xlabel('Season'); plt.ylabel('Min')
    plt.savefig(os.path.join(out,'dep_delay_by_season.png')); plt.close()
    # Weekend vs Weekday
    df['Is_Weekend'] = df['DAY_OF_WEEK'].isin([6,7])
    wk = df.groupby('Is_Weekend')['DEPARTURE_DELAY'].mean().reset_index()
    plt.figure(figsize=(6,4))
    sns.barplot(data=wk, x='Is_Weekend', y='DEPARTURE_DELAY')
    plt.title('Weekend vs Weekday Delay'); plt.xlabel('Weekend'); plt.ylabel('Min')
    plt.savefig(os.path.join(out,'weekend_vs_weekday.png')); plt.close()
    logging.info("Q3 temporal stats:\n%s", df[['DEP_HOUR','Season','Is_Weekend','DEPARTURE_DELAY']].groupby(['Season','Is_Weekend']).mean())

# ------------------------------------------------------------------------------
# Q4: Airport & Route Analysis (with city/state labels)
# ------------------------------------------------------------------------------
def analyze_q4(df: pd.DataFrame):
    out = os.path.join('results','q4'); os.makedirs(out, exist_ok=True)
    # Top 10 origins by departure delay
    orig = df.groupby(['ORIGIN_AIRPORT','ORIG_CITY','ORIG_STATE'])['DEPARTURE_DELAY']\
             .mean().nlargest(10).reset_index()
    orig['Label'] = orig['ORIGIN_AIRPORT'] + " (" + orig['ORIG_CITY'] + ", " + orig['ORIG_STATE'] + ")"
    plt.figure(figsize=(12,6))
    sns.barplot(data=orig, y='Label', x='DEPARTURE_DELAY')
    plt.title('Top 10 Origin Airports by Avg Departure Delay'); plt.xlabel('Min'); plt.ylabel('')
    plt.savefig(os.path.join(out,'top10_origin_dep_delay.png')); plt.close()

    # Top 10 destinations by arrival delay
    dest = df.groupby(['DESTINATION_AIRPORT','DEST_CITY','DEST_STATE'])['ARRIVAL_DELAY']\
             .mean().nlargest(10).reset_index()
    dest['Label'] = dest['DESTINATION_AIRPORT'] + " (" + dest['DEST_CITY'] + ", " + dest['DEST_STATE'] + ")"
    plt.figure(figsize=(12,6))
    sns.barplot(data=dest, y='Label', x='ARRIVAL_DELAY')
    plt.title('Top 10 Destination Airports by Avg Arrival Delay'); plt.xlabel('Min'); plt.ylabel('')
    plt.savefig(os.path.join(out,'top10_dest_arr_delay.png')); plt.close()

    # Top 10 routes (>=500 flights) by arrival delay
    route = df.groupby(['ORIGIN_AIRPORT','DESTINATION_AIRPORT'])['ARRIVAL_DELAY']\
              .agg(['mean','size']).query('size>=500')\
              .nlargest(10,'mean').reset_index()
    route['Route'] = route['ORIGIN_AIRPORT'] + "→" + route['DESTINATION_AIRPORT']
    plt.figure(figsize=(12,6))
    sns.barplot(data=route, y='Route', x='mean')
    plt.title('Top 10 Routes by Avg Arrival Delay (≥500 flights)'); plt.xlabel('Min'); plt.ylabel('')
    plt.savefig(os.path.join(out,'top10_routes_arr_delay.png')); plt.close()

    # Scatter: flight count vs avg delay
    all_routes = df.groupby(['ORIGIN_AIRPORT','DESTINATION_AIRPORT'])['ARRIVAL_DELAY']\
                   .agg(['mean','size']).reset_index()
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=all_routes, x='size', y='mean', alpha=0.5)
    plt.title('Route Volume vs Avg Arrival Delay'); plt.xlabel('Flights'); plt.ylabel('Min')
    plt.savefig(os.path.join(out,'route_volume_vs_delay.png')); plt.close()

    # Top 10 busiest airports by volume
    counts = pd.concat([df['ORIGIN_AIRPORT'], df['DESTINATION_AIRPORT']])\
               .value_counts().nlargest(10).reset_index()
    counts.columns = ['IATA','Total_Flights']
    counts = counts.merge(df[['ORIGIN_AIRPORT','ORIG_CITY','ORIG_STATE']].drop_duplicates(),
                          left_on='IATA', right_on='ORIGIN_AIRPORT', how='left')
    counts['Label'] = counts['IATA'] + " (" + counts['ORIG_CITY'] + ", " + counts['ORIG_STATE'] + ")"
    plt.figure(figsize=(12,6))
    sns.barplot(data=counts, y='Label', x='Total_Flights')
    plt.title('Top 10 Busiest Airports by Flight Volume'); plt.xlabel('Flights'); plt.ylabel('')
    plt.savefig(os.path.join(out,'top10_busiest_airports.png')); plt.close()

# ------------------------------------------------------------------------------
# Q5: Delay Attribution
# ------------------------------------------------------------------------------
def analyze_q5(df: pd.DataFrame):
    out = os.path.join('results','q5'); os.makedirs(out, exist_ok=True)
    causes = ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']
    sums = df[causes].sum()
    pct  = (sums / sums.sum() * 100).sort_values(ascending=False)

    # Bar
    plt.figure(figsize=(8,6))
    sns.barplot(x=pct.index, y=pct.values)
    plt.title('Delay Attribution (%)'); plt.ylabel('%'); plt.xlabel('')
    plt.savefig(os.path.join(out,'delay_attr_pct.png')); plt.close()

    # Boxplot
    df_m = df[causes].melt(var_name='Cause', value_name='Delay')
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df_m, x='Cause', y='Delay')
    plt.title('Delay Cause Distribution'); plt.xticks(rotation=45)
    plt.savefig(os.path.join(out,'delay_causes_boxplot.png')); plt.close()

# ------------------------------------------------------------------------------
# Q6: Predictive Modeling
# ------------------------------------------------------------------------------
def analyze_q6(df: pd.DataFrame):
    out = os.path.join('results','q6'); os.makedirs(out, exist_ok=True)
    feats = ['DEPARTURE_DELAY','TAXI_OUT','TAXI_IN','DISTANCE']
    data = df[feats + ['ARRIVAL_DELAY']].dropna()
    X_train,X_test,y_train,y_test = train_test_split(
        data[feats], data['ARRIVAL_DELAY'], test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test,y_pred)
    r2  = r2_score(y_test,y_pred)
    logging.info(f"Q6 MAE={mae:.2f}, R²={r2:.3f}")

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--')
    plt.title('Actual vs Predicted Arrival Delay'); plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.savefig(os.path.join(out,'pred_vs_actual.png')); plt.close()

# ------------------------------------------------------------------------------
# Q7: Cancellation & Diversion Analysis
# ------------------------------------------------------------------------------
def analyze_q7(df: pd.DataFrame):
    out = os.path.join('results','q7'); os.makedirs(out, exist_ok=True)
    # Reasons
    reasons = df[df['CANCELLED']==1]['CANCELLATION_DESC']\
                .value_counts(normalize=True)*100
    plt.figure(figsize=(8,6))
    sns.barplot(x=reasons.index, y=reasons.values)
    plt.title('Cancellation Reasons (%)'); plt.ylabel('%'); plt.xlabel('')
    plt.savefig(os.path.join(out,'cancel_reasons_pct.png')); plt.close()

    # Delay by cancel/divert
    for flag in ['CANCELLED','DIVERTED']:
        grp = df.groupby(flag)['ARRIVAL_DELAY'].mean().reset_index()
        plt.figure(figsize=(6,4))
        sns.barplot(data=grp, x=flag, y='ARRIVAL_DELAY')
        plt.title(f'Avg Arrival Delay by {flag}'); plt.ylabel('Min'); plt.xlabel(flag)
        plt.savefig(os.path.join(out,f'arr_delay_by_{flag.lower()}.png')); plt.close()

    # Corr heatmap
    corr = df[['CANCELLED','DIVERTED','DEPARTURE_DELAY','ARRIVAL_DELAY']].corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
    plt.title('Correlation: Cancel/Divert & Delays')
    plt.savefig(os.path.join(out,'corr_cancel_divert.png')); plt.close()

# ------------------------------------------------------------------------------
# Q8: Operational Efficiency & Turnaround Times
# ------------------------------------------------------------------------------
def analyze_q8(df: pd.DataFrame):
    out = os.path.join('results','q8'); os.makedirs(out, exist_ok=True)
    df['GROUND_TIME'] = df['ELAPSED_TIME'] - df['AIR_TIME']

    # Scatter: Taxi Out vs Ground Time
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='TAXI_OUT', y='GROUND_TIME', alpha=0.3)
    plt.title('Taxi Out vs Ground Time'); plt.xlabel('Taxi Out'); plt.ylabel('Ground Time')
    plt.savefig(os.path.join(out,'taxi_out_vs_ground.png')); plt.close()

    # Scatter: Taxi In vs Ground Time
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='TAXI_IN', y='GROUND_TIME', alpha=0.3)
    plt.title('Taxi In vs Ground Time'); plt.xlabel('Taxi In'); plt.ylabel('Ground Time')
    plt.savefig(os.path.join(out,'taxi_in_vs_ground.png')); plt.close()

    # Top 10 origin taxi out
    top_orig = df.groupby(['ORIGIN_AIRPORT','ORIG_CITY','ORIG_STATE'])['TAXI_OUT']\
                 .mean().nlargest(10).reset_index()
    top_orig['Label'] = top_orig['ORIGIN_AIRPORT'] + " (" + top_orig['ORIG_CITY'] + ")"
    plt.figure(figsize=(12,6))
    sns.barplot(data=top_orig, y='Label', x='TAXI_OUT')
    plt.title('Top 10 Origin Taxi Out'); plt.xlabel('Min'); plt.ylabel('')
    plt.savefig(os.path.join(out,'top10_orig_taxi_out.png')); plt.close()

    # Top 10 dest taxi in
    top_dest = df.groupby(['DESTINATION_AIRPORT','DEST_CITY','DEST_STATE'])['TAXI_IN']\
                 .mean().nlargest(10).reset_index()
    top_dest['Label'] = top_dest['DESTINATION_AIRPORT'] + " (" + top_dest['DEST_CITY'] + ")"
    plt.figure(figsize=(12,6))
    sns.barplot(data=top_dest, y='Label', x='TAXI_IN')
    plt.title('Top 10 Destination Taxi In'); plt.xlabel('Min'); plt.ylabel('')
    plt.savefig(os.path.join(out,'top10_dest_taxi_in.png')); plt.close()

    # Correlation heatmap
    cols = ['TAXI_OUT','TAXI_IN','GROUND_TIME','DEPARTURE_DELAY','ARRIVAL_DELAY']
    corr = df[cols].corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
    plt.title('Corr Turnaround Metrics')
    plt.savefig(os.path.join(out,'corr_turnaround.png')); plt.close()

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    base = os.path.dirname(__file__)
    flights_file  = os.path.join(base, 'flights.csv')
    airports_file = os.path.join(base, 'airports.csv')

    # Load
    df_airports = load_airports(airports_file)
    df_flights  = load_data(flights_file)

    # Preprocess & enrich
    df = preprocess_core(df_flights, df_airports)

    # Run all analyses
    analyze_q1(df)
    analyze_q2(df)
    analyze_q3(df)
    analyze_q4(df)
    analyze_q5(df)
    analyze_q6(df)
    analyze_q7(df)
    analyze_q8(df)

    logging.info("All analyses complete. Check results/q1 … results/q8.")

if __name__ == '__main__':
    main()
