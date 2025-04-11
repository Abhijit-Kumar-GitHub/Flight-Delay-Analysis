#!/usr/bin/env python3
"""
flight_delay_analysis.py

Consolidated script for Questions 1-8 of the Flight Delay Analysis project.
Each analysis writes its outputs (plots, stats) into results/q1 … results/q8.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# Data Loading & Core Preprocessing
# ---------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame."""
    try:
        df = pd.read_csv(path, low_memory=False)
        logging.info(f"Loaded {len(df)} records from {path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        sys.exit(1)


def preprocess_core(df: pd.DataFrame) -> pd.DataFrame:
    """Convert key columns to numeric and impute missing values."""
    # Numeric conversions
    num_cols = [
        'DEPARTURE_DELAY','ARRIVAL_DELAY','TAXI_OUT','TAXI_IN',
        'DISTANCE','ELAPSED_TIME','AIR_TIME',
        'AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY',
        'LATE_AIRCRAFT_DELAY','WEATHER_DELAY'
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c].fillna(df[c].mean(), inplace=True)

    # Cancellation/diversion flags
    df['CANCELLED'] = pd.to_numeric(df['CANCELLED'], errors='coerce').fillna(0).astype(int)
    df['DIVERTED']  = pd.to_numeric(df['DIVERTED'],  errors='coerce').fillna(0).astype(int)
    df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].fillna('None')

    # Datetime
    df['FLIGHT_DATE'] = pd.to_datetime(df[['YEAR','MONTH','DAY']])
    df['DEP_HOUR'] = df['SCHEDULED_DEPARTURE'] // 100

    return df

# ---------------------------
# Q1: Delay Propagation & Causality
# ---------------------------
def analyze_q1(df: pd.DataFrame):
    out = os.path.join('results','q1')
    os.makedirs(out, exist_ok=True)
    # Scatter: departure vs arrival delay
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='DEPARTURE_DELAY', y='ARRIVAL_DELAY', alpha=0.3)
    plt.title('Departure Delay vs Arrival Delay')
    plt.xlabel('Departure Delay (min)')
    plt.ylabel('Arrival Delay (min)')
    plt.savefig(os.path.join(out,'dep_vs_arr.png'))
    plt.close()
    # Correlation
    corr = df[['DEPARTURE_DELAY','ARRIVAL_DELAY','TAXI_OUT','TAXI_IN']].corr()
    logging.info("Q1 correlation:\n%s", corr)

# ---------------------------
# Q2: Airline Reliability & Performance
# ---------------------------
def analyze_q2(df: pd.DataFrame):
    out = os.path.join('results','q2')
    os.makedirs(out, exist_ok=True)
    grp = df.groupby('AIRLINE').agg(
        Avg_Dep_Delay=('DEPARTURE_DELAY','mean'),
        Avg_Arr_Delay=('ARRIVAL_DELAY','mean'),
        Cancel_Rate=('CANCELLED','mean')
    ).reset_index()
    sns.barplot(data=grp, x='AIRLINE', y='Avg_Dep_Delay')
    plt.title('Avg Departure Delay by Airline')
    plt.savefig(os.path.join(out,'avg_dep_by_airline.png'))
    plt.close()
    logging.info("Q2 airline performance:\n%s", grp)

# ---------------------------
# Q3: Temporal Patterns & Seasonality
# ---------------------------
def analyze_q3(df: pd.DataFrame):
    out = os.path.join('results','q3')
    os.makedirs(out, exist_ok=True)
    df['Season'] = pd.cut(df['MONTH'], bins=[0,2,5,8,11,12],
                          labels=['Winter','Spring','Summer','Fall','Winter'], right=True)
    grp = df.groupby('Season')['DEPARTURE_DELAY'].mean().reset_index()
    sns.barplot(data=grp, x='Season', y='DEPARTURE_DELAY')
    plt.title('Avg Dep Delay by Season')
    plt.savefig(os.path.join(out,'dep_by_season.png'))
    plt.close()
    logging.info("Q3 seasonal delays:\n%s", grp)

# ---------------------------
# Q4: Airport & Route Analysis
# ---------------------------
def analyze_q4(df: pd.DataFrame):
    out = os.path.join('results','q4')
    os.makedirs(out, exist_ok=True)
    # Top origin airports
    orig = df.groupby('ORIGIN_AIRPORT')['DEPARTURE_DELAY'].mean().nlargest(10).reset_index()
    sns.barplot(data=orig, x='ORIGIN_AIRPORT', y='DEPARTURE_DELAY')
    plt.title('Top 10 Origin Airports by Dep Delay')
    plt.savefig(os.path.join(out,'top10_orig.png'))
    plt.close()
    # Top routes
    route = df.groupby(['ORIGIN_AIRPORT','DESTINATION_AIRPORT'])['ARRIVAL_DELAY']\
             .agg(['mean','size']).query('size>=500').nlargest(10,'mean').reset_index()
    route['Route']=route['ORIGIN_AIRPORT']+'→'+route['DESTINATION_AIRPORT']
    sns.barplot(data=route, x='Route', y='mean')
    plt.title('Top 10 Routes by Avg Arr Delay')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(out,'top10_routes.png'))
    plt.close()
    logging.info("Q4 top routes:\n%s", route)

# ---------------------------
# Q5: Delay Attribution
# ---------------------------
def analyze_q5(df: pd.DataFrame):
    out = os.path.join('results','q5')
    os.makedirs(out, exist_ok=True)
    causes = ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']
    sums = df[causes].sum()
    pct = (sums/sums.sum()*100).sort_values(ascending=False)
    # Bar
    sns.barplot(x=pct.index, y=pct.values)
    plt.title('Delay Attribution (%)')
    plt.savefig(os.path.join(out,'delay_attr.png'))
    plt.close()
    # Boxplot
    melted = df[causes].melt(var_name='Cause', value_name='Delay')
    sns.boxplot(data=melted, x='Cause', y='Delay')
    plt.title('Delay Cause Distribution')
    plt.savefig(os.path.join(out,'delay_box.png'))
    plt.close()
    logging.info("Q5 attribution:\n%s", pct)

# ---------------------------
# Q6: Predictive Modeling
# ---------------------------
def analyze_q6(df: pd.DataFrame):
    out = os.path.join('results','q6')
    os.makedirs(out, exist_ok=True)
    feat = ['DEPARTURE_DELAY','TAXI_OUT','TAXI_IN','DISTANCE']
    data = df[feat+['ARRIVAL_DELAY']].dropna()
    X_train,X_test,y_train,y_test = train_test_split(data[feat], data['ARRIVAL_DELAY'], test_size=0.2, random_state=42)
    model=LinearRegression().fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mae = mean_absolute_error(y_test,y_pred)
    r2  = r2_score(y_test,y_pred)
    logging.info(f"Q6 MAE={mae:.2f}, R2={r2:.3f}")
    # Plot
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--')
    plt.title('Actual vs Predicted Arr Delay')
    plt.savefig(os.path.join(out,'pred_scatter.png'))
    plt.close()

# ---------------------------
# Q7: Cancellation & Diversion Analysis
# ---------------------------
def analyze_q7(df: pd.DataFrame):
    out = os.path.join('results','q7')
    os.makedirs(out, exist_ok=True)
    # Reasons
    reasons = df[df.CANCELLED==1]['CANCELLATION_REASON'].value_counts(normalize=True)*100
    sns.barplot(x=reasons.index, y=reasons.values)
    plt.title('Cancellation Reasons (%)')
    plt.savefig(os.path.join(out,'cancel_reasons.png'))
    plt.close()
    # Delay by cancel/divert
    for flag in ['CANCELLED','DIVERTED']:
        grp = df.groupby(flag)['ARRIVAL_DELAY'].mean().reset_index()
        sns.barplot(data=grp, x=flag, y='ARRIVAL_DELAY')
        plt.title(f'Avg Arr Delay by {flag}')
        plt.savefig(os.path.join(out,f'arr_by_{flag.lower()}.png'))
        plt.close()
    # Correlation
    corr = df[['CANCELLED','DIVERTED','DEPARTURE_DELAY','ARRIVAL_DELAY']].corr()
    sns.heatmap(corr, annot=True)
    plt.title('Corr Cancel/Divert & Delays')
    plt.savefig(os.path.join(out,'corr_cancel.png'))
    plt.close()
    logging.info("Q7 correlation:\n%s", corr)

# ---------------------------
# Q8: Turnaround Efficiency
# ---------------------------
def analyze_q8(df: pd.DataFrame):
    out = os.path.join('results','q8')
    os.makedirs(out, exist_ok=True)
    df['GROUND_TIME'] = df['ELAPSED_TIME'] - df['AIR_TIME']
    # Scatter plots
    for x in ['TAXI_OUT','TAXI_IN']:
        sns.scatterplot(data=df, x=x, y='GROUND_TIME', alpha=0.3)
        plt.title(f'{x} vs Ground Time')
        plt.savefig(os.path.join(out,f'{x.lower()}_vs_ground.png'))
        plt.close()
    # Top airports
    orig = df.groupby('ORIGIN_AIRPORT')['TAXI_OUT'].mean().nlargest(10).reset_index()
    sns.barplot(data=orig, x='ORIGIN_AIRPORT', y='TAXI_OUT')
    plt.title('Top 10 Origin Taxi Out')
    plt.savefig(os.path.join(out,'top10_orig_taxi_out.png'))
    plt.close()
    dest = df.groupby('DESTINATION_AIRPORT')['TAXI_IN'].mean().nlargest(10).reset_index()
    sns.barplot(data=dest, x='DESTINATION_AIRPORT', y='TAXI_IN')
    plt.title('Top 10 Dest Taxi In')
    plt.savefig(os.path.join(out,'top10_dest_taxi_in.png'))
    plt.close()
    # Corr heatmap
    cols = ['TAXI_OUT','TAXI_IN','GROUND_TIME','DEPARTURE_DELAY','ARRIVAL_DELAY']
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True)
    plt.title('Corr Turnaround Metrics')
    plt.savefig(os.path.join(out,'corr_turnaround.png'))
    plt.close()
    logging.info("Q8 correlation:\n%s", corr)

# ---------------------------
# Main
# ---------------------------
def main():
    data_file = os.path.join(os.path.dirname(__file__),'flights.csv')
    df = load_data(data_file)
    df = preprocess_core(df)
    analyze_q1(df)
    analyze_q2(df)
    analyze_q3(df)
    analyze_q4(df)
    analyze_q5(df)
    analyze_q6(df)
    analyze_q7(df)
    analyze_q8(df)
    logging.info("All analyses complete.")

if __name__ == '__main__':
    main()
