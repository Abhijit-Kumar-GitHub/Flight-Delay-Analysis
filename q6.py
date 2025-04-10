"""
Flight Delay Analysis – Predictive Modeling Feasibility

Research Question 6:
  "Can we accurately predict flight delays using the available features?"

Objective:
  Develop and evaluate a predictive model (linear regression) to predict arrival delays 
  using features such as departure delay, taxi out, taxi in, and distance.

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
# 
# ---------------------------
def load_data(file_path, nrows=None):
    """
    Load the dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path, nrows=nrows, low_memory=False)
        logging.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)


def preprocess_data(df):
    """
    Preprocess the dataset:
      - Convert relevant columns to numeric.
      - Impute missing values with the column mean.
      
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # List of columns to convert and impute
    numeric_cols = ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'TAXI_OUT', 'TAXI_IN', 'DISTANCE']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
    return df


# ---------------------------
# 
# ---------------------------
def train_predict_model(df):
    """
    Train a linear regression model to predict ARRIVAL_DELAY using selected features.
    
    Returns:
        model (LinearRegression): Trained model.
        X_test, y_test (pd.DataFrame, pd.Series): Test features and target.
        y_pred (np.ndarray): Predicted arrival delays.
    """
    # Feature selection
    features = ['DEPARTURE_DELAY', 'TAXI_OUT', 'TAXI_IN', 'DISTANCE']
    target = 'ARRIVAL_DELAY'
    
    # Drop any remaining missing values
    df_model = df[features + [target]].dropna()
    
    X = df_model[features]
    y = df_model[target]
    
    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate the regression model using Mean Absolute Error and R² Score.
    """
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nModel Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
    print(f"R^2 Score: {r2:.3f}")
    logging.info(f"Model MAE: {mae:.2f}, R^2: {r2:.3f}")
    return mae, r2


def plot_predictions(y_test, y_pred, output_filename):
    """
    Plot a scatter plot of actual vs. predicted arrival delays.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Actual vs. Predicted Arrival Delays")
    plt.xlabel("Actual Arrival Delay (min)")
    plt.ylabel("Predicted Arrival Delay (min)")
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    logging.info(f"Saved prediction scatter plot: {output_filename}")


# ---------------------------
# 
# ---------------------------
def main():
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'results', 'q6')
    os.makedirs(results_dir, exist_ok=True)
    
    data_file = os.path.join(base_dir, 'flights.csv')
    
    # Load and preprocess data
    df = load_data(data_file, )  # or nrows 10000 for testing
    df = preprocess_data(df)
    
    # Train and predict using Linear Regression
    model, X_test, y_test, y_pred = train_predict_model(df)
    
    # Evaluate model performance
    evaluate_model(y_test, y_pred)
    
    # Plot Actual vs. Predicted Arrival Delays
    plot_predictions(y_test, y_pred, os.path.join(results_dir, 'actual_vs_predicted.png'))
    
    logging.info("Question 6 analysis complete. Check the 'results/q6' folder for outputs.")

if __name__ == "__main__":
    main()
