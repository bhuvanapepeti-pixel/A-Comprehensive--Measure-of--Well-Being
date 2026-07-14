"""
train_model.py
----------------------------------
Trains a Linear Regression model to predict the Human Development
Index (HDI) score based on Life Expectancy, Mean Years of Schooling,
and GNI Per Capita. Performs basic EDA, evaluates the model, and
serializes the trained model to disk for use by the Flask app.
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "hdi_dataset.csv")
MODEL_OUTPUT_PATH = "hdi_model.pkl"
VISUALS_DIR = os.path.join("static", "css")
FEATURES = ["Life_Expectancy", "Mean_Years_Schooling", "GNI_Per_Capita"]
TARGET = "HDI_Score"
RANDOM_STATE = 42

os.makedirs(VISUALS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Load the HDI dataset from a CSV file into a Pandas DataFrame."""
    print(f"[INFO] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Dataset shape: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic preprocessing:
    - Handle missing values via mean imputation for numeric columns.
    """
    numeric_cols = FEATURES + [TARGET]
    missing_before = df[numeric_cols].isnull().sum().sum()

    if missing_before > 0:
        print(f"[INFO] Found {missing_before} missing values. Applying mean imputation.")
        for col in numeric_cols:
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
    else:
        print("[INFO] No missing values detected.")

    return df


def perform_eda(df: pd.DataFrame) -> None:
    """
    Generate and save exploratory data analysis visuals:
    1. A correlation heatmap of all numeric features.
    2. Scatter plots of each feature against the target (HDI_Score).
    """
    sns.set_style("whitegrid")

    # --- Correlation Heatmap ---
    plt.figure(figsize=(7, 5))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap - HDI Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    heatmap_path = os.path.join(VISUALS_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved correlation heatmap to: {heatmap_path}")

    # --- Scatter plots: Features vs HDI Score ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, feature in zip(axes, FEATURES):
        sns.scatterplot(x=df[feature], y=df[TARGET], ax=ax, color="#2C5F8A", s=60)
        ax.set_title(f"{feature} vs HDI Score", fontsize=11, fontweight="bold")
        ax.set_xlabel(feature)
        ax.set_ylabel("HDI Score")
    plt.tight_layout()
    scatter_path = os.path.join(VISUALS_DIR, "feature_scatterplots.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved scatter plots to: {scatter_path}")


def train_and_evaluate(df: pd.DataFrame) -> LinearRegression:
    """Split the data, train a Linear Regression model, and evaluate it."""
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    print(f"[INFO] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n----------- Model Evaluation -----------")
    print(f"R-squared (R2) Score : {r2:.4f}")
    print(f"Mean Absolute Error  : {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print("-----------------------------------------")

    print("\n[INFO] Model Coefficients:")
    for feature, coef in zip(FEATURES, model.coef_):
        print(f"   {feature}: {coef:.6f}")
    print(f"   Intercept: {model.intercept_:.6f}\n")

    return model


def save_model(model: LinearRegression, path: str) -> None:
    """Serialize the trained model to disk using pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model successfully saved to: {path}")


def main():
    df = load_data(DATA_PATH)
    df = preprocess_data(df)
    perform_eda(df)
    model = train_and_evaluate(df)
    save_model(model, MODEL_OUTPUT_PATH)
    print("\n[SUCCESS] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
