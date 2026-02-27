### main.py focuses on data clearning and training the hybrid model for this project

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import r2_score

SPLIT_DATE = '2012-07-01' # Train data before this date, test data after this date

def calculate_wmae(y_true, y_pred, is_holiday):
    weights = is_holiday.map({1: 5, 0: 1})
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def run_pipeline():
    df = pd.read_csv('../data/train.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Data cleaning includes using only departments that exist across all stores for prototype
    store_count = df['Store'].nunique()
    dept_counts = df.groupby('Dept')['Store'].nunique()
    universal_depts = dept_counts[dept_counts == store_count].index.tolist()
    
    df = df[df['Dept'].isin(universal_depts)].copy()

    # Feature engineering
    df = df.sort_values(['Store', 'Dept', 'Date'])
    # This feature is to check the sales from a year ago to compare, as many departments are seasonal
    df['Sales_Lag_52'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(52) 
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day'] = df['Date'].dt.day
    df['IsHoliday'] = df['IsHoliday'].astype(int)

    # Historical average for a specific department within a store during each week to create a baseline for the model
    seasonal_lookup = df.groupby(['Store', 'Dept', 'Week'])['Weekly_Sales'].mean().reset_index()
    seasonal_lookup.rename(columns={'Weekly_Sales': 'Seasonal_Mean'}, inplace=True)
    df = df.merge(seasonal_lookup, on=['Store', 'Dept', 'Week'], how='left')

    df = df.dropna(subset=['Sales_Lag_52']).copy()

    train_df = df[df['Date'] < SPLIT_DATE].copy()
    val_df = df[df['Date'] >= SPLIT_DATE].copy()

    features = ['Store', 'Dept', 'Year', 'Month', 'Week', 'Day', 'IsHoliday', 'Sales_Lag_52']
    
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=6, random_state=42)
    model.fit(train_df[features], train_df['Weekly_Sales'])

    # Hybrid blend of 95% XGBoost and 5% seasonal mean
    val_df['XGB_Pred'] = model.predict(val_df[features])
    val_df['Hybrid_Pred'] = (0.95 * val_df['XGB_Pred']) + (0.05 * val_df['Seasonal_Mean'])

    # Calculate and print model metrics
    r2 = r2_score(val_df['Weekly_Sales'], val_df['Hybrid_Pred'])
    wmae = calculate_wmae(val_df['Weekly_Sales'], val_df['Hybrid_Pred'], val_df['IsHoliday'])
    print(f"Model Results:\nR2 = {r2:.4f}\nWMAE = {wmae:.2f}")

    # Store models in models folder
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/model_1.pkl')
    joblib.dump(seasonal_lookup, '../models/seasonal_lookup.pkl')
    joblib.dump(universal_depts, '../models/universal_depts.pkl') # This one just saves the list of univeral departments across all stores for this iteration

if __name__ == "__main__":
    run_pipeline()