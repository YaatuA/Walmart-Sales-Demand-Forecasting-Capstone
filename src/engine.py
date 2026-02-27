### engine.py uses the models generated from main.py to generate the future forecast

import pandas as pd
import joblib
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/model_1.pkl')
LOOKUP_PATH = os.path.join(BASE_DIR, '../models/seasonal_lookup.pkl')
TRAIN_DATA = os.path.join(BASE_DIR, '../data/train.csv')

model = joblib.load(MODEL_PATH)
seasonal_df = joblib.load(LOOKUP_PATH)
train_df = pd.read_csv(TRAIN_DATA)
train_df['Date'] = pd.to_datetime(train_df['Date'])

def get_future_forecast(store, dept, date_str, is_holiday):
    # Text input from dashboard -> datetime for seasonal lookups
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    week = int(date_obj.isocalendar()[1])
    s_id, d_id = int(store), int(dept)
    
    target_date = date_obj - pd.Timedelta(weeks=52)
    match = train_df[(train_df['Store'] == s_id) & (train_df['Dept'] == d_id) & 
                     (train_df['Date'] >= target_date - pd.Timedelta(days=3)) &
                     (train_df['Date'] <= target_date + pd.Timedelta(days=3))]
    
    # If no memory of the performance of a department of a store exists from 1 year (52 weeks) ago, the median weekly sales is used instead
    lag_val = match.iloc[0]['Weekly_Sales'] if not match.empty else train_df['Weekly_Sales'].median()

    input_data = pd.DataFrame([{
        'Store': s_id, 'Dept': d_id, 'Year': date_obj.year, 
        'Month': date_obj.month, 'Week': week, 'Day': date_obj.day, 
        'IsHoliday': 1 if is_holiday else 0, 'Sales_Lag_52': lag_val
    }])
    xgb_pred = model.predict(input_data)[0]

    stat_match = seasonal_df[(seasonal_df['Store'] == s_id) & 
                             (seasonal_df['Dept'] == d_id) & 
                             (seasonal_df['Week'] == week)]
    
    # If no memory of the seasonal mean for a department of a store exists from 1 year (52 weeks) ago, the median weekly sales is used instead
    stat_pred = stat_match.iloc[0]['Seasonal_Mean'] if not stat_match.empty else train_df['Weekly_Sales'].median()

    # Returns a hybrid value to app.py of 95% of XGBoost's prediction and 5% historical data with two decimal places to account for currency
    return round(float((0.95 * xgb_pred) + (0.05 * stat_pred)), 2)