### app.py starts a web server to test the model data using data returned from engine.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import joblib
import os
from engine import get_future_forecast

app = Flask(__name__, static_folder='../web', static_url_path='')
CORS(app)

DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/train.csv')
DEPT_PATH = os.path.join(os.path.dirname(__file__), '../models/universal_depts.pkl')
df_full = pd.read_csv(DATA_PATH)
df_full['Date'] = pd.to_datetime(df_full['Date'])

# Triggers when server is entered in a browser to load HTML
@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

# Returns list of stores/departments for dropdown menus
@app.route('/get_options', methods=['GET'])
def get_options():
    stores = sorted(df_full['Store'].unique().tolist())
    universal_depts = joblib.load(DEPT_PATH)
    return jsonify({"stores": stores, "depts": universal_depts})

# After button is pressed, store ID and department ID are used to get forecast from engine.py to send back to the dashboard
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        s, d = int(data['store']), int(data['dept'])
        
        # Training dataset ends at the end of October 2012, so dates closest to end of training dataset/beginning of testing dataset are used so there is actual values along with predictions
        mask = (df_full['Store'] == s) & \
               (df_full['Dept'] == d) & \
               (df_full['Date'] >= '2012-06-01') & \
               (df_full['Date'] <= '2012-10-31')
        
        subset = df_full[mask].sort_values('Date')
        
        labels, actuals, predictions = [], [], []
        for _, row in subset.iterrows():
            date_str = row['Date'].strftime('%Y-%m-%d')
            pred = get_future_forecast(s, d, date_str, bool(row['IsHoliday']))
            labels.append(date_str)
            actuals.append(round(float(row['Weekly_Sales']), 2))
            predictions.append(pred)

        return jsonify({
            "status": "success",
            "history": {"labels": labels, "actuals": actuals, "predictions": predictions}
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)