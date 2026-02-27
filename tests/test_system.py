import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

MOCK_TRAIN_DATA = pd.DataFrame([
    {'Store': 1, 'Dept': 1, 'Date': pd.to_datetime('2011-01-07'), 'Weekly_Sales': 10000.0},
    {'Store': 1, 'Dept': 1, 'Date': pd.to_datetime('2011-12-30'), 'Weekly_Sales': 15000.0}
])

class TestForecastingSystem(unittest.TestCase):
    @patch('engine.model')
    @patch('engine.seasonal_df')
    @patch('engine.train_df', new=MOCK_TRAIN_DATA)
    def test_end_of_year_logic(self, mock_eng_seasonal, mock_eng_model):
        """
        Transition between December (week 52) to January (week 1) without indexing errors and considering weekly data from previous month within previous year
        """
        mock_eng_model.predict.return_value = np.array([12000.0])
        mock_seasonal_data = pd.DataFrame([{'Store': 1, 'Dept': 1, 'Week': 52, 'Seasonal_Mean': 14000.0}])
        mock_eng_seasonal.__getitem__.side_effect = mock_seasonal_data.__getitem__
        mock_eng_seasonal.loc = mock_seasonal_data.loc

        from engine import get_future_forecast
        result = get_future_forecast(1, 1, '2012-12-28', False)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)

    @patch('engine.model')
    @patch('engine.train_df', new=MOCK_TRAIN_DATA)
    def test_string_input_casting(self, mock_eng_model):
        """
        Convert string inputs from dashboard into integers for model
        """
        mock_eng_model.predict.return_value = np.array([5000.0])
        from engine import get_future_forecast
        try:
            get_future_forecast("1", "1", "2012-06-01", False)
        except TypeError:
            self.fail("get_future_forecast failed to handle string inputs for IDs")

    @patch('engine.train_df', new=MOCK_TRAIN_DATA)
    def test_lag_fallback_mechanism(self):
        """
        If there is no annual historical data, use the median instead of producing an error
        """
        from engine import get_future_forecast
        with patch('engine.model') as mock_model:
            mock_model.predict.return_value = np.array([8000.0])
            result = get_future_forecast(1, 1, '2010-01-01', False)
            self.assertEqual(result, 8225.0)

if __name__ == '__main__':
    unittest.main()