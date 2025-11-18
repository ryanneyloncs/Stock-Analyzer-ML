"""
Trend analysis module - detects trends and generates signals
"""

import pandas as pd


def analyze_trends(data, ma_short=50, ma_long=200):
    """
    Analyze trends and generate trading signals
    
    Args:
        data (pd.DataFrame): Stock data with indicators
        ma_short (int): Short-term MA period
        ma_long (int): Long-term MA period
        
    Returns:
        pd.DataFrame: Data with trend signals added
    """
    print("Analyzing trends...")
    
    # MA Signal
    data['MA_Signal'] = 0
    data.loc[data[f'MA{ma_short}'] > data[f'MA{ma_long}'], 'MA_Signal'] = 1
    data.loc[data[f'MA{ma_short}'] < data[f'MA{ma_long}'], 'MA_Signal'] = -1
    
    # Short-term Signal
    data['ST_Signal'] = 0
    data.loc[data['EMA10'] > data['EMA30'], 'ST_Signal'] = 1
    data.loc[data['EMA10'] < data['EMA30'], 'ST_Signal'] = -1
    
    # Price vs MA Signal
    data['Price_MA_Signal'] = 0
    data.loc[data['Close'] > data[f'MA{ma_short}'], 'Price_MA_Signal'] = 1
    data.loc[data['Close'] < data[f'MA{ma_short}'], 'Price_MA_Signal'] = -1
    
    # Composite Signal
    data['Signal'] = data['MA_Signal']
    data.loc[(data['Price_MA_Signal'] == -1) | (data['ST_Signal'] == -1), 'Signal'] = -1
    
    # Signal Changes
    data['Signal_Change'] = data['Signal'].diff()
    
    print("Trends analyzed")
    return data
