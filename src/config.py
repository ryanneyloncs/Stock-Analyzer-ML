"""
Configuration settings for stock analysis
Edit these values to customize your analysis
"""

# Stock Settings
SYMBOL = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2025-11-11"
DISPLAY_START = "2024-01-01"

# Technical Indicator Settings
MA_SHORT = 50
MA_LONG = 200
RSI_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Data Fetching Settings
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds

# Output Settings
CHART_DPI = 300
SAVE_CHART = True
SHOW_CHART = False  # Set to True to display chart window

# Machine Learning Settings
ENABLE_ML_PREDICTION = True  # Set to False to disable ML predictions
ML_LOOKBACK_DAYS = 60  # Number of past days to use for prediction
ML_EPOCHS = 100  # Training epochs (more = better but slower)
ML_VERBOSE = 0  # 0=silent, 1=progress bar
