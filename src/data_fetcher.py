"""
Data fetching module - handles retrieving stock data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import time


def fetch_data_with_retry(symbol, start_date, end_date, max_retries=3, retry_delay=10):
    """
    Fetch stock data with retry logic
    
    Args:
        symbol (str): Stock ticker symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Base delay between retries in seconds
        
    Returns:
        pd.DataFrame: Stock data with OHLCV columns, or None if failed
    """
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}: Fetching data for {symbol}...")
            if attempt > 0:
                wait_time = retry_delay * attempt
                print(f"Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if not data.empty:
                print(f"Successfully fetched {len(data)} days of data")
                return data
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
    
    return None


def clean_data(data):
    """
    Clean and prepare the fetched data
    
    Args:
        data (pd.DataFrame): Raw stock data
        
    Returns:
        pd.DataFrame: Cleaned stock data
    """
    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    # Remove rows with missing Close prices
    data = data.dropna(subset=['Close'])
    
    return data
