"""
Reporting module - generates analysis reports
"""

import pandas as pd
import numpy as np


def print_report(data, display_data, symbol, ma_short=50, ma_long=200):
    """
    Print comprehensive analysis report
    
    Args:
        data (pd.DataFrame): Full stock data
        display_data (pd.DataFrame): Filtered display data
        symbol (str): Stock ticker symbol
        ma_short (int): Short-term MA period
        ma_long (int): Long-term MA period
    """
    valid_mask = (~display_data[f'MA{ma_short}'].isna()) & (~display_data[f'MA{ma_long}'].isna())
    valid_data = display_data[valid_mask]
    
    if len(valid_data) == 0:
        print("WARNING: Not enough data for report")
        return
    
    # Trend stats
    uptrend_days = (valid_data['Signal'] == 1).sum()
    downtrend_days = (valid_data['Signal'] == -1).sum()
    neutral_days = (valid_data['Signal'] == 0).sum()
    total_days = len(valid_data)
    
    # Performance metrics
    first_close = display_data['Close'].iloc[0]
    last_close = display_data['Close'].iloc[-1]
    total_return = (last_close / first_close - 1) * 100
    
    # Volatility
    daily_returns = display_data['Daily_Return'].dropna()
    if len(daily_returns) > 0:
        volatility = daily_returns.std()
        annualized_volatility = volatility * np.sqrt(252)
    else:
        annualized_volatility = 0
    
    # Max drawdown
    cumulative_max = display_data['Close'].cummax()
    drawdown = (display_data['Close'] / cumulative_max - 1) * 100
    max_drawdown = drawdown.min()
    
    # Print report
    print(f"\n{'=' * 20} {symbol} ANALYSIS REPORT {'=' * 20}")
    print(f"\n----- TREND ANALYSIS -----")
    print(f"Total trading days: {total_days}")
    print(f"Days in uptrend: {uptrend_days} ({uptrend_days/total_days*100:.2f}%)")
    print(f"Days in downtrend: {downtrend_days} ({downtrend_days/total_days*100:.2f}%)")
    
    print(f"\n----- PERFORMANCE METRICS -----")
    print(f"Starting price: ${first_close:.2f}")
    print(f"Ending price: ${last_close:.2f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Annualized volatility: {annualized_volatility:.2f}%")
    print(f"Maximum drawdown: {max_drawdown:.2f}%")
    
    print(f"\n----- TECHNICAL INDICATORS (LATEST) -----")
    latest_rsi = display_data['RSI'].iloc[-1]
    if pd.notna(latest_rsi):
        print(f"RSI: {latest_rsi:.2f}", end="")
        if latest_rsi > 70:
            print(" → Overbought")
        elif latest_rsi < 30:
            print(" → Oversold")
        else:
            print(" → Neutral")
    
    current_signal = display_data['Signal'].iloc[-1]
    trend_status = "UPTREND" if current_signal == 1 else "DOWNTREND" if current_signal == -1 else "NEUTRAL"
    print(f"\nCurrent trend: {trend_status}")
    print(f"{'=' * 60}\n")


def print_ml_report(metrics, next_day_prediction):
    """
    Print ML prediction report
    
    Args:
        metrics (dict): Training metrics from ML model
        next_day_prediction (float): Predicted next day price
    """
    if metrics is None or next_day_prediction is None:
        return
    
    print(f"\n----- MACHINE LEARNING PREDICTIONS -----")
    print(f"Model Performance:")
    print(f"  Training MAE: ${metrics['train_mae']:.2f}")
    print(f"  Testing MAE: ${metrics['test_mae']:.2f}")
    print(f"  Training RMSE: ${metrics['train_rmse']:.2f}")
    print(f"  Testing RMSE: ${metrics['test_rmse']:.2f}")
    print(f"\nNext Day Prediction: ${next_day_prediction:.2f}")
    print(f"{'=' * 60}\n")
