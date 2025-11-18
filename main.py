"""
Stock Technical Analysis Tool - Modular Version
Main entry point that orchestrates all modules
"""

import sys
import pandas as pd

# Import configuration
from src import config

# Import modules
from src.data_fetcher import fetch_data_with_retry, clean_data
from src.indicators import calculate_indicators
from src.trend_analyzer import analyze_trends
from src.visualization import create_chart
from src.reporting import print_report, print_ml_report
from src.ml_predictor import predict_stock_price


def main():
    """Main function - orchestrates the analysis"""
    print("=" * 60)
    print("STOCK TECHNICAL ANALYSIS TOOL")
    print("=" * 60)
    print(f"\nAnalyzing: {config.SYMBOL}")
    print(f"Period: {config.START_DATE} to {config.END_DATE}\n")
    
    # Step 1: Fetch data
    data = fetch_data_with_retry(
        config.SYMBOL, 
        config.START_DATE, 
        config.END_DATE, 
        config.MAX_RETRIES,
        config.RETRY_DELAY
    )
    
    if data is None or data.empty:
        print("Failed to fetch data")
        sys.exit(1)
    
    # Step 2: Clean data
    data = clean_data(data)
    
    # Step 3: Calculate indicators
    data = calculate_indicators(
        data,
        ma_short=config.MA_SHORT,
        ma_long=config.MA_LONG,
        rsi_period=config.RSI_PERIOD,
        bollinger_period=config.BOLLINGER_PERIOD,
        bollinger_std=config.BOLLINGER_STD
    )
    
    # Step 4: Analyze trends
    data = analyze_trends(
        data,
        ma_short=config.MA_SHORT,
        ma_long=config.MA_LONG
    )
    
    # Step 5: Filter display data
    display_data = data[data.index >= pd.Timestamp(config.DISPLAY_START)]
    if display_data.empty:
        display_data = data
    
    # Step 6: ML Prediction (optional)
    ml_metrics = None
    ml_next_day = None
    ml_predictions_df = None
    
    if config.ENABLE_ML_PREDICTION:
        print("\n" + "=" * 60)
        print("MACHINE LEARNING PREDICTION")
        print("=" * 60)
        predictor, ml_metrics, ml_next_day, ml_predictions_df = predict_stock_price(
            data,
            lookback_days=config.ML_LOOKBACK_DAYS,
            epochs=config.ML_EPOCHS,
            verbose=config.ML_VERBOSE
        )
    
    # Step 7: Create visualization
    fig = create_chart(
        data,
        display_data,
        config.SYMBOL,
        ma_short=config.MA_SHORT,
        ma_long=config.MA_LONG,
        dpi=config.CHART_DPI,
        save=config.SAVE_CHART,
        show=config.SHOW_CHART,
        ml_predictions=ml_predictions_df  # Add ML predictions to chart
    )
    
    # Step 8: Print report
    print_report(
        data,
        display_data,
        config.SYMBOL,
        ma_short=config.MA_SHORT,
        ma_long=config.MA_LONG
    )
    
    # Step 9: Print ML report (if enabled)
    if config.ENABLE_ML_PREDICTION and ml_metrics is not None:
        print_ml_report(ml_metrics, ml_next_day)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()
