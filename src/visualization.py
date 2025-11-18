"""
Visualization module - creates charts and plots
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter


def create_chart(data, display_data, symbol, ma_short=50, ma_long=200, 
                dpi=300, save=True, show=False, ml_predictions=None):
    """
    Create the analysis chart with multiple panels

    Args:
        data (pd.DataFrame): Full stock data
        display_data (pd.DataFrame): Filtered data for display
        symbol (str): Stock ticker symbol
        ma_short (int): Short-term MA period
        ma_long (int): Long-term MA period
        dpi (int): Chart resolution
        save (bool): Whether to save the chart
        show (bool): Whether to display the chart
        ml_predictions (pd.DataFrame): ML predictions dataframe (optional)

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print("Creating chart...")

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)

    # Price formatter
    def price_formatter(x, pos):
        return f'${x:.2f}'

    # Main price plot
    ax1 = plt.subplot(gs[0])
    ax1.plot(display_data.index, display_data['Close'], label='Close Price', color='blue', linewidth=1.5)
    ax1.plot(display_data.index, display_data[f'MA{ma_short}'], label=f'{ma_short}-Day MA', color='orange', linewidth=1)
    ax1.plot(display_data.index, display_data[f'MA{ma_long}'], label=f'{ma_long}-Day MA', color='purple', linewidth=1)
    ax1.plot(display_data.index, display_data['BB_upper'], '--', label='BB Upper', color='gray', alpha=0.7, linewidth=0.8)
    ax1.plot(display_data.index, display_data['BB_middle'], '-', label='BB Middle', color='gray', alpha=0.7, linewidth=0.8)
    ax1.plot(display_data.index, display_data['BB_lower'], '--', label='BB Lower', color='gray', alpha=0.7, linewidth=0.8)

    # Add ML predictions if available
    if ml_predictions is not None and not ml_predictions.empty:
        # Filter predictions to match display period
        ml_display = ml_predictions[ml_predictions.index >= display_data.index[0]]
        if not ml_display.empty:
            ax1.plot(ml_display.index, ml_display['ML_Prediction'],
                    label='ML Prediction', color='red', linewidth=2, linestyle='--', alpha=0.8)
    
    # Trend shading
    valid_mask = (~display_data[f'MA{ma_short}'].isna()) & (~display_data[f'MA{ma_long}'].isna())
    uptrend = (display_data['Signal'] == 1) & valid_mask
    downtrend = (display_data['Signal'] == -1) & valid_mask
    
    if uptrend.any():
        ax1.fill_between(display_data.index, 0, display_data['Close'].max() * 1.1,
                         where=uptrend, color='lightgreen', alpha=0.2, label='Uptrend')
    if downtrend.any():
        ax1.fill_between(display_data.index, 0, display_data['Close'].max() * 1.1,
                         where=downtrend, color='lightcoral', alpha=0.2, label='Downtrend')
    
    ax1.set_title(f'{symbol} Stock Price Analysis', fontsize=14)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(FuncFormatter(price_formatter))
    
    # Volume plot
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.bar(display_data.index, display_data['Volume'], color='blue', alpha=0.5, label='Volume')
    ax2.plot(display_data.index, display_data['Volume_MA20'], color='red', label='20-Day Volume MA', linewidth=1)
    ax2.set_ylabel('Volume', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # RSI plot
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(display_data.index, display_data['RSI'], color='purple', label='RSI')
    ax3.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax3.set_ylabel('RSI', fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # MACD plot
    ax4 = plt.subplot(gs[3], sharex=ax1)
    ax4.plot(display_data.index, display_data['MACD'], label='MACD', color='blue')
    ax4.plot(display_data.index, display_data['MACD_signal'], label='Signal Line', color='red')
    
    macd_hist_positive = display_data['MACD_hist'] >= 0
    ax4.bar(display_data.index[macd_hist_positive], display_data['MACD_hist'][macd_hist_positive],
            color='green', width=1, alpha=0.5)
    ax4.bar(display_data.index[~macd_hist_positive], display_data['MACD_hist'][~macd_hist_positive],
            color='red', width=1, alpha=0.5)
    
    ax4.set_ylabel('MACD', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
    
    # Save chart
    if save:
        output_filename = f"{symbol}_analysis.png"
        plt.savefig(output_filename, dpi=dpi)
        print(f"Chart saved as {output_filename}")
    
    # Show chart
    if show:
        plt.show()
    
    return fig
