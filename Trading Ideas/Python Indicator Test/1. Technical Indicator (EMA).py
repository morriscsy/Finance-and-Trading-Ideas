import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#------------------------------------------------------------------------------------

# Download historical data
stock_symbol = '1799.HK'
start_date = '2022-10-01'
end_date = '2023-3-20'

data = yf.download(stock_symbol, start=start_date, end=end_date)
data.reset_index(inplace=True)

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

#------------------------------------------------------------------------------------

# Calculate trading position (1 for buy, -1 for sell)
rolling_mean = data['Close'].rolling(window=20).mean()
rolling_std = data['Close'].rolling(window=20).std()

#[**** Create z_score Variable ***]
data['z_score'] = (data['Close'] - rolling_mean) / rolling_std
#[**** Create position Variable ***]
data['position'] = np.where(data['z_score'] > 1, -1, np.where(data['z_score'] < -1, 1, 0))

# Identify buy and sell pairs
#[**** Create buy_signal Variable ***]
data['buy_signal'] = np.where((data['position'] == 1) & (data['position'].shift(-1) == -1), data['Close'], np.nan)
#[**** Create sell_signal Variable ***]
data['sell_signal'] = np.where((data['position'] == -1) & (data['position'].shift(1) == 1), data['Close'], np.nan)

# Save stock chart with buy and sell points as PNG
stock_chart_fig = plt.figure(figsize=(10, 6))
stock_chart_ax = stock_chart_fig.add_subplot(111)
stock_chart_ax.plot(data['Date'], data['Close'], label='Price')
stock_chart_ax.scatter(data[data['position'] == 1]['Date'], data[data['position'] == 1]['Close'],
                       marker='^', color='green', label='Buy')
stock_chart_ax.scatter(data[data['position'] == -1]['Date'], data[data['position'] == -1]['Close'],
                       marker='v', color='red', label='Sell')
stock_chart_ax.plot(data['Date'], data['buy_signal'], marker='^', markersize=8, linewidth=0, color='green')
stock_chart_ax.plot(data['Date'], data['sell_signal'], marker='v', markersize=8, linewidth=0, color='red')

# Draw lines connecting buy and sell points
buy_dates = data[data['position'] == 1]['Date']
sell_dates = data[data['position'] == -1]['Date']
buy_prices = data[data['position'] == 1]['Close']
sell_prices = data[data['position'] == -1]['Close']
for buy_date, buy_price, sell_date, sell_price in zip(buy_dates, buy_prices, sell_dates, sell_prices):
    stock_chart_ax.plot([buy_date, sell_date], [buy_price, sell_price], color='grey', linestyle='--')

stock_chart_ax.set_title('Stock Chart with Buy and Sell Points')
stock_chart_ax.set_xlabel('Date')
stock_chart_ax.set_ylabel('Price')
stock_chart_ax.legend()
stock_chart_ax.grid(True)
stock_chart_fig.savefig('stock_chart.png')
plt.show()

#------------------------------------------------------------------------------------

# Calculate cumulative returns
#[**** Create returns ***]
data['returns'] = data['Close'].pct_change().fillna(0)
#[**** Create cumulative_returns Variable ***]
data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1

# Save cumulative returns chart as PNG
cumulative_returns_fig = plt.figure(figsize=(10, 6))
cumulative_returns_ax = cumulative_returns_fig.add_subplot(111)
cumulative_returns_ax.plot(data['Date'], data['cumulative_returns'])
cumulative_returns_ax.set_title('Cumulative Returns')
cumulative_returns_ax.set_xlabel('Date')
cumulative_returns_ax.set_ylabel('Cumulative Returns')
cumulative_returns_ax.grid(True)
cumulative_returns_fig.savefig('cumulative_returns_chart.png')
plt.show()

#------------------------------------------------------------------------------------

# Generate classification report
y_true = np.where(data['returns'] > 0, 1, 0)
y_pred = np.where(data['position'] == 1, 1, 0)
classification_report_str = classification_report(y_true, y_pred, output_dict=True)

# Create a table from the classification report dictionary
table_data = []
table_data.append(['', 'precision', 'recall', 'f1-score', 'support'])
for class_label, metrics in classification_report_str.items():
    if class_label.isdigit():
        row = [class_label]
        row.extend([metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])
        table_data.append(row)

# Save classification report as PNG
classification_report_fig = plt.figure()
classification_report_ax = classification_report_fig.add_subplot(111)
classification_report_ax.axis('off')
classification_report_ax.table(cellText=table_data,
                               colLabels=table_data[0],
                               cellLoc='center',
                               loc='center')
classification_report_fig.savefig('classification_report.png')
plt.show()

 