import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Sample data for the DataFrame (replace this with your actual data)
df = pd.read_pickle('./fake_lob/ORDERBOOK_ABM_FULL.bz2').iloc[500:20000,:]

mid_prices = []
for index, row in df.iterrows():
    neg_prices = [price for price in row.index if row[price] < 0]
    pos_prices = [price for price in row.index if row[price] > 0]
    mid_price = (max(neg_prices) + min(pos_prices)) / 2
    mid_prices.append(mid_price)

def map_to_color(value):
    if value == 0:
        return 'white'
    elif value > 0:  # Ask order
        intensity = min(1, value / 30000)  # Maximum intensity for red
        return mcolors.to_hex((1, 1 - intensity, 1 - intensity))
    else:  # Bid order
        intensity = min(1, abs(value) / 30000)  # Maximum intensity for blue
        return mcolors.to_hex((1 - intensity, 1 - intensity, 1))


fig, ax = plt.subplots()

sns.lineplot(pd.DataFrame({'time':list(df.index),'price':mid_prices}),x='time',y='price',linewidth=2.5, color='black',ax=ax)

# Get the timestamps as numerical values for the y-axis
timestamps_num = mdates.date2num(pd.to_datetime(df.index))

# Create the heatmap
cax = ax.imshow(df.T, cmap='coolwarm', aspect='auto', extent=[timestamps_num.min(), timestamps_num.max(), df.columns.min(), df.columns.max()], vmin=-500, vmax=500)

# Apply colors to the cells based on order volume values
for i in range(len(df.columns)):
    for j in range(len(df.index)):
        value = df.iat[j, i]
        cell_color = map_to_color(value)
        ax.add_patch(plt.Rectangle((timestamps_num[j], df.columns[i]-0.5), 5, 10, facecolor=cell_color, edgecolor='none'))

# Set y-axis labels as price levels
ax.set_yticks(df.columns)
ax.set_ylim(df.columns.min() - 0.5, df.columns.max() + 0.5)
ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

# Set x-axis as datetime format
ax.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

# Hide grid lines
ax.grid(False)

# Set labels for axes
ax.set_ylabel('Order Price')
ax.set_xlabel('Time')

# Add color bar
cbar = plt.colorbar(cax)
cbar.set_label('Order Volume')

# Show the plot
plt.show()

