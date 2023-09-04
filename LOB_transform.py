import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns

lob = pd.read_csv('./fake_lob/fake_lob_INTC.csv',header = 0,index_col=0)
lob_ = pd.read_csv('./fake_lob/real_lob_INTC.csv',header = 0,index_col=0)
lob.set_index('20',drop=True,inplace=True)
lob_.set_index(lob.index,inplace=True)
lob.iloc[:,list(range(1,21,2))] = (lob.iloc[:,list(range(1,21,2))] - lob_.iloc[:,list(range(1,21,2))]).abs()
lob = lob[(lob.index>46800)&(lob.index<54000)]
lob.iloc[:,list(range(0,20,2))] = lob.iloc[:,list(range(0,20,2))] / 100
lob.iloc[:,list(range(3,23,4))] = -lob.iloc[:,list(range(3,23,4))]
col_names = np.arange(start=lob.iloc[:,18].min(),stop=lob.iloc[:,16].max(),step=1)
df = pd.DataFrame(np.zeros((len(lob.index),len(col_names))),index=lob.index,columns=col_names)
for i in range(10):
    selected = lob.iloc[:,[2*i,2*i+1]]
    selected.columns = ['price','volume']
    pivoted = selected.pivot_table(values='volume',index=lob.index,columns='price')
    df.update(pivoted)

mid_prices = []
for index, row in df.iterrows():
    neg_prices = [price for price in row.index if row[price] < 0]
    pos_prices = [price for price in row.index if row[price] > 0]
    mid_price = (max(neg_prices) + min(pos_prices)) / 2
    mid_prices.append(mid_price)

def map_to_color(value,v_max):
    if value == 0:
        return 'white'
    elif value > 0:  # Ask order
        intensity = min(1, value / (v_max*100))  # Maximum intensity for red
        return mcolors.to_hex((1, 1 - intensity, 1 - intensity))
    else:  # Bid order
        intensity = min(1, abs(value) / (v_max*100))  # Maximum intensity for blue
        return mcolors.to_hex((1 - intensity, 1 - intensity, 1))


fig, ax = plt.subplots()

sns.lineplot(pd.DataFrame({'time':list(df.index),'price':mid_prices}),x='time',y='price',linewidth=2.5, color='black',ax=ax)

# Get the timestamps as numerical values for the y-axis
timestamps_num = mdates.date2num(pd.to_datetime(df.index,unit='s'))

# Create the heatmap
v_max = 250
cax = ax.imshow(df.T, cmap='coolwarm', aspect='auto', extent=[timestamps_num.min(), timestamps_num.max(), df.columns.min(), df.columns.max()], vmin=-v_max, vmax=v_max)

# Apply colors to the cells based on order volume values
for i in range(len(df.columns)):
    for j in range(len(df.index)):
        value = df.iat[j, i]
        cell_color = map_to_color(value,v_max)
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
ax.set_ylabel('Order Price ($)')
ax.set_xlabel('Time')

# Add color bar
cbar = plt.colorbar(cax)
cbar.set_label('Order Volume ($10^{2}$)')

# Get current y-ticks
current_yticks = ax.get_yticks()

# Divide y-ticks by 100 and set them back as new ytick labels
new_yticklabels = current_yticks / 100
new_yticklabels = ['{:.2f}'.format(label) for label in new_yticklabels]
ax.set_yticklabels(new_yticklabels)

ax.set_title('Volume heatmap: absolute error',fontsize=14)
plt.tight_layout()
# Show the plot
plt.show()


