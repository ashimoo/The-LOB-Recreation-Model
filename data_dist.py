import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats.mstats import winsorize
import utils
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec



for index, dataset in enumerate(['INTC','MSFT','JPM']):
    path_lob_train = './LOBSTER/{}_orderbook_part_5.csv'.format(dataset)
    path_msb_train = './LOBSTER/{}_message_part_5.csv'.format(dataset)

    lob, _ = utils.time_parser(path_lob_train, path_msb_train)

    selected = np.array(lob.iloc[:,[1,5,9,13,17]])
    selected = winsorize(selected,limits=[0.005,0.005])
    lob.iloc[:,[1,5,9,13,17]] = np.ceil(selected / 100.0)
    selected = np.array(lob.iloc[:,[3,7,11,15,19]])
    selected = winsorize(selected,limits=[0.005,0.005])
    lob.iloc[:,[3,7,11,15,19]] = np.ceil(selected / 100.0)

    data_ask = []
    data_bid = []
    lob=np.array(lob)
    for i in range(len(lob)):
        for j in range(0,20,4):
            data_ask.append([lob[i,j]/10000,lob[i,j+1]])
            data_bid.append([lob[i,j+2]/10000,lob[i,j+3]])
    data_ask=np.vstack(data_ask)
    data_bid=np.vstack(data_bid)
    g1 = sns.jointplot(x=data_bid[:,0],y=data_bid[:,1], kind="hex", gridsize=15,color="#4CB391")
    g2 = sns.jointplot(x=data_ask[:,0],y=data_ask[:,1], kind="hex", gridsize=15,color="#4CB391")
    data_ls = [data_bid,data_ask]
    for i,g in enumerate([g1,g2]):
        # Access the marginal axes
        ax_marg_x = g.ax_marg_x
        ax_marg_y = g.ax_marg_y

        # Clear the automatically generated histograms
        ax_marg_x.clear()
        ax_marg_y.clear()

        # Create new histograms with custom number of bins
        ax_marg_x.hist(data_ls[i][:, 0], bins=15, color="#4CB391", edgecolor="k")
        ax_marg_y.hist(data_ls[i][:, 1], bins=15, color="#4CB391", edgecolor="k", orientation="horizontal")

        # Remove y-axis labels for the histograms
        ax_marg_x.set_yticklabels([])
        ax_marg_y.set_xticklabels([])


        g.set_axis_labels("price ($)", "volume (in hundreds)",fontsize=16)
        title = g.fig.suptitle("{} {}".format(dataset,'BID' if i==0 else 'ASK') )  # Adjust y position here
        title.set_fontsize(18)
        # Adjust layout to make room for labels and title
        g.fig.tight_layout()

        # Additional space for title
        g.fig.subplots_adjust(top=0.9)

plt.show()
