import pandas as pd
import numpy as np

date_list = [20,21,22,25,26]
stock = ['INTC','MSFT','JPM']
for stk in stock:
    for index,date in enumerate(date_list):
        path_lob = './LOBSTER/{}_2012-06-20_2012-06-26_10/{}_2012-06-{}_34200000_57600000_orderbook_10.csv'.format(stk,stk,date)
        path_msb = './LOBSTER/{}_2012-06-20_2012-06-26_10/{}_2012-06-{}_34200000_57600000_message_10.csv'.format(stk,stk,date)

        with open(path_lob) as f:
            data_lob = pd.read_csv(f, header=None)
        with open(path_msb) as f:
            data_msb = pd.read_csv(f, header=None).iloc[:,:-1]
            data_msb.columns = ['time', 'type', 'index', 'quantity', 'price', 'direction']

        data_combined = data_lob.join(data_msb)
        data_combined = data_combined[(data_combined['time']>36000) & (data_combined['time']<55800)]
        data_combined = np.array(data_combined)
        trimmed_list = []

        for i in range(data_combined.shape[0]):
            max = data_combined[i, 16]
            min = data_combined[i, 18]
            if data_combined[i,-2] >=min and data_combined[i,-2] <=max:
                trimmed_list.append(data_combined[i,:])
        trimmed_array = np.stack(trimmed_list)
        data_lob_5 = pd.DataFrame(trimmed_array[:,list(range(20))])
        data_msb_5 = pd.DataFrame(trimmed_array[:,40:])

        data_lob_5.to_csv('./LOBSTER/{}_orderbook_part_{}.csv'.format(stk,index+1),header=0,index=0)
        data_msb_5.to_csv('./LOBSTER/{}_message_part_{}.csv'.format(stk,index+1),header=0,index=0)