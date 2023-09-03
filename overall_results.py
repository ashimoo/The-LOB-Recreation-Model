import numpy as np
import os

folder_path = './logs/'

stocks = ['MSFT','INTC','JPM']
modules = ['ode']
sides = ['ask','bid']

for stock in stocks:
    for side in sides:
        for module in modules:

            val = []
            test = []
            for i in range(5):
                with open(folder_path+'_'.join([stock,side,module,str(i)])+'_True_False_ex.log','r') as file:
                    lines = file.readlines()
                    val.append(float(lines[-2].split(' ')[-1][:-2]))
                    test.append(float(lines[-1].split(' ')[-1][:-2]))
                    file.close()
            print(stock,side,module,np.mean(val),np.mean(test))

            # val = []
            # test = []
            # for i in range(5):
            #     with open(folder_path+'_'.join([stock,side,module,str(i)])+'_True_False.log','r') as file:
            #         lines = file.readlines()
            #         val.append(float(lines[-2].split(' ')[-1][:-2]))
            #         test.append(float(lines[-1].split(' ')[-1][:-2]))
            #         file.close()
            # print(stock,side,module,np.mean(val),np.mean(test))
            #
            # val = []
            # test = []
            # for i in range(5):
            #     with open(folder_path+'_'.join([stock,side,module,str(i)])+'_True_{}_nows.log'.format("true"),'r') as file:
            #         lines = file.readlines()
            #         val.append(float(lines[-2].split(' ')[-1][:-2]))
            #         test.append(float(lines[-1].split(' ')[-1][:-2]))
            #         file.close()
            # print(stock,side,module,np.mean(val),np.mean(test))
            #
            # val = []
            # test = []
            # for i in range(5):
            #     with open(folder_path+'_'.join([stock,side,module,str(i)])+'_True_{}.log'.format("true"),'r') as file:
            #         lines = file.readlines()
            #         val.append(float(lines[-2].split(' ')[-1][:-2]))
            #         test.append(float(lines[-1].split(' ')[-1][:-2]))
            #         file.close()
            # print(stock,side,module,np.mean(val),np.mean(test))

# stocks = ['MSFT','INTC','JPM']
# modules = ['ode','simple','decay','attention']
# sides = ['ask','bid']
#
# for stock in stocks:
#     for side in sides:
#         for module in modules:
#             val = []
#             test = []
#             for i in range(5):
#                 with open(folder_path + '_'.join([stock, side, module, str(i)]) + '_True_{}.log'.format(module!='attention'),
#                           'r') as file:
#                     lines = file.readlines()
#                     val.append(float(lines[-2].split(' ')[-1][:-2]))
#                     test.append(float(lines[-1].split(' ')[-1][:-2]))
#                     file.close()
#             print(stock, side, module, np.mean(val), np.mean(test))

