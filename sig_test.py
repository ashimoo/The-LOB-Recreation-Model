import numpy as np
import scipy.stats

a = np.load('./ode_ask_full_pred.npy')
b = np.load('./ode_ask_ES_pred.npy')
c = np.load('./ode_ask_HC_pred.npy')
d = np.load('./ode_ask_0.5_pred.npy')
base = np.load('./ode_ask_real.npy')
a = np.abs(a - base)
b = np.abs(b - base)
c = np.abs(c - base)
d = np.abs(d - base)
for i in [a,b,c,d]:
    print(np.std(i))

statistic, p = scipy.stats.wilcoxon(a, y=d, zero_method='wilcox', correction=False, alternative='less')
dd = 0
