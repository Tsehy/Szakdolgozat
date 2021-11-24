#%%
import numpy as np
from scipy.stats import chi2
from scipy.stats import chisquare
import matplotlib.pyplot as plt

df = 3
rv = chi2(df)

vals = chi2.ppf([0.001, 0.5, 0.999], df)
np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))

stat = [101, 189, 317, 393]
e = [100, 200, 300, 400]

k2 = 0
for i in range(len(e)):
    k2 += (stat[i] - e[i])**2 / e[i]
    
print(k2)
print(chisquare(stat, f_exp=e))

p = 1 - chi2.cdf(k2, df)
print(p)

x = np.linspace(0, 10, 1000)
alpha = 0.05
k2 = chisquare(stat, f_exp=e)[:1][0]
y = chi2.sf(x, df)
plt.plot(x, y)
plt.hlines(alpha, 0, 10, colors='r')
plt.vlines(chi2.isf(alpha, df), 0, 1, colors='r')
plt.vlines(k2, 0, 1, colors='g')

# %%
