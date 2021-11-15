#%%
import numpy as np
from scipy.stats import chi2
from scipy.stats import chisquare

df = 3
rv = chi2(df)

vals = chi2.ppf([0.001, 0.5, 0.999], df)
np.allclose([0.001, 0.5, 0.999], chi2.cdf(vals, df))

stat = [69, 205, 279, 347]
e = [90, 180, 270, 360]

k2 = 0
for i in range(len(e)):
    k2 += (stat[i] - e[i])**2 / e[i]
    
print(k2)
print(chisquare(stat, f_exp=e))

p = 1 - chi2.cdf(k2, df)
print(p)

# %%
