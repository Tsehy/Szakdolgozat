#A mért MSE és chi2 értékek legkisebb négyzetek módszerével történő összehasonlítása
import numpy as np

mse = [0.012605000000000002,
0.005730000000000002,
0.004737500000000001,
0.0024330000000000007,
0.001435500000000001,
0.0010555000000000004,
0.00019650000000000003,
0.0010955000000000001,
7.399999999999977e-05]
m = np.array(mse)[np.newaxis]
o = [1] * len(mse)
one = np.array(o)[np.newaxis]

chi2 = [309.3075,
133.22333333333333,
123.45833333333333,
69.02083333333334,
33.730000000000004,
16.908333333333335,
3.2125,
21.553333333333335,
1.0333333333333332]

A = np.hstack([one.T, m.T])
b = np.array(chi2)[np.newaxis]
b = b.T

ATA = A.T.dot(A)
ATb = A.T.dot(b)
ATAinv = np.linalg.inv(ATA)

a = ATAinv.dot(ATb)

print(a)
# %%
