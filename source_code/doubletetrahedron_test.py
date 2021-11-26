from jelly import DoubleTetrahedron
from scipy.stats import chi2
from scipy.stats import chisquare

err = 0.05
N = 1000
threshold = N * err**2
alpha = 0.05 #szignifikancia szint
lambda_ = 1.0

doubletetrahedron = DoubleTetrahedron()

df = doubletetrahedron.n - 1
P = chi2.isf(alpha, df)

#test 01
expected_probability = [1/12, 2/12, 3/12, 3/12, 2/12, 1/12]
print(f"1st test\n{expected_probability}")

dt1 = doubletetrahedron.estimateBodyFace2(expected_probability, lambda_, threshold, N)[0]

measured_frequency = dt1.estimateFrequencies(1000)
[ khi2, pvalue ] = chisquare(measured_frequency, f_exp=[250/3, 500/3, 250, 250, 500/3, 250/3])
print(f"{measured_frequency}, chi2 = {khi2}, level of significance = {1 - pvalue}, {khi2 < P}")

dt1.saveObj("doubletetrahedron_01")

#test 02
expected_probability = [2/15, 2/15, 2/15, 2/15, 2/15, 1/3]
print(f"2st test\n{expected_probability}")

dt2 = doubletetrahedron.estimateBodyFace2(expected_probability, lambda_, threshold, N)[0]

measured_frequency = dt2.estimateFrequencies(1000)
[ khi2, pvalue ] = chisquare(measured_frequency, f_exp=[400/3, 400/3, 400/3, 400/3, 400/3, 1000/3])
print(f"{measured_frequency}, chi2 = {khi2}, level of significance = {1 - pvalue}, {khi2 < P}")

dt2.saveObj("doubletetrahedron_02")