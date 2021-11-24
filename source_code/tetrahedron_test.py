from jelly import Tetrahedron
from scipy.stats import chi2
from scipy.stats import chisquare

expected_probability = [0.1, 0.2, 0.3, 0.4]
err = 0.05
N = 1000
treshold = N * err**2
alpha = 0.05 #szignifikancia szint

df = len(expected_probability) - 1
P = chi2.isf(alpha, df)

tetrahedron = Tetrahedron()

#print(tetrahedron.isConvex())

t2 = tetrahedron.estimateBodyRandom(expected_probability, 300, 200)
#t2 = tetrahedron.estimateBodyFace(expected_probability, treshold, N)
#t2, y1 = tetrahedron.estimateBodyFace2(expected_probability, lambda_ = 1, treshold, N)

measured_frequency = t2.estimateFrequencies(200)
[ khi2, pvalue ] = chisquare(measured_frequency, f_exp=[20, 40, 60, 80])
print(f"{measured_frequency}, chi2 = {khi2}, level of significance = {1 - pvalue}, {khi2 < P}")

measured_frequency = t2.estimateFrequencies(1000)
[ khi2, pvalue ] = chisquare(measured_frequency, f_exp=[100, 200, 300, 400])
print(f"{measured_frequency}, chi2 = {khi2}, level of significance = {1 - pvalue}, {khi2 < P}")
