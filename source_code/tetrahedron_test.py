from jelly import Tetrahedron
from scipy.stats import chi2
from scipy.stats import chisquare

err = 0.05
N = 1000
threshold = N * err**2
alpha = 0.05 #szignifikancia szint
lambda_ = 1.0

tetrahedron = Tetrahedron()

df = len(tetrahedron.faces) - 1
P = chi2.isf(alpha, df)

#test 01
expected_probability = [0.1, 0.2, 0.3, 0.4]
print(f"1st test\n{expected_probability}")

t1 = tetrahedron.estimateBodyFace2(expected_probability, lambda_, threshold, N)[0]

measured_frequency = t1.estimateFrequencies(1000)
[ khi2, pvalue ] = chisquare(measured_frequency, f_exp=[100, 200, 300, 400])
print(f"{measured_frequency}, chi2 = {khi2}, {khi2 < P}")

t1.saveObj("tetrahedron_01")

#dark souls: board game - orange dice
expected_probability = [1/6, 1/3, 1/3, 1/6]
print(f"2st test\n{expected_probability}")

t2 = tetrahedron.estimateBodyFace2(expected_probability, lambda_, threshold, N)[0]

measured_frequency = t2.estimateFrequencies(1000)
[ khi2, pvalue ] = chisquare(measured_frequency, f_exp=[1000/6, 1000/3, 1000/3, 1000/6])
print(f"{measured_frequency}, chi2 = {khi2}, {khi2 < P}")

t2.saveObj("tetrahedron_02")