from jelly import Octahedron
from scipy.stats import chi2
from scipy.stats import chisquare

err = 0.05
N = 1000
threshold = N * err**2
alpha = 0.05 #szignifikancia szint
lambda_ = 1.0

octahedron = Octahedron()

df = len(octahedron.faces) - 1
P = chi2.isf(alpha, df)

#test 01
expected_probability = [1/128, 7/128, 21/128, 35/128, 35/128, 21/128, 7/128, 1/128]
print(f"1st test\n{expected_probability}")

o1 = octahedron.estimateBodyFace2(expected_probability, lambda_, threshold, N)[0]

measured_frequency = o1.estimateFrequencies(1000)
[ khi2, pvalue ] = chisquare(measured_frequency, f_exp=[125/16, 875/16, 2625/16, 4375/16, 4375/16, 2625/16, 875/16, 125/16])
print(f"{measured_frequency}, chi2 = {khi2}, {khi2 < P}")

o1.saveObj("octahedron_01")

#test 02
expected_probability = [1/4, 1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/4]
print(f"2st test\n{expected_probability}")

o2 = octahedron.estimateBodyFace2(expected_probability, lambda_, threshold, N)[0]

measured_frequency = o2.estimateFrequencies(1000)
[ khi2, pvalue ] = chisquare(measured_frequency, f_exp=[250, 250/3, 250/3, 250/3, 250/3, 250/3, 250/3, 250])
print(f"{measured_frequency}, chi2 = {khi2}, {khi2 < P}")

o2.saveObj("octahedron_02")