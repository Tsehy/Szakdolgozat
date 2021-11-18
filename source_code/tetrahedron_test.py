from jelly import Tetrahedron
from scipy.stats import chisquare

N = 100

expected_probability = [0.1, 0.2, 0.3, 0.4]

tetrahedron = Tetrahedron()

print(tetrahedron.isConvex())

#t2 = tetrahedron.estimateBodyRandom(expected_probability, 0.00001, N)
t2 = tetrahedron.estimateBodyFace(expected_probability, 0.00001, N)
#t2 = tetrahedron.estimateBodyFace2(expected_probability, 0.00001, N)

measured_frequency = t2.estimateFrequencies(1000)

expected_frequency = [0] * len(expected_probability)
for i in range(len(expected_probability)):
    expected_frequency = 1000 * expected_probability

print(f"{measured_frequency}, p = {chisquare(measured_frequency, f_exp=expected_frequency)[1:][0]}")