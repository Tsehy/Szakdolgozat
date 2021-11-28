from jelly import Tetrahedron
import numpy as np
import matplotlib.pyplot as plt

expected_probability = [0.1, 0.2, 0.3, 0.4]
err = 0.05
N = 1000
threshold = N * err**2
print(f"threshold = {threshold}")

tetrahedron = Tetrahedron()
iteriter = []
L = np.linspace(0.5, 2.0, 16)

for i in range(200):
    iter = []
    for lambda_ in L:
        print(f"lambda = {lambda_}")
        y = tetrahedron.estimateBodyFace2(expected_probability, lambda_, threshold, N)[1]
        iter.append(len(y))
    print(i, iter)
    iteriter.append(iter)

print(iteriter)

#plt.plot(L, iter)
#plt.xlabel("lambda")
#plt.ylabel("iteration")
#plt.savefig('graphs/lambatest')[7, 9, 11, 7, 4, 3, 4, 2, 4, 6, 5, 14, 4, 8, 4, 3]