from jelly import Tetrahedron
import matplotlib.pyplot as plt

expected_probability = [0.1, 0.2, 0.3, 0.4]
err = 0.05
N = 1000
treshold = N * err**2
print(f"treshold = {treshold}")

tetrahedron = Tetrahedron()

#print("lambda = 1.0")
#y1 = tetrahedron.estimateBodyFace2(expected_probability, 1.0, treshold, N)[1]
print("lambda = 2.5")
y2 = tetrahedron.estimateBodyFace2(expected_probability, 2.5, treshold, N)[1]

#plt.plot(y1, 'r')
plt.plot(y2, 'g')
plt.xlabel("iteration")
plt.ylabel("mse")
#plt.legend(["lambda = 1.0", "lambda = 10.0"])
plt.savefig('graphs/lambatest')