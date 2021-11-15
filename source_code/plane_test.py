import numpy as np

p1 = np.array([1, 0, 0])
p2 = np.array([0, 1, 0])
p3 = np.array([0, 0, 1])
one = np.array([[1], [1], [1], [1]])

p = np.array([0, 0, 0])

A0 = np.matrix([p1, p2, p3])
A = np.vstack([p, A0])
#or: A = np.vstack([p, p1, p2, p3])

A = np.append(A, one, axis=1)

print(A)
print(np.linalg.det(A))