from jelly import *

N = 100

expected = [0.1, 0.2, 0.3, 0.4]

tetrahedron = Tetrahedron()

tetrahedron.saveObj("test")

#t2 = tetrahedron.estimateBodyFace2(expected, 0.8, N)
#
#stat = t2.droptest(1000)
#e = [0] * len(expected)
#for i in range(len(e)):
#    e[i] = 1000 * expected[i]
#print("p = {}".format(chisquare(stat, f_exp=e)[1:][0]))


#tetrahedron.estimateBody(expected, 0.8, N)

#stat = tetrahedron.droptest(N)
#t2 = tetrahedron.getFaceModified(expected, stat)

#print(stat)
#print("p = {}".format(chisquare(stat)[1:][0]))
#print("p = {}".format(chisquare(stat, f_exp=expected)[1:][0]))