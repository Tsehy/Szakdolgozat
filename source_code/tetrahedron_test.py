from jelly import Tetrahedron

N = 100

expected = [0.1, 0.2, 0.3, 0.4]

tetrahedron = Tetrahedron()

print(tetrahedron.isConvex())

#t2 = tetrahedron.estimateBodyFace2(expected, 0.8, N)

#tetrahedron.saveObj("tetrahedron_expected_01_02_03_04")

#tetrahedron.estimateBody(expected, 0.8, N)

#stat = tetrahedron.droptest(N)
#t2 = tetrahedron.getFaceModified(expected, stat)

#print(stat)
#print("p = {}".format(chisquare(stat)[1:][0]))
#print("p = {}".format(chisquare(stat, f_exp=expected)[1:][0]))