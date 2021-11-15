import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from random import uniform
from random import randrange
from scipy.stats import chisquare

TIME = 1
GRAVITY = np.array([0, 0, -0.01], dtype=float)
ZMIN = 0.001 #if the z value is less than this value it is considered on the ground

class Particle:
    def __init__(self, a = [0, 0, 0]):
        self.mass = 1
        self.position = np.array(a, dtype=float) 
        self.velocity = np.array([0, 0, 0], dtype=float)
        self.force = np.array([0, 0, 0], dtype=float)
        #self.correction = 0.0
        
    def __str__(self):
        return "({:.3f}, {:.3f}, {:.3f})".format(self.position[0], self.position[1], self.position[2])
        
class Spring:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.length = np.linalg.norm(self.b.position - self.a.position)
        
    def __str__(self):
        return "{}, {}, {}".format(self.a, self.b, self.length)
    
class Dice:
    def __init__(self):
        self.initSprings()

    def initVerticies(self):
        raise NotImplementedError

    def initSprings(self):
        self.springs = []
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                self.springs.append(Spring(self.particles[i], self.particles[j]))

    def getCenter(self):
        center = np.array([0, 0, 0], dtype=float)
        for particle in self.particles:
            center += particle.position
        center /= self.n
        return center

    def update(self, time, gravity):
        #set gravity
        for particle in self.particles:
            particle.force = np.array(gravity)
        
        #use actual velocity
        for particle in self.particles:
            particle.force += particle.velocity
            
        #use springs
        for spring in self.springs:
            currentLength = np.linalg.norm(spring.b.position - spring.a.position)
            
            #dx = spring.b.position[0] - spring.a.position[0]
            #dy = spring.b.position[1] - spring.a.position[1]
            #dz = spring.b.position[2] - spring.a.position[2]
            #print(dx, dy, dz)
            #currentLength = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            unit = spring.b.position - spring.a.position
            unit /= currentLength
            
            diff = spring.length - currentLength
            alpha = 0.1
            correction = diff * diff * alpha
            if diff < 0:
                correction *= -1
            #self.correction = correction
            unit = unit * correction
                
            spring.a.force -= unit
            spring.b.force += unit
            
        #use ground
        for particle in self.particles:
            if particle.position[2] < 0:
                particle.position[2] = 0
                particle.force[2] = 0
                #particle.force *= 0.99
                
        #update velocities
        for particle in self.particles:
            particle.velocity = particle.force * time
            
        #update positions
        for particle in self.particles:
            particle.position += particle.velocity * time
            
    def translate(self, vector):
        for particle in self.particles:
            particle.position += vector
         
    def rotate(self, xAlpha, yAlpha, zAlpha): #rad!
        cx = math.cos(xAlpha)
        sx = math.sin(xAlpha)
        cy = math.cos(yAlpha)
        sy = math.sin(yAlpha)
        cz = math.cos(zAlpha)
        sz = math.sin(zAlpha)

        Rx = np.array([[1,  0,  0, 0],
                       [0, cx,-sx, 0],
                       [0, sx, cx, 0],
                       [0,  0,  0, 1]])
        Ry = np.array([[ cy, 0, sy, 0],
                       [  0, 1,  0, 0],
                       [-sy, 0, cy, 0],
                       [  0, 0,  0, 1]])
        Rz = np.array([[cz,-sz, 0, 0],
                       [sz, cz, 0, 0],
                       [ 0,  0, 1, 0],
                       [ 0,  0, 0, 1]])

        R = Rz.dot(Ry).dot(Rx)
        
        for particle in self.particles:
            v = np.append(particle.position, 0)[np.newaxis]
            particle.position = R.dot(v.T).T[0][:3]
            
    def scale(self, a):
        for particle in self.particles:
            particle.position *= a

    def randRotate(self):
        #particles[0] is the pivot
        #we assume the origo is the center of the object
        #this means only apply this after translate(-c) !

        #random point on the sphere surface
        #hivatkozás!
        while True:
            r_point = np.array([np.random.normal() for _ in range(3)]) #with normal distribution
            if np.linalg.norm(r_point) != 0:
                break
        radius = np.linalg.norm(self.particles[0].position)
        r_point *= radius / np.linalg.norm(r_point)

        #rotate pivot onto the x axis (radius, 0, 0)
        if self.particles[0].position[1] != 0:
            temp_r = math.sqrt(self.particles[0].position[0]**2 + self.particles[0].position[1]**2)
            fi_z = np.arccos(self.particles[0].position[0] / temp_r)
            if self.particles[0].position[1] > 0:
                fi_z *= -1
            self.rotate(0, 0, fi_z)

        if self.particles[0].position[2] != 0:
            temp_r = math.sqrt(self.particles[0].position[0]**2 + self.particles[0].position[2]**2)
            fi_y = np.arccos(self.particles[0].position[0] / temp_r)
            if self.particles[0].position[2] > 0:
                fi_y *= -1
            self.rotate(0, -fi_y, 0)

        #random rotation arount the x axis
        fi_x = uniform(0, 2*math.pi)
        self.rotate(fi_x, 0, 0)

        #rotate pivot to the random point
        #1st rotate around the y axis until pivot.z = r_point.z
        if self.particles[0].position[2] != r_point[2]:
            a = np.array([self.particles[0].position[0], self.particles[0].position[2]])
            rx = math.sqrt(radius**2 - r_point[2]**2)
            b = np.array([rx, r_point[2]])
            a /= np. linalg.norm(a)
            b /= np. linalg.norm(b)
            fi_y = np.arccos(np.dot(a, b))
            if r_point[2] > 0:  
                fi_y *= -1
            self.rotate(0, fi_y, 0)

        #2nd rotate around the z axis
        if self.particles[0].position[1] != r_point[1]:
            a = np.array([self.particles[0].position[0], self.particles[0].position[1]])
            b = np.array([r_point[0], r_point[1]])
            a /= np. linalg.norm(a)
            b /= np. linalg.norm(b)
            fi_z = np.arccos(np.dot(a, b))
            if r_point[1] < 0:  
                fi_z *= -1
            self.rotate(0, 0, fi_z)

        #with np.printoptions(precision=3, suppress=True):
        #    print(r_point)
        #    print(self.particles[0].position)

    def getTotalVelocityLength(self):
        totalVelocityLength = 0
        for particle in self.particles:
            totalVelocityLength += abs(np.linalg.norm(particle.velocity))
        return totalVelocityLength
            
    def isStopped(self):
        ground = 0
        for particle in self.particles:
            if particle.position[2] < ZMIN:
                ground += 1
        totalVelocityLength = 0
        for particle in self.particles:
            totalVelocityLength += abs(np.linalg.norm(particle.velocity))
        ground
        totalVelocityLength
        return ((ground > 2) and (totalVelocityLength < 0.13))

    def faceOnGround(self):
        #returns the index of the face on the ground
        # -1 if not stopped
        # -2 if no match found
        # -3 of multiple mach is found
        if not self.isStopped():
            print("not stopped")
            return -1
        
        onGround = set()
        for i in range(self.n):
            if self.particles[i].position[2] < ZMIN:
                onGround.add(i)

        face = []
        for i in range(len(self.faces)):
            if self.faces[i].issubset(onGround):
                face.append(i)

        if len(face) == 0:
            print("no face detected")
            return -2
        elif len(face) == 1:
            return face[0]
        else:
            print("multiple faces detected")
            return -3

    def droptest(self, n):
        stat = [0] * len(self.faces)

        while sum(stat) < n:
            #print(sum(stat))

            height = 100
            dropHeight = np.array([0, 0, height], dtype=float)

            #create copy of the dice
            tmp = deepcopy(self)

            #rotation + raise to dropheight
            c = tmp.getCenter()
            tmp.translate(-1 * c)
            tmp.randRotate()
            tmp.translate(c)
            tmp.translate(dropHeight)

            #droptest
            i = 0
            while (not tmp.isStopped()) and i < 10000:
                tmp.update(TIME, GRAVITY)
                i += 1

            #get face
            index = tmp.faceOnGround()
            if index >= 0:
                stat[index] += 1

        return stat

    def saveDroptestGraph(self, n, name):
        height = 100
        dropHeight = np.array([0, 0, height], dtype=float)

        tmp = deepcopy(self)
        x=[]

        tmp.translate(dropHeight)

        stop = 0
        i = 0
        while i < n:
            tmp.update(TIME, GRAVITY)
            x.append(tmp.getTotalVelocityLength())
            if tmp.isStopped() and stop == 0:
                stop = i
            i += 1

        plt.plot(x)
        plt.axvline(stop, color="r")
        plt.xlabel("time")
        plt.ylabel("total velocity length")

        plt.savefig("graphs/{}.png".format(name))
        plt.close()

    def getNormalised(self):
        tmp = deepcopy(self)

        c = tmp.getCenter()
        max = np.linalg.norm(tmp.particles[0].position - c)
        for i in range(1, tmp.n):
            l = np.linalg.norm(tmp.particles[i].position - c)
            if l > max:
                max = l
        tmp.scale(1/max)

        return tmp

    def saveObj(self, name):
        filepath= "objects/{}.obj".format(name)

        with open(filepath, 'w') as f:
            f.write("# OBJ file\n")
            for particle in self.particles:
                f.write("v {} {} {}\n".format(particle.position[0], particle.position[1], particle.position[2]))
            for i in range(len(self.faces)):
                f.write("f")
                for p in self.faces[i]:
                    f.write(" {}".format(p + 1))
                f.write("\n")

        print("Obj file saved to '{}'.".format(filepath))

    def getPlaneOfFace(self, face):
        f = list(face)
        A = np.matrix([self.particles[f[0]].position,
                       self.particles[f[1]].position,
                       self.particles[f[2]].position])
        
        p = np.array([0, 0, 0])
        one = np.array([[1], [1], [1], [1]])

        A = np.vstack([p, A])
        A = np.append(A, one, axis=1)

        return A

    def isConvex(self):
        for face in self.faces:
            A = self.getPlaneOfFace(face)
            sign = set()

            for i in range(self.n):
                if i not in face:
                    A[0, :3] = self.particles[i].position
                    sign.add(np.linalg.det(A) < 0)
            
            if len(sign)!=1:
                return False
            
        return True

    #modification by moving a random point
    def getRandomModified(self):
        tmp = deepcopy(self)

        index = randrange(tmp.n)
        v = np.random.rand(3,)
        v *= 2
        v -= np.array([1, 1, 1])
        
        tmp.particles[index].position += v

        tmp.initSprings()

        return tmp

    def estimateBodyRandom(self, expected, sig, n):
        e = [0] * len(expected)
        for i in range(len(e)):
            e[i] = expected[i] * n

        stat0 = self.droptest(n)
        [ p0 ] = chisquare(stat0, f_exp=e)[1:]
        print(p0)

        while p0 < sig:
            tmp = self.getRandomModified()
            stat1 = tmp.droptest(n)
            [ p1 ] = chisquare(stat1, f_exp=e)[1:]

            if p0 < p1:
                self = tmp
                p0 = p1
                stat0 = stat1
            print(p0)


    #modification by changing the size of the faces
    def getFaceModified(self, expected, stat):
        e = [0] * len(expected)
        for i in range(len(e)):
            e[i] = expected[i] * sum(stat)

        tmp = deepcopy(self)

        move = np.zeros((tmp.n, 3), dtype=float)

        #for every face
        for i in range(len(tmp.faces)):

            #center of the face
            #trinagular!
            c = np.array([0, 0, 0], dtype=float)
            for p in tmp.faces[i]:
                c += tmp.particles[p].position
            c /= len(tmp.faces[i])

            #for every particle on a face
            for p in tmp.faces[i]:
                v = tmp.particles[p].position - c #from center to particle (out)
                v /= np.linalg.norm(v)
                #alpha = expected[i] - stat[i]
                #v *= alpha

                if expected[i] > stat[i]:
                    move[p,:] += v
                else:
                    move[p,:] -= v

        #print(move)

        for i in range(tmp.n):
            tmp.particles[i].position += move[i,:]

        tmp.initSprings()

        #tmp = tmp.getNormalised()
        #tmp.scale(tmp.maxd)

        return tmp

    def estimateBodyFace(self, expected, sig, n):
        print("n = {}".format(n))

        e = [0] * len(expected)
        for i in range(len(e)):
            e[i] = expected[i] * n
        
        tmp = deepcopy(self)

        stat = tmp.droptest(n)
        [ p ] = chisquare(stat, f_exp=e)[1:]
        print("{}, p = {}".format(stat, p))

        while p < sig:
            tmp = tmp.getFaceModified(e, stat)
            stat = tmp.droptest(n)
            [ p ] = chisquare(stat, f_exp=e)[1:][0]
            print("{}, p = {}".format(stat, p))

        if n < 1000:
            tmp = tmp.estimateBodyFace(expected, 0.8, n + 100)
        
        return tmp


    #modification by changing the size of the faces
    #based on the differece of the expected and the measured values
    def getFaceModified2(self, expected, stat):
        e = [0] * len(expected)
        for i in range(len(e)):
            e[i] = expected[i] * sum(stat)

        tmp = deepcopy(self)

        move = np.zeros((tmp.n, 3), dtype=float)

        #for every face
        for i in range(len(tmp.faces)):

            #center of the face
            c = np.array([0, 0, 0], dtype=float)
            for p in tmp.faces[i]:
                c += tmp.particles[p].position
            c /= len(tmp.faces[i])

            #for every particle on a face
            for p in tmp.faces[i]:
                v = tmp.particles[p].position - c #from center to particle (out)
                #v /= np.linalg.norm(v)
                diff = e[i]/sum(e) - stat[i]/sum(stat)
                v *= diff 

                move[p,:] += v

        for i in range(tmp.n):
            tmp.particles[i].position += move[i,:]

        tmp.initSprings()

        return tmp

    def estimateBodyFace2(self, expected, sig, n):
        print("n = {}".format(n))
        e = [0] * len(expected)
        for i in range(len(e)):
            e[i] = expected[i] * n
        
        tmp = deepcopy(self)

        stat = tmp.droptest(n)
        [ p ] = chisquare(stat, f_exp=e)[1:]
        print("{}, p = {}".format(stat, p))

        while p < sig:
            tmp = tmp.getFaceModified(e, stat)
            stat = tmp.droptest(n)
            [ p ] = chisquare(stat, f_exp=e)[1:]
            print("{}, p = {}".format(stat, p))

        if n < 1000:
            tmp = tmp.estimateBodyFace2(expected, 0.8, n + 100)
        
        return tmp

#---------------------------------

class Tetrahedron(Dice):
    def __init__(self):
        self.initVerticies()
        super().__init__()

    def initVerticies(self):
        a = 50
        p = [[   0,                         0,                                         0],
             [ a/2,     a*math.sin(math.pi/3),                                         0],
             [-a/2,     a*math.sin(math.pi/3),                                         0],
             [   0, a/(2*math.sin(math.pi/3)), a*math.sin(math.pi/4)/math.sin(math.pi/3)]]
        
        self.particles = []
        for value in p:
            self.particles.append(Particle(value))

        self.n = len(self.particles)

        self.faces = [{0, 1, 2},
                      {0, 1, 3},
                      {0, 2, 3},
                      {1, 2, 3}]

class Cube(Dice):
    def __init__(self):
        self.initVerticies()
        super().__init__()

    def initVerticies(self):
        a = 50
        p = [[0, 0, 0],
             [0, a, 0],
             [a, 0, 0],
             [a, a, 0],
             [0, 0, a],
             [0, a, a],
             [a, 0, a],
             [a, a, a]]

        self.particles = []
        for value in p:
            self.particles.append(Particle(value))

        self.n = len(self.particles)

        self.faces = [{0, 1, 2, 3},
                      {0, 1, 4, 5},
                      {0, 2, 4, 6},
                      {1, 3, 5, 7},
                      {2, 3, 6, 7},
                      {4, 5, 6, 7}]

class DoubleTetrahedron(Dice):
    def __init__(self):
        self.initVerticies()
        super().__init__()

    def initVerticies(self):
        a = 50
        b = a*math.sin(math.pi/4)/math.sin(math.pi/3)
        p = [[   0, a/(2*math.sin(math.pi/3)),   0],
             [   0,                         0,   b],
             [ a/2,     a*math.sin(math.pi/3),   b],
             [-a/2,     a*math.sin(math.pi/3),   b],
             [   0, a/(2*math.sin(math.pi/3)), 2*b]]
        
        self.particles = []
        for value in p:
            self.particles.append(Particle(value))

        self.n = len(self.particles)

        self.faces = [{0, 1, 2},
                      {0, 1, 3},
                      {0, 2, 3},
                      {1, 2, 4},
                      {1, 3, 4},
                      {2, 3, 4}]

class Octahedron(Dice):
    def __init__(self):
        self.initVerticies()
        super().__init__()

    def initVerticies(self):
        a = 50
        p = [[ 0,  0,   0],
             [ 0,  a,   a],
             [ 0, -a,   a],
             [ a,  0,   a],
             [-a,  0,   a],
             [ 0,  0, 2*a]]

        self.particles = []
        for value in p:
            self.particles.append(Particle(value))

        self.n = len(self.particles)

        self.faces = [{0, 1, 3},
                      {0, 1, 4},
                      {0, 2, 3},
                      {0, 2, 4},
                      {1, 3, 5},
                      {1, 4, 5},
                      {2, 3, 5},
                      {2, 4, 5}]
