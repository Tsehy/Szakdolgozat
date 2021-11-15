#%%
import math
import numpy as np
from random import random

def rotate(point, xAlpha, yAlpha, zAlpha):
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
    
    v = np.append(point, 1)[np.newaxis]
    return R.dot(v.T).T[0][:3]

def randrotate(point):
    #particles[0] is the pivot
    #we assume the origo is the center of the object
    #this means only apply this after translate(-c) !

    #random point on the sphere surface
    #hivatkozás!
    while True:
        r_point = np.array([np.random.normal() for _ in range(3)]) 
        if np.linalg.norm(r_point) != 0:
            break
    radius = np.linalg.norm(point)
    r_point *= radius / np.linalg.norm(r_point)

    #rotate pivot onto the x axis (radius, 0, 0)
    temp_r = math.sqrt(point[0]**2 + point[1]**2)
    fi_z = np.arccos(point[0] / math.sqrt(point[0]**2 + point[1]**2))
    if point[1] > 0:
        fi_z *= -1
    point = rotate(point, 0, 0, fi_z)

    temp_r = math.sqrt(point[0]**2 + point[2]**2)
    fi_y = np.arccos(point[0] / math.sqrt(point[0]**2 + point[2]**2))
    if point[2] > 0:
        fi_y *= -1
    point = rotate(point, 0, -fi_y, 0)

    #random rotation arount the x axis
    fi_x = random() * 2 * math.pi
    point = rotate(point, fi_x, 0, 0)

    #rotate pivot to the random point
    #1st rotate around the y axis until pivot.z = r_point.z
    if point[2] != r_point[2]:
        a = np.array([point[0], point[2]])
        rx = math.sqrt(radius**2 - r_point[2]**2)
        b = np.array([rx, r_point[2]])
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        fi_y = np.arccos(np.dot(a, b))
        if r_point[2] > 0:  
            fi_y *= -1
        print(fi_y)
        point = rotate(point, 0, fi_y, 0)

    #2nd rotate around the z axis
    if point[1] != r_point[1]:
        a = np.array([point[0], point[1]])
        b = np.array([r_point[0], r_point[1]])
        a /= np. linalg.norm(a)
        b /= np. linalg.norm(b)
        fi_z = np.arccos(np.dot(a, b))
        if r_point[1] < 0:  
            fi_z *= -1
        point = rotate(point, 0, 0, fi_z)
        
    return point

p = np.array([1, 0, 0])
p = randrotate(p)
#p = rotate(p, 0, 0, math.pi/2)
#p = rotate(p, math.pi/2, 0, 0)
with np.printoptions(precision=3, suppress=True):
    print(p)

# %%
