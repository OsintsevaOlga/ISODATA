#!/usr/bin/python3

import sys
import cv2
import numpy as np
import math
import time
import random

pict = cv2.imread('learn1.png')
krest = cv2.imread('red.png')
result = cv2.matchTemplate(pict, krest, cv2.TM_CCORR_NORMED)

threshold = 0.5916
loc = np.where(result >= threshold)

X = []
Y = []
for pt in zip(*loc[::-1]):
    X.append(pt[0])
    Y.append(pt[1])
    
K = 4
Qn = 3
Qs = 80
Qc = 20
L = 4
Iter = 20
num_cluster = 2
z_x = []
z_y = []
sigma_X = []
sigma_Y = []
sigma_Max = []
dist1 = []
dist2 = []

def First_Distribution():
    for i in range(0, num_cluster):
        z_x.append(X[i])
        z_y.append(Y[i])
    for i in range (0, len(z_x)):
        cv2.circle(pict, (z_x[i], z_y[i]), 10, (255, 0, 0), -1)

cluster_X = []
cluster_Y = []


for i in range(0, num_cluster):
    cluster_X.append([])
    cluster_Y.append([])

r = []
MD = 0

for i in range(0, num_cluster):
    r.append(0)

def Rand():
    global num_cluster
    global X
    global Y
    global z_x
    global z_y
    for i in range (0, num_cluster):
        tmp = random.randrange(0, num_cluster, 1)
        z_x.append(X[tmp])
        z_y.append(Y[tmp])
    for i in range (0, len(z_x)):
        cv2.circle(pict, (z_x[i], z_y[i]), 10, (255, 0, 0), -1)

def Min_Distance():
    global num_cluster
    global X
    global Y
    global z_x
    global z_y
    
    k = 0
    d = []
    d1 = []
    d2 = []
    for i in range(0, len(X)):
        for j in range(i + 1, len(X)):
            d1.append(i)
            d2.append(j)
            d.append(math.sqrt((X[i] - X[j]) * (X[i] - X[j]) + (Y[i] - Y[j]) * (Y[i] - Y[j])))

    for i in range(0, len(d)):
        for j in range(0, len(d) - 1):
            if (d[j] > d[j + 1]):
                k = d[j]
                d[j] = d[j + 1]
                d[j + 1] = k
                k = d1[j]
                d1[j] = d1[j + 1]
                d1[j + 1] = k
                k = d2[j]
                d2[j] = d2[j + 1]
                d2[j + 1] = k
    w = []
    for i in range(0, num_cluster):
        if ((d1[i] in w) == False):
                w.append(d1[i])
        if ((d2[i] in w) == False):
                w.append(d2[i])       
    for i in range (0, num_cluster):
        z_x.append(X[w[i]])
        z_y.append(Y[w[i]])
    for i in range (0, len(z_x)):
        cv2.circle(pict, (z_x[i], z_y[i]), 10, (255, 0, 0), -1)

def Max_Distance():
    global num_cluster
    global X
    global Y
    global z_x
    global z_y
    
    k = 0
    d = []
    d1 = []
    d2 = []
    for i in range(0, len(X)):
        for j in range(i + 1, len(X)):
            d1.append(i)
            d2.append(j)
            d.append(math.sqrt((X[i] - X[j]) * (X[i] - X[j]) + (Y[i] - Y[j]) * (Y[i] - Y[j])))

    for i in range(0, len(d)):
        for j in range(0, len(d) - 1):
            if (d[j] < d[j + 1]):
                k = d[j]
                d[j] = d[j + 1]
                d[j + 1] = k
                k = d1[j]
                d1[j] = d1[j + 1]
                d1[j + 1] = k
                k = d2[j]
                d2[j] = d2[j + 1]
                d2[j + 1] = k
    w = []
    for i in range(0, num_cluster):
        if ((d1[i] in w) == False):
                w.append(d1[i])
        if ((d2[i] in w) == False):
                w.append(d2[i])       
    for i in range (0, num_cluster):
        z_x.append(X[w[i]])
        z_y.append(Y[w[i]])
    for i in range (0, len(z_x)):
        cv2.circle(pict, (z_x[i], z_y[i]), 10, (255, 0, 0), -1)

def Image_Distribution():
    print ("dist")
    global X
    global Y
    global num_cluster
    global z_x
    global z_y
    global cluster_X
    global cluster_Y
    for i in range(0, len(X)):
        min_d = 666666
        num = -1
        for j in range(0, num_cluster):
            d = math.sqrt((X[i] - z_x[j]) * (X[i] - z_x[j]) + (Y[i] - z_y[j]) * (Y[i] - z_y[j]))
            if d < min_d:
                min_d = d
                num = j

        cluster_X[num].append(X[i])
        cluster_Y[num].append(Y[i])
    X.clear()
    Y.clear()
    for i in range (0, len(cluster_X)):
        print (cluster_X[i])

def Delete_cluster(j):
    print("Delete", j)
    global X
    global Y
    global num_cluster
    global cluster_X
    global cluster_Y
    global z_x
    global z_y
    for k in range(0, len(cluster_X[j])):
        X.append(cluster_X[j][k])
        Y.append(cluster_Y[j][k])
    cluster_X[j].clear()
    cluster_Y[j].clear()
    cluster_X.pop(j)
    cluster_Y.pop(j)
    z_x.pop(j)
    z_y.pop(j)
    r.pop(j)
    num_cluster -= 1

def Recount(i):
    print ("recount", i)
    global z_x
    global z_y
    global r
    min1 = 666666
    for j in range(0, len(cluster_X[i])):
        distant = 0
        for k in range(0, len(cluster_X[i])):
            distant = distant + math.sqrt((cluster_X[i][j] - cluster_X[i][k]) * (cluster_X[i][j] - cluster_X[i][k]) + (
                cluster_Y[i][j] - cluster_Y[i][k]) * (cluster_Y[i][j] - cluster_Y[i][k]))
        if (distant / len(cluster_X[i])) < min1:
            min1 = (distant / len(cluster_X[i]))
            z_x[i] = cluster_X[i][j]
            z_y[i] = cluster_Y[i][j]
    r[i] = min1

def Middle_Distance(num_cluster, cluster_X, r):
    global MD
    MD = 0
    for i in range(0, num_cluster):
        MD = MD + (r[i] * len(cluster_X[i]))
    MD = MD / num_cluster

def Standard_deviation(num_cluster, cluster_X, cluster_Y):
    global sigma_X
    global sigma_Y
    sigma_X = []
    sigma_Y = []
    for i in range(0, num_cluster):
        sum_X = 0
        sum_Y = 0
        for j in range(0, len(cluster_X[i])):
            sum_Y += (cluster_Y[i][j] - z_y[i]) * (cluster_Y[i][j] - z_y[i])
            sum_X += (cluster_X[i][j] - z_x[i]) * (cluster_X[i][j] - z_x[i])
        sum_X /= len(cluster_X[i])
        sum_X = math.sqrt(sum_X)
        sigma_X.append(sum_X)
        sum_Y /= len(cluster_Y[i])
        sum_Y = math.sqrt(sum_Y)
        sigma_Y.append(sum_Y)


def Sigma_Max(num_cluster):
    global sigma_Max
    sigma_Max = []
    for i in range(0, num_cluster):
        if sigma_X[i] > sigma_Y[i]:
            sigma_Max.append(sigma_X[i])
        else:
            sigma_Max.append(sigma_Y[i])


def Split_Cluster(i):
    print ("split", i)
    global z_x
    global z_y
    global X
    global Y
    global num_cluster
    global cluster_X
    global cluster_Y
    global sigma_Max
    gamma = 0.29611 * sigma_Max[i]
    z_x.append(int(z_x[i] + gamma))
    z_x.append(int(z_x[i] - gamma))
    z_y.append(int(z_y[i] + gamma))
    z_y.append(int(z_y[i] - gamma))
    r.append(0)
    r.append(0)
    num_cluster += 2

    Delete_cluster(i)
    last1 = num_cluster - 1
    last2 = num_cluster - 2
    cluster_X.append([])
    cluster_Y.append([])
    cluster_X.append([])
    cluster_Y.append([])
    for j in range(0, len(X)):
        if math.sqrt((X[j] - z_x[last1]) * (X[j] - z_x[last1]) + (Y[j] - z_y[last1]) * (Y[j] - z_y[last1])) < math.sqrt((X[j] - z_x[last2]) * (X[j] - z_x[last2]) + (Y[j] - z_y[last2]) * (Y[j] - z_y[last2])):
            cluster_X[last1].append(X[j])
            cluster_Y[last1].append(Y[j])
        else:
            cluster_X[last2].append(X[j])
            cluster_Y[last2].append(Y[j])
    X.clear()
    Y.clear()


def Distance():
    global num_cluster
    global dist1
    global dist2
    global L
    k = 0
    dist = []

    for i in range(0, num_cluster):
        for j in range(i + 1, num_cluster):
            dist1.append(i)
            dist2.append(j)
            dist.append(math.sqrt((z_x[i] - z_x[j]) * (z_x[i] - z_x[j]) + (z_y[i] - z_y[j]) * (z_y[i] - z_y[j])))

    for i in range(0, len(dist)):
        for j in range(0, len(dist) - 1):
            if (dist[j] > dist[j + 1]):
                k = dist[j]
                dist[j] = dist[j + 1]
                dist[j + 1] = k
                k = dist1[j]
                dist1[j] = dist1[j + 1]
                dist1[j + 1] = k
                k = dist2[j]
                dist2[j] = dist2[j + 1]
                dist2[j + 1] = k
    L = 0
    while (dist[L] < Qc and L < len(dist)):
        L += 1
   

def Merge(c1, c2):
    print ("merge", c1, c2)
    global z_x
    global z_y
    global num_cluster
    global cluster_X
    global cluster_Y
    tmp_x = (int((z_x[c1] * len(cluster_X[c1]) + z_x[c2] *
               len(cluster_X[c2])) / (len(cluster_X[c1]) + len(cluster_X[c2]))))
    tmp_y = (int((z_y[c1] * len(cluster_Y[c1]) + z_y[c2] *
               len(cluster_Y[c2])) / (len(cluster_Y[c1]) + len(cluster_Y[c2]))))

    t_x = []
    t_y = []
    for i in range(0, len(cluster_X[c1])):
        t_x.append(cluster_X[c1][i])
        t_y.append(cluster_Y[c1][i])
    for i in range(0, len(cluster_X[c2])):
        t_x.append(cluster_X[c2][i])
        t_y.append(cluster_Y[c2][i])
    r.append(0)
    num_cluster -= 1
    z_x.pop(c2)
    z_y.pop(c2)
    z_x.pop(c1)
    z_y.pop(c1)
    z_x.append(tmp_x)
    z_y.append(tmp_y)
    cluster_X[c2].clear
    cluster_X.pop(c2)
    cluster_Y[c2].clear
    cluster_Y.pop(c2)
    cluster_X[c1].clear
    cluster_X.pop(c1)
    cluster_Y[c1].clear
    cluster_Y.pop(c1)
    cluster_X.append(t_x)
    cluster_Y.append(t_y)
    t_x.clear
    t_y.clear


First_Distribution()
q = 0
while (q <= Iter):

    print("Iter", q)
    print ("num", num_cluster)
    for i in range (0, len(cluster_X)):
        print (cluster_X[i])
    contin = 0
    Image_Distribution()

    for i in range(0, num_cluster):
        if (len(cluster_X[i]) < Qn):
            Delete_cluster(i)
            Image_Distribution()
            break

    for k in range (0, num_cluster):
        Recount(k)
           
    Middle_Distance(num_cluster, cluster_X, r)

    if (q == Iter):
        Qc = 0
        Distance()
        Visited = []
        for i in range(0, num_cluster):
            Visited.append(True)
        tmp = num_cluster

        for i in range(0, L):
            if Visited[dist1[i]] and Visited[dist2[i]]:
                Visited[dist1[i]] = False
                Visited[dist2[i]] = False
                Merge(dist1[i], dist2[i])

        for i in range(tmp, 0):
            if Visited[i] == False:
                
                Delete_cluster(i)

        q += 1

    else:
        if (((q % 2) == 0) or (num_cluster >= 2 * K)):
            Distance()
            Visited = []
            for i in range(0, num_cluster):
                Visited.append(True)
            tmp = num_cluster

            for i in range(0, L):
                if Visited[dist1[i]] and Visited[dist2[i]]:
                   
                    Visited[dist1[i]] = False
                    Visited[dist2[i]] = False
                    Merge(dist1[i], dist2[i])

            for i in range(tmp, 0):
                if Visited[i] == False:
                    Delete_cluster(i)

            q += 1

        else:
            Standard_deviation(num_cluster, cluster_X, cluster_Y)
            Sigma_Max(num_cluster)
            for i in range(0, num_cluster):
                check = False
                if ((sigma_Max[i] > Qs) and ((r[i] < MD) and (len(cluster_X[i]) > 2 * (Qn + 1)) or (num_cluster <= K / 2))):
                    check = True
                    Split_Cluster(i)
                if (check == True):
                    q += 1
                    contin = 666
                    break
            if (contin == 666):
                continue
            Distance()
            Visited = []
            for i in range(0, num_cluster):
                Visited.append(True)
            tmp = num_cluster

            for i in range(0, L):
                if Visited[dist1[i]] and Visited[dist2[i]]:
                    Visited[dist1[i]] = False
                    Visited[dist2[i]] = False
                    Merge(dist1[i], dist2[i])

            for i in range(tmp, 0):
                if Visited[i] == False:
                    Delete_cluster(i)

            q +=1

for i in range (0, len(z_x)):
    cv2.circle(pict, (z_x[i], z_y[i]), 7, (0, 255, 255), -1)
for i in range (0, len(cluster_X[0])):
    cv2.circle(pict, (cluster_X[0][i], cluster_Y[0][i]), 5, (0, 0, 0), -1)
for i in range (0, len(cluster_X[1])):
    cv2.circle(pict, (cluster_X[1][i], cluster_Y[1][i]), 5, (0, 255, 0), -1)
for i in range (0, len(cluster_X[2])):
    cv2.circle(pict, (cluster_X[2][i], cluster_Y[2][i]), 5, (0, 0, 255), -1)

cv2.imwrite('res.png',pict)