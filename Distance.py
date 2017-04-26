#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
from numpy import linalg as la
import scipy.spatial.distance as dist


class Distance(object):
    def __init__(self):
        ''

    @classmethod
    def ma_distance(cls, x, y, covMat):
        if (x==y).all():
            return 0
        #covMat = np.matrix(np.cov(x, y))
        if np.isnan(covMat).any():
            return 0
        distance = dist.mahalanobis(x, y, covMat)
        return distance

    @classmethod
    def person_sim(cls, x, y):
        return 0.5 + 0.5 * np.corrcoef(x, y, rowvar=0)[0][1]

    @classmethod
    def cos_distance(cls, x, y ):
        x, y = np.mat(x), np.mat(y)
        #num = float(x.dot(y.T))
        num = float(x * y.T)
        denom = la.norm(x) * la.norm(y)
        #dist = 0.5 + 0.5 * (num / denom)
        dist = 1.0 if denom == 0 else num / denom 
        return 3 * (1 - dist)

    @classmethod
    def ecld_distance(cls, x, y, std, weight):
        return math.sqrt(sum([(((v1 - v2)/(v3+0.000001))**2)*v4 for v1,v2,v3,v4 in zip(x,y,std,weight)])) / len(x)
 
if __name__ == '__main__':
    print Distance.cos_distance([878.7178982088466, 0.3791653059858058, 0.8194141106127972, 0.962201125180818, 285], [171.51709517771863, 8.677810644349934, 0.9737752361596916, 0.7040231423300713, 254])
    print Distance.cos_distance([1,2,3,4,5],[1,200,3,4,100])
    print Distance.cos_distance([0,0,0], [0,0,0])

