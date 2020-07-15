#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:23:44 2020

@author: filipe
"""

from scipy import linalg
import numpy as np

class Camera(object):
    
    def __init__(self, P):
        self.P = P
        self.K = None # calibration matrix
        self.R = None # rotation
        self.t = None # translation 
        self.c = None # camer center
        
    def project(self, X):
        x = np.dot(self.P, X)
        for i in range(3):
            x[i] /= x[2]
            
        return x
    
    def rotation_matrix(a):
        '''Creates a 3D rotation matrix for rotation
        around the axis of the vector a.'''
        # https://en.wikipedia.org/wiki/Rotation_matrix
        # (Rotation matrix from axis and angle)
        # Exponential map
        R = np.eye(4)
        R[:3,:3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
        return R

    def factor(self):
        ''' to be used when P is given '''
        K, R = linalg.rq(self.P[:,:3])
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T)< 0:
            T[1,1] *= -1
        self.K = np.dot(K,T)
        self.R = np.dot(T,R)
        self.t = np.dot(linalg.inv(self.K), self.P[:,3])

        return self.K, self.R, self.t

    def center(self):
        if self.c is not None:
            return self.c
        else:
            self.factor()
            self.c = -np.dot(self.R.T, self.c)
            return self.c
