#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 22:38:00 2020

@author: filipe
"""

''' Pose estimation from planes and markers using my own phone'''

import camera
import cv2
import homography
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

#%%
''' Load camera intrinsics from file '''
K, dist = np.load('cam-info.npy', allow_pickle = True)

#%%
''' Load images and apply SIFT '''
im1 = cv2.imread('imgs/test3.jpeg')
im2 = cv2.imread('imgs/test4.jpeg')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
l0, d0 = sift.detectAndCompute(im1_gray, None)
l1, d1 = sift.detectAndCompute(im2_gray, None)

#%%
''' Match points from img1 to points in img2 '''
bf = cv2.BFMatcher(crossCheck = True)
matches = bf.match(d0, d1)

#https://stackoverflow.com/questions/30716610/how-to-get-pixel-coordinates-from-feature-matching-in-opencv-python
kp0 = [l0[mat.queryIdx].pt for mat in matches] 
kp1 = [l1[mat.trainIdx].pt for mat in matches]
kp0 = np.array(kp0)
kp1 = np.array(kp1)

#%%
''' Compute from and to points '''
fp = homography.make_homog(kp0[:100,:2].T)
tp = homography.make_homog(kp1[:100,:2].T)

model = homography.RansacModel()
H, _ = homography.H_from_ransac(fp,tp,model)

#%%
''' Draw cone '''
def cube_points(c,wid):
    """ Creates a list of points for plotting
    a cube with plot. (the first 5 points are
    the bottom square, some sides repeated). """
    p = []
    #bottom
    p.append([c[0]-wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    p.append([c[0]-wid,c[1]-wid,c[2]-wid]) #same as first to close plot
    #top
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]-wid,c[2]+wid]) #same as first to close plot
    #vertical sides
    p.append([c[0]-wid,c[1]-wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]+wid])
    p.append([c[0]-wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]-wid])
    p.append([c[0]+wid,c[1]+wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]+wid])
    p.append([c[0]+wid,c[1]-wid,c[2]-wid])
    return np.array(p).T

#%%
box = cube_points([0,0,0.1], 0.2)

''' Camera 1 at position 0 '''
P1 = np.hstack((K, np.dot(K, np.array([[0],[0],[-1]]))))
cam1 = camera.Camera(P1)

''' Project cube (first points are the bottom square) '''
box_P1 = homography.make_homog(box[:,:5])
box_P1 = cam1.project(box_P1)

''' Transfer points from P1 to P2 using homography '''
box_trans = np.dot(H, box_P1)
box_trans = homography.normalize(box_trans)

''' Compute P2 '''
P2 = np.dot(H, cam1.P)
cam2 = camera.Camera(P2)
A = np.dot(linalg.inv(K), cam2.P[:,:3])
A = np.array([A[:,0],A[:,1], np.cross(A[:,0],A[:,1])]).T
cam2.P[:,:3] = np.dot(K,A)

''' Project with the second camera '''
box_P2 = homography.make_homog(box)
box_P2 = cam2.project(box_P2)

#%%
''' Plot projections '''
# 2D projection of bottom square
plt.figure(1)
plt.imshow(im1)
plt.plot(box_P1[0,:], box_P1[1,:], linewidth=3)

# 2D projection transferred with H
plt.figure(2)
plt.imshow(im2)
plt.plot(box_trans[0,:], box_trans[1,:], linewidth=3)

# 3D cube
plt.figure(3)
plt.imshow(im2)
plt.plot(box_P2[0,:], box_P2[1,:], linewidth=3)

#%%
''' DEBUG '''
#plt.figure()
#img3 = cv2.drawMatches(im1, l0, im2, l1, matches[0:100], None, flags=2)
#plt.imshow(img3)
