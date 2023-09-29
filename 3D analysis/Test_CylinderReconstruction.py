#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:41:24 2023

@author: leo
"""

import numpy as np
import open3d as o3d
import copy
import glob
from Filtering3D import *
from pyransac.base import Model
from pyransac import ransac
import pyransac

#Data acquisition ============================================================


pcd0 = o3d.io.read_point_cloud('data_23-05-04_14-56-00/pc0.ply')
pcd1=pcd0
pcd1 = filterDATA(pcd0)
pcd2=pcd1
pcd1=pcd1.voxel_down_sample(voxel_size=0.05)
normal_param=o3d.geometry.KDTreeSearchParamRadius(0.05)
pcd1.estimate_normals()


center = pcd1.get_center()
points = np.asarray(pcd1.points)
normals=np.asarray(pcd1.normals)
PN=np.array([points,normals])
PN=np.swapaxes(PN,0,1)
PNL=np.ndarray.tolist(PN)


o3d.visualization.draw_geometries([pcd0])


#here you can choose between fixed (RCylModel()) and variable (CylModel()) radius for the cylinder research
Mymodel=RCylModel()
Bestmodel=RCylModel()

###Actual ransac =============================================================

#Ransac parameters
params=ransac.RansacParams(samples=3, iterations=1000, confidence=0.99999, threshold=0.003)
#Ransac application

Inliers,Bestmodel,ratio,inliers=pyransac.find_inliers(PNL, Mymodel, params)
#Inlier reformatting
Inliers=np.array(Inliers)
Inliers=Inliers[:,0,:]
Inliers=np.ndarray.tolist(Inliers)
pcdInliers=o3d.geometry.PointCloud()
pcdInliers.points=o3d.utility.Vector3dVector(Inliers)
pcdInliers.paint_uniform_color([1,0,0])

Cylinder=CylinderDipslay(Bestmodel,pcdInliers)

R=Bestmodel.radius
#Display
o3d.visualization.draw_geometries([Cylinder,pcdInliers,pcd1])
o3d.visualization.draw_geometries([pcd1,pcdInliers])
