#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 18:37:25 2023

@author: leo
"""

import numpy as np
import open3d as o3d
from pyransac.base import Model
from pyransac import ransac
import pyransac
from Filtering3D import *
 # Data Acquisition ===========================================================
 
pcd0 = o3d.io.read_point_cloud('PCC_Generated2/2_segments/13-Camera-0.0.ply')
pcdGT= o3d.io.read_point_cloud('PCC_Generated2/2_segments/5-Ground_Truth.ply')
pcd1=pcd0
pcd2=pcd1
pcd1=pcd1.voxel_down_sample(voxel_size=0.05)
pcdGT=pcdGT.voxel_down_sample(voxel_size=0.02)
normal_param=o3d.geometry.KDTreeSearchParamRadius(0.15)
pcd1.estimate_normals()


center = pcd1.get_center()
points = np.asarray(pcd1.points)
normals=np.asarray(pcd1.normals)
PN=np.array([points,normals])
PN=np.swapaxes(PN,0,1)
PNL=np.ndarray.tolist(PN)
#here you can choose between fixed (RCylModel()) and variable (CylModel()) radius for the cylinder research
Mymodel=RCylModel()
Bestmodel=RCylModel()

#Center points computation ====================================================
N=2 #number of segment
L=2 #Length of one segment

P,Cylinder,pcdVox,pcdInliers=Voxelized_Cylinder(points,pcd1,PNL,0.22,0.05)

#Input parameters =============================================================
Start_point=np.array([0,0,0])
Start_tang=np.array([0,0,1])
Start_normal=np.array([1,0,0])
Length=L

# #Variable setup
# Circle_Points=[Start_point]
# Circle_tang=[Start_tang]
# Input_Points=P

# End_point=Start_point
# End_tang=Start_tang

# for i in range(N):
#     End_point,End_tang,PointInliers,Input_Points=PCCRegresion(Input_Points, Length, End_point,End_tang)
#     End_point=End_point[0,:]
#     Circle_Points.append(End_point)
#     Circle_tang.append(End_tang)
Input_Points=P
Circle_Points, Circle_tang=MultiPCCRegression(Input_Points, Length, Start_point, Start_tang, N)

#Display
linepoints=[Start_point,2*Start_tang,Start_normal,Circle_Points[1],Circle_Points[1]+Circle_tang[1],Circle_Points[2],Circle_Points[2]+Circle_tang[2]]

line=[[0,1],[0,2],[3,4],[5,6]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)

pcd0.paint_uniform_color([0.3,0.3,0.3])
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(P)
pcd.paint_uniform_color([1,0,1])
pcd1=o3d.geometry.PointCloud()
# pcd1.points=o3d.utility.Vector3dVector(PointInliers)
# pcd1.paint_uniform_color([1,0,0])
# pcd2=o3d.geometry.PointCloud()
# pcd2.points=o3d.utility.Vector3dVector(Input_Points)
# pcd2.paint_uniform_color([0,1,0])



o3d.visualization.draw_geometries([line_set,pcd0,pcd])

