#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:12:40 2023

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

#Variable setup
Circle_Points=[Start_point]
Circle_tang=[Start_tang]
Input_Points=P

radius=0.2
Range_extension=1
#First section computation ====================================================
End_point,End_tang,End_normal,PointInliers,PointOutliers,Phi,Theta,r,EndInliers,Succes=PCCRegresion(Input_Points, Length, Start_point, Start_tang,Start_normal,radius,Range_extension)
End_point=End_point[0,:]
Circle_Points.append(End_point)
Circle_tang.append(End_tang)



#Display
linepoints=[Start_point,2*Start_tang,Start_normal,Circle_Points[1],Circle_Points[1]+Circle_tang[1]]

line=[[0,1],[0,2],[3,4]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)

pcd0.paint_uniform_color([0.3,0.3,0.3])
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(P)
pcd.paint_uniform_color([1,0,1])
pcd1=o3d.geometry.PointCloud()
pcd1.points=o3d.utility.Vector3dVector(PointInliers)
pcd1.paint_uniform_color([1,0,0])
pcd2=o3d.geometry.PointCloud()
pcd2.points=o3d.utility.Vector3dVector(Input_Points)
pcd2.paint_uniform_color([0,1,0])
# pcd3=o3d.geometry.PointCloud()
# pcd3.points=o3d.utility.Vector3dVector(Input_Point2)
# pcd3.paint_uniform_color([0,0,1])


o3d.visualization.draw_geometries([pcd2,line_set,pcd0,pcd1])


