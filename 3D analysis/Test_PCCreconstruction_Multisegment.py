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
from Arm_Gen import *
import timeit
 # Data Acquisition ===========================================================
start=timeit.default_timer()
pcd0 = o3d.io.read_point_cloud('Record/pc2.ply')
pcd0 = filterDATA(pcd0)
pcd1=pcd0
pcd2=pcd1
pcd1=pcd1.voxel_down_sample(voxel_size=0.005)
stop1=timeit.default_timer()
print("Acquisition         :",stop1-start,"s")
pcd1.estimate_normals()
stop2=timeit.default_timer()
print("Normal computation  :",stop2-stop1,"s")

o3d.visualization.draw_geometries([pcd1])
start=timeit.default_timer()
center = pcd1.get_center()
points = np.asarray(pcd1.points)
normals=np.asarray(pcd1.normals)
PN=np.array([points,normals])
PN=np.swapaxes(PN,0,1)
PNL=np.ndarray.tolist(PN)
#here you can choose between fixed (RCylModel()) and variable (CylModel()) radius for the cylinder research


#Center points computation ====================================================
N=2 #number of segment
radius=0.02
L=radius*17/2 #Length of one segment

P,Cylinder,pcdVox,pcdInliers=Voxelized_Cylinder(points,pcd1,PNL,radius,radius/5)
stop1=timeit.default_timer()
print("Cylinder generation :",stop1-start,"s")
#Input parameters =============================================================
Start_point=np.array([0.015,-0.25,-0.42])
Start_tang=np.array([0,1,-0.28])
Start_tang=Start_tang/np.linalg.norm(Start_tang)
Start_normal=np.array([0,0.28,1])
Start_normal=Start_normal/np.linalg.norm(Start_normal)
linepoints=[Start_point,Start_point+Start_tang*L,Start_point+Start_normal*L]
line=[[0,1],[0,2]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)

pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(P)
pcd.paint_uniform_color([1,0,1])


#o3d.visualization.draw_geometries([line_set,pcd1,pcd])
Length=L

Input_Points=P

#PCC model computation
Circle_Points, Circle_tang,Circle_normal,Phi,Theta,r,Qn,Inliers=MultiPCCRegression(Input_Points, Length, Start_point, Start_tang, N,Start_normal,radius)

stop2=timeit.default_timer()
print("Arm reconstruction  :",stop2-stop1,"s")
#Surface generation
Pn=np.array(Circle_Points)
m=100
k=50
S=SmoothConstructPCC(Pn, Length, k, radius, Qn, m)
points=[]
for j in range(N):
    
    for l in range(m):
        
        for i in range(k):
            points.append(S[:,l,j,i])
stop3=timeit.default_timer()
print("Surface generation  :",stop3-stop2,"s")

#Display
linepoints=[Start_point,Start_point+Start_tang*L,Start_point+Start_normal*L,Circle_Points[1],Circle_Points[1]+Circle_tang[1]*L,Circle_Points[2],Circle_Points[2]+Circle_tang[2]*L,
            Circle_Points[1]+Circle_normal[1]*L,Circle_Points[2]+Circle_normal[2]*L]
#linepoints=[Start_point,2*Start_tang,Start_normal,Circle_Points[1],Circle_Points[1]+Circle_tang[1],Circle_Points[2],Circle_Points[2]+Circle_tang[2],Circle_Points[3],Circle_Points[3]+Circle_tang[3]]
#linepoints=[Start_point,2*Start_tang,Start_normal,Circle_Points[1],Circle_Points[1]+Circle_tang[1],
            # Circle_Points[2],Circle_Points[2]+Circle_tang[2],
            # Circle_Points[3],Circle_Points[3]+Circle_tang[3],
            # Circle_Points[1],Circle_Points[1]+Circle_normal[1],
            # Circle_Points[2],Circle_Points[2]+Circle_normal[2],
            # Circle_Points[3],Circle_Points[3]+Circle_normal[3],]

line=[[0,1],[0,2],[3,4],[5,6],[3,7],[5,8]]
#line=[[0,1],[0,2],[3,4],[5,6],[7,8]]
#line=[[0,1],[0,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)


pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(P)
pcd.paint_uniform_color([1,0,1])
pcd0.estimate_normals()
pcd1=o3d.geometry.PointCloud()
pcd1.points=o3d.utility.Vector3dVector(points)
pcd1.paint_uniform_color([0,0,0.7])
pcd1.estimate_normals()
pcd2=o3d.geometry.PointCloud()
pcd2.points=o3d.utility.Vector3dVector(Inliers)
pcd2.paint_uniform_color([0,1,1])

o3d.visualization.draw_geometries([line_set,pcd,pcd2,pcd0,pcd1])

