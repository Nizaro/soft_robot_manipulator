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

pcd0 = o3d.io.read_point_cloud('PCC_Generated2/2_segments/0-Camera-0.0.ply')
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

N=1 #number of segment
L=2 #Length of one segment

P,Cylinder,pcdVox,pcdInliers=Voxelized_Cylinder(points,pcd1,PNL,0.22,0.05)

params=ransac.RansacParams(1, iterations=100, confidence=0.9999, threshold=0.2)
PCCmodel=PCCregression()
PCCBestmodel=PCCregression()
PCCmodel.Start_point=np.array([0,0,0])
PCCmodel.Start_tang=np.array([0,0,1])
PCCmodel.Start_normal=np.array([1,0,0])
PCCmodel.Length=L

Endpoints=[]
NonValid=[]
PP=[]
for  i in range(len(P)):
    newpoint,Valid=PCCinversion(PCCmodel.Start_point, PCCmodel.Start_tang, PCCmodel.Start_normal, PCCmodel.Length, np.array([P[i]]))
    if Valid==True:
        Endpoints.append(newpoint)
        PP.append(P[i])
    else :
        NonValid.append(P[i])

Inliers,PCCBestmodel,ratio=pyransac.find_inliers(P, PCCmodel, params)


EndInliers=[]
for i in range(len(Inliers)):
    newpoint,Valid=PCCinversion(PCCmodel.Start_point, PCCmodel.Start_tang, PCCmodel.Start_normal, PCCmodel.Length, np.array([Inliers[i]]))
    EndInliers.append(newpoint)
Endpoint=np.mean(np.array([EndInliers]),axis=1)  

Dist=Endpoint-PCCmodel.Start_point
Phi_vector=np.cross(PCCmodel.Start_tang,np.cross(Dist/np.linalg.norm(Dist),PCCmodel.Start_tang)[0,:])
Phi_vector=Phi_vector/np.linalg.norm(Phi_vector)
Theta=np.arccos(np.dot(Dist,PCCmodel.Start_tang)/(np.linalg.norm(Dist)))
rho=0.5*PCCmodel.Length/Theta
rho_Vector=Phi_vector*rho
End_Tang=np.cos(2*Theta)*PCCmodel.Start_tang+np.sin(2*Theta)*Phi_vector
End_Tang=End_Tang/np.linalg.norm(End_Tang)



linepoints=[PCCmodel.Start_point,2*PCCmodel.Start_tang,PCCmodel.Start_normal,Endpoint[0,:],Endpoint[0,:]+End_Tang]

line=[[0,1],[0,2],[3,4]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)

pcd0.paint_uniform_color([0.3,0.3,0.3])
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(PP)
pcd.paint_uniform_color([1,0,1])
pcdEndpoint=o3d.geometry.PointCloud()
pcdEndpoint.points=o3d.utility.Vector3dVector([Endpoint[0,:]])
pcdEndpoint.paint_uniform_color([0,1,0])
pcdNV=o3d.geometry.PointCloud()
pcdNV.points=o3d.utility.Vector3dVector(NonValid)
pcdNV.paint_uniform_color([1,0,0])
pcdEndpoints=o3d.geometry.PointCloud()
pcdEndpoints.points=o3d.utility.Vector3dVector(Endpoints)
pcdEndpoints.paint_uniform_color([0,0,1])
o3d.visualization.draw_geometries([pcd,pcdNV,pcdEndpoints,pcdEndpoint,pcd0,line_set])


