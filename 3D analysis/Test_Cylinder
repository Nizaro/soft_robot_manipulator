#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:37:13 2023

@author: leo
"""

import numpy as np
import open3d as o3d
import glob
import copy
from Filtering3D import *

DIR=''
img=glob.glob(DIR +'*.ply')
img_name=copy.deepcopy(img)
for i in range(len(img)):
    img_name[i]=img_name[i].replace(DIR,'')
    img_name[i]=img_name[i].replace('.ply','')

File=0
File_name=img_name[File]
'''
'''
pcd0 = o3d.io.read_point_cloud('Record/pc20.ply')
#pcdGT= o3d.io.read_point_cloud('PCC_Generated2/4_segments/8-Ground_Truth.ply')
pcd1=pcd0
Start=np.array([0.02,-0.24,-0.430])
pcd1 = filterDATA(pcd0)
pcd2=pcd1
pcd1=pcd1.voxel_down_sample(voxel_size=0.001)
#pcdGT=pcdGT.voxel_down_sample(voxel_size=0.02)
normal_param=o3d.geometry.KDTreeSearchParamRadius(0.05)
pcd1.estimate_normals()


#Starts=np.load(DIR+'Start_point.npy')


center = pcd1.get_center()
points = np.asarray(pcd1.points)
normals=np.asarray(pcd1.normals)
PN=np.array([points,normals])
PN=np.swapaxes(PN,0,1)
PNL=np.ndarray.tolist(PN)

o3d.visualization.draw_geometries([pcd1])


#here you can choose between fixed (RCylModel()) and variable (CylModel()) radius for the cylinder research
Mymodel=RCylModel()
Bestmodel=RCylModel()


###space partition ===========================================================

N=2 #number of segment

radius=0
L=radius*17/2 #Length of one segment

P,Cylinder,pcdVox,pcdInliers=Voxelized_Cylinder(points,pcd1,PNL,radius,radius/5)

pcdcenter=o3d.geometry.PointCloud()    
pcdcenter.points=o3d.utility.Vector3dVector(P)
pcdcenter.paint_uniform_color([1,0,0])
o3d.visualization.draw_geometries([pcdcenter,pcd1])



pcdInlierscenter=o3d.geometry.PointCloud()
pcdInlierscenter.points=o3d.utility.Vector3dVector(CurvInlier)
pcdInlierscenter.paint_uniform_color([0,0,1])



pcdSurf,line_set,Curve,Curvedot=Generate_ModelSurf(Curv,radius)

#Result_angle,Result_length,Result_mean_Inl_dist,Result_mean_all_dist,Result_med_Inl_dist,Result_med_all_dist,Result_1q_Inl_dist,Result_1q_all_dist,Result_3q_Inl_dist,Result_3q_all_dist=Evaluate_model(pcdSurf,Curve,Curvedot, pcd2, pcdInliers)

pcdcenter=o3d.geometry.PointCloud()    
pcdcenter.points=o3d.utility.Vector3dVector(P)
pcdcenter.paint_uniform_color([1,0,0])


vis1 = o3d.visualization.VisualizerWithEditing()
vis2 = o3d.visualization.VisualizerWithEditing()
vis3 = o3d.visualization.VisualizerWithEditing()
vis4 = o3d.visualization.VisualizerWithEditing()
vis5 = o3d.visualization.VisualizerWithEditing()
vis6 = o3d.visualization.VisualizerWithEditing()
vis7 = o3d.visualization.VisualizerWithEditing() 
vis1.create_window(window_name='Input', width=480, height=300, left=0, top=0)
vis2.create_window(window_name='Partition', width=480, height=300, left=480, top=0)
vis3.create_window(window_name='RANSAC Cylinder search', width=480, height=300, left=960, top=0)
vis4.create_window(window_name='Cylinder centers', width=480, height=300, left=0, top=400)
vis5.create_window(window_name='RANSAC center line', width=480, height=300, left=480, top=400)
vis6.create_window(window_name='Generated surface', width=480, height=300, left=960, top=400)
vis1.add_geometry(pcd1)
vis2.add_geometry(pcdVox)
vis3.add_geometry(Cylinder)
vis4.add_geometry(pcdcenter)
vis5.add_geometry(line_set)
vis6.add_geometry(pcdSurf)




#o3d.visualization.draw_geometries([pcdSurf,pcdSurf],window_name='Generated surface')
o3d.visualization.draw_geometries([pcdcenter,pcd1],window_name='output')

