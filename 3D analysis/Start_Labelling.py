#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:08:20 2023

@author: leo
"""
import numpy as np
import open3d as o3d
import glob
import copy

point=[]
normal=[]


DIR="DAta_1_joint/20deg/"
img=glob.glob(DIR +'*.ply')
img_name=copy.deepcopy(img)
PN=np.empty([len(img),6])
for i in range(len(img)):
    img_name[i]=img_name[i].replace(DIR,'')
    img_name[i]=img_name[i].replace('.ply','')
    pcd0 = o3d.io.read_point_cloud(img[i])
    pcd0.estimate_normals()
    vis1 = o3d.visualization.VisualizerWithEditing()
    
    vis1.create_window(window_name=img_name[i], width=1400, height=700, left=0, top=0)
    vis1.add_geometry(pcd0)
    vis1.run()
    vis1.destroy_window()
    
    Ind=vis1.get_picked_points()
    points = np.asarray(pcd0.points)
    normals = np.asarray(pcd0.normals)
    point=points[Ind]
    normal=normals[Ind]
    PN[i,0:3]=point
    PN[i,3:6]=normal


np.save(DIR+'Start_point.npy',PN) 
PNN=np.load(DIR+'Start_point.npy')