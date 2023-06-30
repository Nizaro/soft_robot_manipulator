#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:57:53 2023

@author: leo
"""

import open3d as o3d
import glob
import copy


DIR='PCC_Generated2/3_segments/'
img=glob.glob(DIR +'*0.0.ply')
img2=glob.glob(DIR +'*Ground_Truth.ply')
img_name=copy.deepcopy(img)
for i in range(len(img)):
    img_name[i]=img_name[i].replace(DIR,'')
    img_name[i]=img_name[i].replace('.ply','')
    
    
    
    pcd0 = o3d.io.read_point_cloud(img[i])
    pcd1 = o3d.io.read_point_cloud(img2[i])
    
    pcd2=pcd1.voxel_down_sample(voxel_size=0.03)
    pcd2.paint_uniform_color([1,0,1])
    
    o3d.visualization.draw_geometries([pcd0,pcd2],window_name=img_name[i])
