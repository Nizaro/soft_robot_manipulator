#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:32:01 2023

@author: leo
"""

import numpy as np
import open3d as o3d
import glob
import timeit
import csv
import copy
from Filtering3D import *

DIR="PCC_Generated2/1_segments/"
N=20 #number of test per image
img=glob.glob(DIR +'*.ply')
img_name=copy.deepcopy(img)

lensurf=300*100

Result_angle=np.empty(len(img)*N)
Result_length=np.empty(len(img)*N)
Result_mean_Inl_dist=np.empty(len(img)*N)
Result_mean_all_dist=np.empty(len(img)*N)
Result_med_Inl_dist=np.empty(len(img)*N)
Result_med_all_dist=np.empty(len(img)*N)
Result_Max_Inl_Dist=np.empty(len(img)*N)
Result_Max_all_Dist=np.empty(len(img)*N)
Result_Time=np.empty(len(img)*N)
Result_dev_all_dist=np.empty(len(img)*N)
Result_dev_Inl_dist=np.empty(len(img)*N)
Analysis_dev_inl=np.empty(len(img))
Analysis_mean_inl=np.empty(len(img))



Test_img=[]

Radius=0.22
N_Segment=1
L_Segment=2

N_Test=5


print(len(img),' image to test')

for i in range(len(img)):
    img_name[i]=img_name[i].replace(DIR,'')
    img_name[i]=img_name[i].replace('.ply','')
    
    if i%N_Test==0 or i%N_Test==1 or i%N_Test==4:
        Noise=0.05
    else:
        if i%N_Test==2:
            Noise=0.1
        if i%N_Test==3:
            Noise=0.15
    if i%N_Test==0:
        pcdref=o3d.io.read_point_cloud(img[i+N_Test-1])
    
        
    if i%N_Test==0 or i%N_Test==4:
        for j in range(N):
            print('Treatment of image ',i+1,'/',len(img),'(',img_name[i],') test ',j+1,'/',N)
            start=timeit.default_timer()
            #acquisition
            pcd0 = o3d.io.read_point_cloud(img[i])
            pcd1 = pcd0
            pcd2=pcd1
            pcd1=pcd1.voxel_down_sample(voxel_size=0.05)
            #normal_param=o3d.geometry.KDTreeSearchParamRadius(0.005)
            pcd1.estimate_normals()
    
            Start=np.array([0,0,0])
            
            #Setup
            center = pcd1.get_center()
            points = np.asarray(pcd1.points)
            normals=np.asarray(pcd1.normals)
            PN=np.array([points,normals])
            PN=np.swapaxes(PN,0,1)
            PNL=np.ndarray.tolist(PN)
            Mymodel=RCylModel()
            Bestmodel=RCylModel()
            
            
            #Computation
            P,Cylinder,pcdVox,pcdInliers=Voxelized_Cylinder(points,pcd1,PNL,Radius,Noise)
            CurvInlier,Curv,CurvRatio=RANSACApprox_Length_Start(P,Start,Noise,Radius,N_Segment,L_Segment)
            pcdSurf,line_set,Curve,Curvedot=Generate_ModelSurf(Curv,Radius)
    
            stop=timeit.default_timer()
            #Analysis
            Test_img.append(img_name[i])
            Result_angle[i*N+j],Result_length[i*N+j],Result_mean_Inl_dist[i*N+j],Result_mean_all_dist[i*N+j],Result_med_Inl_dist[i*N+j],Result_med_all_dist[i*N+j],Result_dev_all_dist[i*N+j],Result_dev_Inl_dist[i*N+j],Result_Max_Inl_Dist[i*N+j],Result_Max_all_Dist[i*N+j]=Evaluate_model(pcdSurf,Curve,Curvedot, pcdref, pcdInliers)
            Result_Time[i*N+j]=stop-start
            if j==0 :
                Analysis_mean_inl[i]=Result_mean_Inl_dist[i*N+j]
                Analysis_dev_inl[i]=Result_dev_Inl_dist[i*N+j]
            else :
                prev_mean=Analysis_mean_inl[i]
                Analysis_mean_inl[i]=((j+1)*Analysis_mean_inl[i]+Result_mean_Inl_dist[i*N+j])/(j+2)
                Analysis_dev_inl[i]=np.sqrt((((j+1)*Analysis_dev_inl[i]**2+Result_dev_Inl_dist[i*N+j]**2)/(j+2))+((j+1)*(prev_mean-Result_mean_Inl_dist[i*N+j])**2/(j+2)**2))
    else:
        for j in range(N):
            Test_img.append(img_name[i])
#Output file 
with open(DIR+'Test_Length_Start.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     Data=np.array([Test_img,Result_angle,Result_length,Result_mean_Inl_dist,
                       Result_mean_all_dist,Result_med_Inl_dist,
                                         Result_med_all_dist,Result_dev_all_dist,Result_dev_Inl_dist,
                                         Result_Max_Inl_Dist,Result_Max_all_Dist,Result_Time])
     
     writer.writerows(Data)

with open(DIR+'Test_Length_Start_Analysis.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     Data=np.array([img_name,Analysis_dev_inl,Analysis_mean_inl])
     
     writer.writerows(Data)