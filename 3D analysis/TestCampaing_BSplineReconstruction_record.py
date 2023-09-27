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
from Arm_Gen import *


DIR="Record_Tangent/"
N=20 #number of test per image
img=glob.glob(DIR +'*.ply')
img_name=copy.deepcopy(img)

Ntest=20
lensurf=300*100

Result_mean_all_dist=np.empty(len(img)*Ntest)
Result_med_all_dist=np.empty(len(img)*Ntest)
Result_Max_all_Dist=np.empty(len(img)*Ntest)
Result_Time=np.empty(len(img)*Ntest)
Result_Acq_Time=np.empty(len(img)*Ntest)
Result_dev_all_dist=np.empty(len(img)*Ntest)
Analysis_dev_all=np.empty(len(img))
Analysis_mean_all=np.empty(len(img))
Result_Theta1=np.empty(len(img)*Ntest)
Result_Phi1=np.empty(len(img)*Ntest)
Result_Theta2=np.empty(len(img)*Ntest)
Result_Phi2=np.empty(len(img)*Ntest)
Fail=np.empty(len(img)*Ntest)
Test_img=[]
First=True



Test_img=[]

Radius=0.02
N_Segment=2
L_Segment=0.16
Noise=0.01
N_Test=20


print(len(img),' image to test')

for i in range(len(img)):
    img_name[i]=img_name[i].replace(DIR,'')
    img_name[i]=img_name[i].replace('.ply','')
    
    
      
    for j in range(N):
        print('Treatment of image ',i+1,'/',len(img),'(',img_name[i],') test ',j+1,'/',N)
        start=timeit.default_timer()
        #acquisition
        pcd0 = o3d.io.read_point_cloud(img[i])
        pcd1 = filterDATA(pcd0)
        pcd2=pcd1
        pcd1=pcd1.voxel_down_sample(voxel_size=0.005)
        stop0=timeit.default_timer()
        #normal_param=o3d.geometry.KDTreeSearchParamRadius(0.005)
        pcd1.estimate_normals()

        Start=np.array([0.015,-0.25,-0.422])
        
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
        CurvInlier,Curv,CurvRatio,success=RANSACApprox_Length_Start(P,Start,Noise,Radius,N_Segment,L_Segment)
        
        stop2=timeit.default_timer()
        
        pcdOutput,line_set,Curve,Curvedot=Generate_ModelSurf(Curv,Radius)
        
        Camera=np.array([0,0,0])
        Cameradir=np.array([0,0,1])
        pcdCamera=filterDATA(pcdOutput)
        pcdCamera,out=Camera_Sim(pcdCamera, Camera, Cameradir, 320, 16/9, 3*np.pi/2, 0)
        
        #o3d.visualization.draw_geometries([pcd2,pcdCamera])
        
        Test_img.append(img_name[i])
        #Analysis
        if success==True:
            Result_mean_all_dist[i*Ntest+j],Result_med_all_dist[i*Ntest+j],Result_dev_all_dist[i*Ntest+j],Result_Max_all_Dist[i*Ntest+j]=Evaluate_model(pcdCamera, pcd1)
            Fail[i*Ntest+j]=0
            Result_Theta1[i*Ntest+j]=0
            Result_Phi1[i*Ntest+j]=0
            Result_Theta2[i*Ntest+j]=0
            Result_Phi2[i*Ntest+j]=0
        else:
            Result_mean_all_dist[i*Ntest+j],Result_med_all_dist[i*Ntest+j],Result_dev_all_dist[i*Ntest+j],Result_Max_all_Dist[i*Ntest+j]=0,0,0,0
            Fail[i*Ntest+j]=1
        Result_Time[i*Ntest+j]=stop2-start
        Result_Acq_Time[i*Ntest+j]=stop0-start
        print(stop2-start)
        
        
        
        if First :
            Analysis_mean_all[i]=Result_mean_all_dist[i*Ntest+j]
            Analysis_dev_all[i]=Result_dev_all_dist[i*Ntest+j]
            First=False
        else :
            prev_mean=Analysis_mean_all[i]
            Analysis_mean_all[i]=((j+1)*Analysis_mean_all[i]+Result_mean_all_dist[i*Ntest+j])/(j+2)
            Analysis_dev_all[i]=np.sqrt((((j+1)*Analysis_dev_all[i]**2+Result_dev_all_dist[i*Ntest+j]**2)/(j+2))+((j+1)*(prev_mean-Result_mean_all_dist[i*Ntest+j])**2/(j+2)**2))
        if j==Ntest:
            First=True

#Output file 
     
with open(DIR+'Test_Spline.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     Data=np.array([Test_img,Result_mean_all_dist,Result_med_all_dist,Result_dev_all_dist,Result_Max_all_Dist,Result_Time,Result_Acq_Time,
                   Result_Theta1,Result_Phi1,Result_Theta2,Result_Phi2,Fail])
     Data=np.transpose(Data)
     
     writer.writerows(Data)

with open(DIR+'Test_Spline_Analysis.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     Data=np.array([img_name,Analysis_dev_all,Analysis_mean_all])
     
     writer.writerows(Data)