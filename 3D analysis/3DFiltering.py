import numpy as np
import open3d as o3d
from pyransac.base import Model
from pyransac import ransac
import pyransac
import copy
from scipy.spatial.transform import Rotation as rot
import random
import geomdl
from geomdl.fitting import approximate_curve
from geomdl.fitting import interpolate_curve

import glob
import timeit
import csv

### Cylinder Model
class CylModel(Model):
    def __init__(self, direction=None, center=None,radius=None):
        self.direction=direction
        self.center=center
        self.radius=radius
        
    def make_model(self, P1N): #Compute a cylinder parameter given 3 point and 2 normals
        
        P2N=P1N[1]
        P3N=P1N[2]
        P1N=P1N[0]
        P1=np.array(P1N[0])
        P2=np.array(P2N[0])
        P3=np.array(P3N[0])
        N1=np.array(P1N[1])
        N2=np.array(P2N[1])

        # Cylinder direction computed from normals
        D=np.cross(N1,N2)
        if np.linalg.norm(np.cross(N1,N2))==0:
            D=np.array([0,0,1])
        else:
            D=np.cross(N1,N2)/np.linalg.norm(np.cross(N1,N2))
            D=D*np.sign(D[2])


        #Coordinate change to the orthogonal plane of D
        alpha=np.arccos(np.dot(D,[0,0,1]))
         
        r=np.cross(D,np.array([0,0,1]))/np.linalg.norm(np.cross(D,np.array([0,0,1]))) 
        if np.isnan(r).any()==True: #For robustness, create random direction if normal are colinear (Which is frequent, if point as no neighbour the ssame arbitrary norml is defined by o3d)
            R=rot.from_quat([1,0,0,0])
            Rinv=rot.from_quat([1,0,0,0])
        else :
            R=rot.from_rotvec(alpha*r)
            Rinv=R.inv()
        Rinv=R.inv()
        P1plane=R.apply(P1)
        P2plane=R.apply(P2)
        P3plane=R.apply(P3)
        P1c=P1plane[0]+P1plane[1]*1j
        P2c=P2plane[0]+P2plane[1]*1j
        P3c=P3plane[0]+P3plane[1]*1j


        #Computation of center and radius
        w=(P3c-P1c)/(P2c-P1c)
        Cc=((P2c-P1c)*(w-abs(w)**2)/(2j*w.imag))+P1c
        R= abs(P2c - Cc)
        Cplane=np.array([np.real(Cc),np.imag(Cc),0])

        C=Rinv.apply(Cplane)
        
        self.direction=D
        self.center=C
        self.radius=R
        return True
    
    def calc_error(self, point): #Compute the distance between a point and the cylinder
        err=np.linalg.norm(np.cross(np.array(point[0])-self.center,self.direction))-self.radius
        err=np.abs(err)
        return err

###Cylinder model with fixed radius
class RCylModel(Model):
    def __init__(self, direction=None, center=None,radius=None):
        self.direction=direction
        self.center=center
        self.radius=radius
        
    def make_model(self, P1N):#Compute a cylinder parameter given 2 point and 2 normals
        R=0.022
        
        P2N=P1N[1]
        P1N=P1N[0]
        P1=np.array(P1N[0])
        P2=np.array(P2N[0])
        N1=np.array(P1N[1])
        N2=np.array(P2N[1])

        # Cylinder direction computed from normals
        D=np.cross(N1,N2)
        if np.linalg.norm(np.cross(N1,N2))==0:
            D=np.array([0,0,1])
        else:
            D=np.cross(N1,N2)/np.linalg.norm(np.cross(N1,N2))
            D=D*np.sign(D[2])


        #Coordinate change to the orthogonal plane of D
        alpha=np.arccos(np.dot(D,[0,0,1]))
         
        r=np.cross(D,np.array([0,0,1]))/np.linalg.norm(np.cross(D,np.array([0,0,1])))
        if np.isnan(r).any()==True:
            Rota=rot.from_quat([1,0,0,0])
            Rinv=rot.from_quat([1,0,0,0])
        else :
            Rota=rot.from_rotvec(alpha*r)
            Rinv=Rota.inv()
        P1plane=Rota.apply(P1)
        P2plane=Rota.apply(P2)
        N1plane=Rota.apply(N1)
        N2plane=Rota.apply(N2)
        P1c=P1plane[0]+P1plane[1]*1j
        P2c=P2plane[0]+P2plane[1]*1j
        N1c=N1plane[0]+N1plane[1]*1j
        N2c=N2plane[0]+N2plane[1]*1j


        #Computation of center as being at distance R from both input points
        C1=P2c+(P1c-P2c)/2+np.sqrt(R**2+((P1c-P2c)**2)/4)*(P1c-P2c)*np.exp(np.pi*1j/2)/np.abs((P1c-P2c))
        C2=P2c+(P1c-P2c)/2-np.sqrt(R**2+((P1c-P2c)**2)/4)*(P1c-P2c)*np.exp(np.pi*1j/2)/np.abs((P1c-P2c))
        beta=np.angle((P1c-P2c)/N2c)
        alpha=np.angle(N1c/N2c)
        inter=P2c+N2c*(np.cos(beta)+np.sin(beta)/np.tan(alpha))
        dist1=np.abs(C1-inter)
        dist2=np.abs(C2-inter)
        #We choose the point being closest to the intersection of the normals
        if dist1 < dist2:
            Cc=C2
        else:
            Cc=C1
        
        Cplane=np.array([np.real(Cc),np.imag(Cc),0])

        C=Rinv.apply(Cplane)
        
        self.direction=D
        self.center=C
        self.radius=R
        return True
    
    def calc_error(self, point): #Compute the distance between a point and the cylinder
        err=np.linalg.norm(np.cross(np.array(point[0])-self.center,self.direction))-self.radius
        err=np.abs(err)
        return err
    
#model for 3D curve (used to find centerline  using ransac on centerpoints)
class Discrete3Dcurve(Model): 
    
    def __init__(self, points=None,obj=None):
        self.points=points
        self.obj=obj

    def make_model(self,input_points): #Compute Nurbs given points
        ###Security against duplicate________________
        is_unique=[]
        is_unique.append(0)
        for i in range(len(input_points)-1):
            test=0
            for j in range(i+1):
               if input_points[i+1][0]==input_points[j][0]and input_points[i+1][1]==input_points[j][1] and input_points[i+1][2]==input_points[j][2]:
                   test+=1
            if test == 0:
                is_unique.append(i+1)
        input_points=[input_points[i] for i in is_unique]
        
        if len(input_points)==2:
            input_points.append((input_points[0]+input_points[1])/2)
            
        ###End points detection______________________ 
        NeigThreshold=100
        is_neighbour=np.empty([len(input_points)],dtype=bool)
        Ends=[]
        is_Ends=np.empty([len(input_points)],dtype=bool)
        pop=[]
        for i in range(len(input_points)):
            dist=input_points-input_points[i]
            dist=np.array(dist)
            for k in range(len(input_points)):
                is_neighbour[k]=np.linalg.norm(dist[k,:]) <= NeigThreshold
            
            dist=dist[is_neighbour]
            (u,s,v)=np.linalg.svd((1/4)*np.matmul(np.transpose(dist),dist))
            Direction=u[:,0]/np.linalg.norm(u[:,0])
            Nbdroite=0
            Nbgauche=0
            for k in range(len(dist)):
                sens=np.dot(dist[k],Direction)
                if sens > 0:
                    Nbdroite+=1
                elif sens <0:
                    Nbgauche+=1
            if Nbdroite==0 or Nbgauche==0 :
                Ends.append(input_points[i])
                is_Ends[i]=True
                pop.append(i)
            else :
                is_Ends[i]=False
                
        ###Reorganization of points__________________
        Porg=input_points
        Porg.pop(max(pop))
        Porg.pop(min(pop))
        Porg.append(Ends[0])
        Porg.reverse()
        Porg.append(Ends[-1])
        
        ###Curve interpolation_______________________
        curve=interpolate_curve(input_points,len(input_points)-1)
        self.obj=curve
        curve.evaluate(start=0,stop=1)
        self.points=curve.evalpts
        return True
    
    def calc_error(self, point):
        dists=np.empty([3,len(self.points)])
        for i in range(len(self.points)):
            dists[:,i]=point-self.points[i]
        err=min(np.sqrt(dists[0,:]**2+dists[1,:]**2+dists[2,:]**2))
        return err


#model for 3D curve with criterion on length
class Discrete3Dcurve_Length(Model):
    
    def __init__(self, points=None,obj=None,target_Length=None):
        self.points=points
        self.obj=obj
        self.target_Length=target_Length

    def make_model(self,input_points):
        ###Security against duplicate________________
        is_unique=[]
        is_unique.append(0)
        for i in range(len(input_points)-1):
            test=0
            for j in range(i+1):
               if input_points[i+1][0]==input_points[j][0]and input_points[i+1][1]==input_points[j][1] and input_points[i+1][2]==input_points[j][2]:
                   test+=1
            if test == 0:
                is_unique.append(i+1)
        input_points=[input_points[i] for i in is_unique]
        ###End points detection______________________ 
        if len(input_points)==2:
            input_points.append((input_points[0]+input_points[1])/2)
            
        NeigThreshold=100
        is_neighbour=np.empty([len(input_points)],dtype=bool)
        Ends=[]
        is_Ends=np.empty([len(input_points)],dtype=bool)
        pop=[]
        for i in range(len(input_points)):
            dist=input_points-input_points[i]
            dist=np.array(dist)
            for k in range(len(input_points)):
                is_neighbour[k]=np.linalg.norm(dist[k,:]) <= NeigThreshold
            
            dist=dist[is_neighbour]
            (u,s,v)=np.linalg.svd((1/4)*np.matmul(np.transpose(dist),dist))
            Direction=u[:,0]/np.linalg.norm(u[:,0])
            Nbdroite=0
            Nbgauche=0
            for k in range(len(dist)):
                sens=np.dot(dist[k],Direction)
                if sens > 0:
                    Nbdroite+=1
                elif sens <0:
                    Nbgauche+=1
            if Nbdroite==0 or Nbgauche==0 :
                Ends.append(input_points[i])
                is_Ends[i]=True
                pop.append(i)
            else :
                is_Ends[i]=False
              
        ###Reorganization of points__________________
        Porg=input_points
        Porg.pop(max(pop))
        Porg.pop(min(pop))
        Porg.append(Ends[0])
        Porg.reverse()
        Porg.append(Ends[-1])
        
        ###Curve interpolation_______________________
        curve=interpolate_curve(input_points,len(input_points)-1)
        self.obj=curve
        curve.evaluate(start=0,stop=1)
        self.points=curve.evalpts
        
        ###CLength Criterion_______________________
        Act_length=geomdl.operations.length_curve(curve)
        Tolerance=0.05
        if Act_length>self.target_Length*(1-Tolerance) and Act_length<self.target_Length*(1+Tolerance):
            Valid=True
        else:
            Valid=False
        return Valid
    
    def calc_error(self, point): #Compute the distance of a given point to the curve
        dists=np.empty([3,len(self.points)])
        for i in range(len(self.points)):
            dists[:,i]=point-self.points[i]
        err=min(np.sqrt(dists[0,:]**2+dists[1,:]**2+dists[2,:]**2))
        return err

#Input data treatment to focus on studied object
def filterDATA(pcd0):
    #Far point removal
    points = np.asarray(pcd0.points)
    z1_threshold=-0.8
    pcd1 = pcd0.select_by_index(np.where(points[:,2] > z1_threshold)[0])
    
    points = np.asarray(pcd1.points)
    z2_threshold=-0.1
    pcd1 = pcd1.select_by_index(np.where(points[:,2] < z2_threshold)[0])
    
    points = np.asarray(pcd1.points)
    x1_threshold=0.2
    pcd1 = pcd1.select_by_index(np.where(points[:,0] < x1_threshold)[0])
    
    points = np.asarray(pcd1.points)
    x2_threshold=-0.2
    pcd1 = pcd1.select_by_index(np.where(points[:,0] > x2_threshold)[0])
    
    #White point removal
    white_threshold=0.3
    colors = np.asarray(pcd1.colors)
    color = (colors[:,0]+colors[:,1]+colors[:,2])/3
    pcd1 = pcd1.select_by_index(np.where(color < white_threshold)[0])
    return pcd1

#Basic linear regression in 3D
def findLine(pcd1): 
    center = pcd1.get_center()
    points = np.asarray(pcd1.points)

    Points_centered=points-center
    (u,s,v)=np.linalg.svd((1/4)*np.matmul(np.transpose(Points_centered),Points_centered))
    D3=u[:,0]/np.linalg.norm(u[:,0])
    
    return center, D3

#Cylinder post-treatment (to be used after ransac). Recompute actual center and return 3D wire cylinder  
def CylinderDipslay(Bestmodel,pcdInliers):
    
    #Parameter retrieaval
    D=Bestmodel.direction
    C=Bestmodel.center
    R=Bestmodel.radius

    alpha=np.arccos(np.dot([0,0,1],D))
    r=np.cross(np.array([0,0,1]),D)/np.linalg.norm(np.cross(np.array([0,0,1]),D)) 
    Rot=rot.from_rotvec(alpha*r)
    Rinv=Rot.inv()

    #Center point recomputing
    pcdDir=copy.deepcopy(pcdInliers)
    #pcdDir.translate(-C/2)
    #pcdDir.rotate(Rinv.as_matrix())
    pointDir=np.asarray(pcdDir.points)
    pointDir=Rinv.apply(pcdDir.points)

    Max=max(pointDir[:,2])
    Min=min(pointDir[:,2])

    H=np.abs(Max-Min)
    center=((Max+Min)/2)
    #center=np.array([0,0,center])

    #center=Rot.apply(center)
    C=C+center*D
    Bestmodel.center=C


    #Cylinder display
    Cylinder=o3d.geometry.TriangleMesh.create_cylinder(radius=R, height=H,split=10)
    Cylinder=Cylinder.rotate(Rot.as_matrix())
    Cylinder=Cylinder.translate(C)
    Cylinder=o3d.geometry.LineSet.create_from_triangle_mesh(Cylinder)
    
    return Cylinder

#Application of cylinder RANSAC on each element of a voxelised space
def Voxelized_Cylinder(points,pcd1,PNL,Mymodel):
    params=ransac.RansacParams(samples=3, iterations=1000, confidence=0.99999, threshold=0.0005)
    densit_threshold=500 #Define the limit to compute cylinder in a voxel

    size=0.02 #Voxel dimension
    densit_threshold*=size
    #voxel grid generation 
    grid=o3d.geometry.VoxelGrid()
    grid=grid.create_from_point_cloud(pcd1,voxel_size=size)

    voxels=np.asarray(grid.get_voxels())
    print('    ',len(voxels)-1,' voxels generated with' ,len(points),'points / applied threshold :', densit_threshold, 'points')
    
    
    X=points[:,0]
    Y=points[:,1]
    Z=points[:,2]
    i=0
    P=[]
    pcdVox=o3d.geometry.PointCloud()
    Cylinder=o3d.geometry.LineSet()
    ratios=[]
    pcdInliers=o3d.geometry.PointCloud()
    for i in range(len(voxels)): #iteration over each voxel
        pcdi=o3d.geometry.PointCloud()
        index = voxels[i].grid_index
        center = grid.get_voxel_center_coordinate(index)
        pcdi=copy.deepcopy(pcd1)
        #Deletion of every point outside voxel
        X=points[:,0]
        Y=points[:,1]
        Z=points[:,2]
        pcdi = pcdi.select_by_index(np.where(X < (center[0]+(size/2)))[0])
        pointsi=np.asarray(pcdi.points)
        X=pointsi[:,0]
        Y=pointsi[:,1]
        Z=pointsi[:,2]
        pcdi = pcdi.select_by_index(np.where(X > (center[0]-(size/2)))[0])
        pointsi=np.asarray(pcdi.points)
        X=pointsi[:,0]
        Y=pointsi[:,1]
        Z=pointsi[:,2]
        pcdi = pcdi.select_by_index(np.where(Y < (center[1]+(size/2)))[0]) 
        pointsi=np.asarray(pcdi.points)
        X=pointsi[:,0]
        Y=pointsi[:,1]
        Z=pointsi[:,2]
        pcdi = pcdi.select_by_index(np.where(Y > (center[1]-(size/2)))[0])
        pointsi=np.asarray(pcdi.points)
        X=pointsi[:,0]
        Y=pointsi[:,1]
        Z=pointsi[:,2]
        pcdi = pcdi.select_by_index(np.where(Z < (center[2]+(size/2)))[0])
        pointsi=np.asarray(pcdi.points)
        X=pointsi[:,0]
        Y=pointsi[:,1]
        Z=pointsi[:,2]
        pcdi = pcdi.select_by_index(np.where(Z > (center[2]-(size/2)))[0])
        pointsi=np.asarray(pcdi.points)
        color=[random.random(),random.random(),random.random()]
        pcdi.paint_uniform_color(color)
        pcdVox +=pcdi
        if len(pcdi.points) > np.abs(densit_threshold): #If there is enought point in given voxel compute Cylinder
            #print('voxel ',i,'/',len(voxels)-1,' computation for ',len(pcdi.points))
            normalsi=np.asarray(pcdi.normals)
            PN=np.array([pointsi,normalsi])
            PN=np.swapaxes(PN,0,1)
            PNL=np.ndarray.tolist(PN)
            
            
            Inliers,Bestmodel,ratio=pyransac.find_inliers(PNL, Mymodel, params)
            if ratio > 0.8:  #If the cylinder is good enough keep it
                #print(' ---->   Inlier ratio :',ratio,' Cylinder kept :',len(P)+1)
                ratios.append(ratio)
                #Inlier reformatting
                Inliers=np.array(Inliers)
                Inliers=Inliers[:,0,:]
                Inliers=np.ndarray.tolist(Inliers)
                newpcdInliers=o3d.geometry.PointCloud()
                newpcdInliers.points=o3d.utility.Vector3dVector(Inliers)
                pcdInliers+=newpcdInliers
                Cylinderi=CylinderDipslay(Bestmodel,newpcdInliers)   # Even if cylinder is not shown keep this line, it recompute the proper center
                Cylinderi.paint_uniform_color(color)
                
                #security against duplicate points
                Punique=True
                for j in range(len(P)):
                    if Bestmodel.center[0]==P[j][0]and Bestmodel.center[1]==P[j][1] and Bestmodel.center[2]==P[j][2]:
                        Punique=False
                        
                if Punique==True:
                    P.append(Bestmodel.center)
                    Cylinder += Cylinderi
        #     else:
        #         print(' ----> Inlier ratio :',ratio,' Cylinder discarded:')
        # else:
        #     print('voxel ',i,'/',len(voxels)-1,' skipped',len(pcdi.points))
        pcdInliers.paint_uniform_color([1,0,0])
    print('    ',len(P),'cylinder generated')
    
    return P,Cylinder,pcdVox,pcdInliers    
    
#Approximation of center line on all center point. Include End points detection /!\ might detect more than two ends 
def DirectApprox(P):
    #End point detection ==========================
    NeigThreshold=0.3
    is_neighbour=np.empty([len(P)],dtype=bool)
    Ends=[]
    is_Ends=np.empty([len(P)],dtype=bool)
    pop=[]
    for i in range(len(P)):
        dist=P-P[i]
        dist=np.array(dist)
        for k in range(len(P)):
            is_neighbour[k]=np.linalg.norm(dist[k,:]) <= NeigThreshold
        
        dist=dist[is_neighbour]
        (u,s,v)=np.linalg.svd((1/4)*np.matmul(np.transpose(dist),dist))
        Direction=u[:,0]/np.linalg.norm(u[:,0])
        Nbdroite=0
        Nbgauche=0
        for k in range(len(dist)):
            sens=np.dot(dist[k],Direction)
            if sens > 0:
                Nbdroite+=1
            elif sens <0:
                Nbgauche+=1
        if Nbdroite==0 or Nbgauche==0 :
            Ends.append(P[i])
            is_Ends[i]=True
            pop.append(i)
        else :
            is_Ends[i]=False
           
    pcdE=o3d.geometry.PointCloud()    
    pcdE.points=o3d.utility.Vector3dVector(Ends)
    pcdE.paint_uniform_color([0,1,0])  
    #Point reorganisation with end points at... both ends
    Porg=P
    #Porg.pop(max(pop))
    #Porg.pop(min(pop))
    Porg.append(Ends[0])
    Porg.reverse()
    Porg.append(Ends[1])
    #Curve approxiamtion
    curve=approximate_curve(Porg, 2,ctrlpts_size=3)
    curve.evaluate(start=0,stop=1)
    curvepoints=curve.evalpts
    Curv=Discrete3Dcurve()
    Curv.points=curvepoints
    Curv.obj=curve
    return curvepoints,pcdE,Curv

#Curve approximation using ransac with Nurbs model
def RANSACApprox(P):
    print('    Centerline RANSAC estimation')
    params=ransac.RansacParams(samples=3, iterations=100, confidence=0.7, threshold=0.01)
    Curv=Discrete3Dcurve()
    CurvInlier,Curv,CurvRatio=pyransac.find_inliers(P, Curv, params)
    print('    Inliers kept :',len(CurvInlier),'/',len(P)-2)
    return CurvInlier,Curv,CurvRatio

#Curve approximation using ransac with Nurbs model with length criterion
def RANSACApprox_Length(P):
    print('    Centerline RANSAC estimation')
    params=ransac.RansacParams(samples=3, iterations=200, confidence=0.7, threshold=0.01)
    Curv=Discrete3Dcurve_Length()
    Curv.target_Length=42*0.044/5
    CurvInlier,Curv,CurvRatio=pyransac.find_inliers(P, Curv, params)
    #If RANSAC with length criterion fail try again without criterion
    if Curv.obj==None:
        Curv=Discrete3Dcurve()
        CurvInlier,Curv,CurvRatio=pyransac.find_inliers(P, Curv, params)
    print('    Inliers kept :',len(CurvInlier),'/',len(P)-2)
    return CurvInlier,Curv,CurvRatio

#Generation of cylindrical surface from Nurbs center line
def Generate_ModelSurf(Curv):
        
    curvepoints2=Curv.points
    Curve=Curv.obj
    
    #Creation of line object with open3d________________________________
    line2=[]
        
    for i in range(len(curvepoints2)-1):
        line2.append([i,i+1])
    
    line_set2 = o3d.geometry.LineSet()
    line_set2.points = o3d.utility.Vector3dVector(curvepoints2)
    line_set2.lines = o3d.utility.Vector2iVector(line2)
    line_set2.paint_uniform_color([1,0,0])
    
    ###Surface parameter________________________________________________
    r=0.022 #radius
    N=100   #Point count along the main direction
    K=50    #Point count along the perimeter
    t=np.linspace(0,1,N)
    theta=np.linspace(0,2*np.pi,K)
    ###Surface computation _____________________________________________
    Surf_Points=[]
    #print('deg=',Curve.degree,'ctrl=',Curve.ctrlpts)
    Curvedot=geomdl.operations.derivative_curve(Curve)
    #Curveddot=geomdl.operations.derivative_curve(Curvedot) #geomdl can't compute second derivativ when degree is too low, it doesn't handle stationnary curve
    for i in range(N): #iteration along axis
        #Fresney Frame computation
        local_center=np.array(Curve.evaluate_single(t[i]))
        local_tangent=np.array(Curvedot.evaluate_single(t[i]))
        local_tangent=local_tangent/np.linalg.norm(local_tangent)
        local_normal=np.cross(local_center,local_tangent)
        local_normal=local_normal/np.linalg.norm(local_normal)
        local_binormal=np.cross(local_normal,local_tangent)
        local_binormal=local_binormal/np.linalg.norm(local_binormal)
        for j in range(K): #iteration along perimeter
            local_point=local_center+r*np.cos(theta[j])*local_normal+r*np.sin(theta[j])*local_binormal
            Surf_Points.append(local_point)
      
    
    
    #o3d surface object creation
    pcdSurf=o3d.geometry.PointCloud()    
    pcdSurf.points=o3d.utility.Vector3dVector(Surf_Points)
    pcdSurf.paint_uniform_color([0,0,0.7])
    pcdSurf.estimate_normals()
    return pcdSurf,line_set2,Curve,Curvedot

#Evaluation of the quality of a model
def Evaluate_model(pcdSurf,Curve,Curvedot,pcd2,pcdInliers):
    Result_all_dist=pcd2.compute_point_cloud_distance(pcdSurf)  
    Result_all_dist=np.asarray(Result_all_dist)
    Result_mean_all_dist=np.mean(Result_all_dist)
    Result_dev_all_dist=np.std(Result_all_dist)
    
    Result_Inl_dist=pcdInliers.compute_point_cloud_distance(pcdSurf)  
    Result_Inl_dist=np.asarray(Result_Inl_dist)
    Result_mean_Inl_dist=np.mean(Result_Inl_dist)
    Result_dev_Inl_dist=np.std(Result_Inl_dist)

    
    Result_length=geomdl.operations.length_curve(Curve)
    Start_dir=np.asarray(Curvedot.evaluate_single(0))
    End_dir=np.asarray(Curvedot.evaluate_single(1))
    Result_angle=np.arcsin(np.linalg.norm(np.cross(End_dir/np.linalg.norm(End_dir),Start_dir/np.linalg.norm(Start_dir))))
    Result_angle*=180/np.pi
    return Result_angle,Result_length,Result_mean_Inl_dist,Result_mean_all_dist

#Data acquisition ============================================================

pcd0 = o3d.io.read_point_cloud("DAta_1_joint/40deg/pc03.ply")
pcd1 = filterDATA(pcd0)
pcd2=pcd1
pcd1=pcd1.voxel_down_sample(voxel_size=0.005)
normal_param=o3d.geometry.KDTreeSearchParamRadius(0.005)
pcd1.estimate_normals()


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

###least squares line=========================================================
'''
(center, D3)=findLine(pcd1)

pcd2=o3d.geometry.PointCloud()

pcd2.points = o3d.utility.Vector3dVector([center])
linepoints=[center-D3/2,center+D3/2]

line=[[0,1]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)

#final visualization
#o3d.visualization.draw_geometries([pcd0,line_set,pcd2])
#o3d.visualization.draw_geometries([pcd1,line_set,pcd2])
#o3d.visualization.draw_geometries([pcd0])
o3d.visualization.draw_geometries([pcd1])
'''

### 3 cylinder point demonstration ===========================================
'''
#input points 
P1=points[1000,:]
P2=points[3000,:]
P3=points[6000,:]
N1=normals[1000,:]
N2=normals[3000,:]
N3=normals[6000,:]
P1N=np.array([P1,N1])
P2N=np.array([P2,N2])
P3N=np.array([P3,N3])
Mymodel.make_model([P1N, P2N, P3N])
D=Mymodel.directionctrlpts_size
C=Mymodel.center
R=Mymodel.radius

linepoints=[P1,P1+N1/10,P2,P2+N2/10]

line=[[0,1],[2,3]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)

pcd3=o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector([P1,P2,P3])
pcd3.paint_uniform_color([0,1,0])
alpha=np.arccos(np.dot(D,[0,0,1]))
 
r=np.cross(D,np.array([0,0,1]))/np.linalg.norm(np.cross(D,np.array([0,0,1]))) 
Rot=rot.from_rotvec(alpha*r)
Rinv=Rot.inv()


Cylinder=o3d.geometry.TriangleMesh.create_cylinder(radius=R, height=0.5)
Cylinder=Cylinder.rotate(Rinv.as_matrix())
Cylinder=Cylinder.translate(C)

Cylinder=o3d.geometry.LineSet.create_from_triangle_mesh(Cylinder)
o3d.visualization.draw_geometries([Cylinder,pcd3,line_set])
'''

###Actual ransac =============================================================
'''
#Ransac parameters
params=ransac.RansacParams(samples=3, iterations=1000, confidence=0.99999, threshold=0.003)
#Ransac application

Inliers,Bestmodel,ratio=pyransac.find_inliers(PNL, Mymodel, params)
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
#o3d.visualization.draw_geometries([pcd1,pcdInliers])
'''

###space partition ===========================================================

P,Cylinder,pcdVox,pcdInliers=Voxelized_Cylinder(points,pcd1,PNL,Mymodel)




#CurvInlier,Curv,CurvRatio=RANSACApprox(P)
CurvInlier,Curv,CurvRatio=RANSACApprox_Length(P)
#curvepoints,pcdE,Curv=DirectApprox(CurvInlier)

pcdInlierscenter=o3d.geometry.PointCloud()
pcdInlierscenter.points=o3d.utility.Vector3dVector(CurvInlier)
pcdInlierscenter.paint_uniform_color([0,0,1])

pcdSurf,line_set,Curve,Curvedot=Generate_ModelSurf(Curv)

Result_angle,Result_length,Result_mean_Inl_dist,Result_mean_all_dist=Evaluate_model(pcdSurf,Curve,Curvedot, pcd2, pcdInliers)

pcdcenter=o3d.geometry.PointCloud()    
pcdcenter.points=o3d.utility.Vector3dVector(P)
pcdcenter.paint_uniform_color([1,0,0])


vis1 = o3d.visualization.VisualizerWithEditing()
vis2 = o3d.visualization.VisualizerWithEditing()
vis3 = o3d.visualization.VisualizerWithEditing()
vis4 = o3d.visualization.VisualizerWithEditing()
vis5 = o3d.visualization.VisualizerWithEditing()
vis6 = o3d.visualization.VisualizerWithEditing()
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


o3d.visualization.draw_geometries([pcd2,line_set,pcdcenter,pcdSurf])
# o3d.visualization.draw_geometries([pcd1,line_set2,pcdcenter,pcdSurf])


###Testing ====================================================================
'''
DIR="DAta_1_joint/40deg/"
N=20 #number of test per image
img=glob.glob(DIR +'*.ply')
img_name=copy.deepcopy(img)

Result_angle=np.empty(len(img)*N)
Result_length=np.empty(len(img)*N)
Result_mean_Inl_dist=np.empty(len(img)*N)
Result_mean_all_dist=np.empty(len(img)*N)
Result_Time=np.empty(len(img)*N)
Test_img=[]

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
        normal_param=o3d.geometry.KDTreeSearchParamRadius(0.005)
        pcd1.estimate_normals()
        
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
        P,Cylinder,pcdVox,pcdInliers=Voxelized_Cylinder(points,pcd1,PNL,Mymodel)
        CurvInlier,Curv,CurvRatio=RANSACApprox_Length(P)
        pcdSurf,line_set,Curve,Curvedot=Generate_ModelSurf(Curv)
        stop=timeit.default_timer()
        #Analysis
        Test_img.append(img_name[i])
        Result_angle[i*N+j],Result_length[i*N+j],Result_mean_Inl_dist[i*N+j],Result_mean_all_dist[i*N+j]=Evaluate_model(pcdSurf,Curve,Curvedot, pcd2, pcdInliers)
        Result_Time[i*N+j]=stop-start

#Output file 
with open(DIR+'Test_Length.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     Data=np.array([Test_img,Result_angle,Result_length,Result_mean_Inl_dist,
                       Result_mean_all_dist,Result_Time])
     
     writer.writerows(Data)
'''