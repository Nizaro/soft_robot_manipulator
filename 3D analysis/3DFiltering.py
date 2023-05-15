import numpy as np
import open3d as o3d
from pyransac.base import Model
from pyransac import ransac
import pyransac
import copy
from scipy.spatial.transform import Rotation as rot
import random

### Cylinder Model
class CylModel(Model):
    def __init__(self, direction=None, center=None,radius=None):
        self.direction=direction
        self.center=center
        self.radius=radius
        
    def make_model(self, P1N):
        
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
        if np.isnan(r).any()==True:
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
        return
    
    def calc_error(self, point):
        err=np.linalg.norm(np.cross(np.array(point[0])-self.center,self.direction))-self.radius
        err=np.abs(err)
        return err

###Cylinder model with fixed radius
class RCylModel(Model):
    def __init__(self, direction=None, center=None,radius=None):
        self.direction=direction
        self.center=center
        self.radius=radius
        
    def make_model(self, P1N):
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


        #Computation of center and radius
        C1=P2c+(P1c-P2c)/2+np.sqrt(R**2+((P1c-P2c)**2)/4)*(P1c-P2c)*np.exp(np.pi*1j/2)/np.abs((P1c-P2c))
        C2=P2c+(P1c-P2c)/2-np.sqrt(R**2+((P1c-P2c)**2)/4)*(P1c-P2c)*np.exp(np.pi*1j/2)/np.abs((P1c-P2c))
        beta=np.angle((P1c-P2c)/N2c)
        alpha=np.angle(N1c/N2c)
        inter=P2c+N2c*(np.cos(beta)+np.sin(beta)/np.tan(alpha))
        dist1=np.abs(C1-inter)
        dist2=np.abs(C2-inter)
        if dist1 < dist2:
            Cc=C2
        else:
            Cc=C1
        
        Cplane=np.array([np.real(Cc),np.imag(Cc),0])

        C=Rinv.apply(Cplane)
        
        self.direction=D
        self.center=C
        self.radius=R
        return
    
    def calc_error(self, point):
        err=np.linalg.norm(np.cross(np.array(point[0])-self.center,self.direction))-self.radius
        err=np.abs(err)
        return err

def filterDATA(pcd0):
    #Far point removal
    points = np.asarray(pcd0.points)
    z_threshold=-1.1
    pcd1 = pcd0.select_by_index(np.where(points[:,2] > z_threshold)[0])
    
    points = np.asarray(pcd1.points)
    x1_threshold=0.2
    pcd1 = pcd1.select_by_index(np.where(points[:,0] < x1_threshold)[0])
    
    points = np.asarray(pcd1.points)
    x2_threshold=-0.2
    pcd1 = pcd1.select_by_index(np.where(points[:,0] > x2_threshold)[0])
    
    #White point removal
    white_threshold=0.4
    colors = np.asarray(pcd1.colors)
    color = (colors[:,0]+colors[:,1]+colors[:,2])/3
    pcd1 = pcd1.select_by_index(np.where(color < white_threshold)[0])
    return pcd1

def findLine(pcd1): 
    center = pcd1.get_center()
    points = np.asarray(pcd1.points)

    Points_centered=points-center
    (u,s,v)=np.linalg.svd((1/4)*np.matmul(np.transpose(Points_centered),Points_centered))
    D3=u[:,0]/np.linalg.norm(u[:,0])
    
    return center, D3

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


    #Cylinder display
    Cylinder=o3d.geometry.TriangleMesh.create_cylinder(radius=R, height=H,split=10)
    Cylinder=Cylinder.rotate(Rot.as_matrix())
    Cylinder=Cylinder.translate(C)
    Cylinder=o3d.geometry.LineSet.create_from_triangle_mesh(Cylinder)
    
    return Cylinder

#Data acquisition ============================================================
pcd0 = o3d.io.read_point_cloud("data_23-05-04_14-56-00/pc3.ply")
pcd1 = filterDATA(pcd0)
normal_param=o3d.geometry.KDTreeSearchParamRadius(0.005)
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
D=Mymodel.direction
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

Inliers,Bestmodel=pyransac.find_inliers(PNL, Mymodel, params)
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
params=ransac.RansacParams(samples=3, iterations=1000, confidence=0.999, threshold=0.003)
densit_threshold=400

color=[[1,0,0],
       [0,1,0],
       [0,0,1],
       [1,1,0],
       [1,0,1],
       [0,0,0],]
size=0.15
grid=o3d.geometry.VoxelGrid()
grid=grid.create_from_point_cloud(pcd1,voxel_size=size)
disp_grid=o3d.geometry.TriangleMesh()
disp_grid=grid.TriangleMesh
voxels=np.asarray(grid.get_voxels())
X=points[:,0]
Y=points[:,1]
Z=points[:,2]
i=0
pcdVox=[]
Cylinder=[]
print(len(voxels))
for i in range(len(voxels)):
    pcdi=o3d.geometry.PointCloud()
    index = voxels[i].grid_index
    center = grid.get_voxel_center_coordinate(index)
    print(center)
    pcdi=copy.deepcopy(pcd1)
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
    pcdi.paint_uniform_color([random.random(),random.random(),random.random()])
    pcdVox.append(pcdi)
    print(i,':',len(pcdi.points))
    if len(pcdi.points) > densit_threshold:
        print('voxel ',i,' computed')
        normalsi=np.asarray(pcdi.normals)
        PN=np.array([pointsi,normalsi])
        PN=np.swapaxes(PN,0,1)
        PNL=np.ndarray.tolist(PN)
        
        
        Inliers,Bestmodel=pyransac.find_inliers(PNL, Mymodel, params)
        #Inlier reformatting
        Inliers=np.array(Inliers)
        Inliers=Inliers[:,0,:]
        Inliers=np.ndarray.tolist(Inliers)
        pcdInliers=o3d.geometry.PointCloud()
        pcdInliers.points=o3d.utility.Vector3dVector(Inliers)
        pcdInliers.paint_uniform_color([1,0,0])
        Cylinderi=CylinderDipslay(Bestmodel,pcdInliers)
        Cylinder.append(Cylinderi)
    else:
        print('voxel ',i,' skipped')
    
    
#disp_grid=o3d.geometry.LineSet.create_from_triangle_mesh(disp_grid)
o3d.visualization.draw_geometries([pcd1]+Cylinder)

