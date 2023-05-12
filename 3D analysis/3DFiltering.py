import numpy as np
import open3d as o3d
from pyransac.base import Model
from pyransac import ransac
import pyransac
import copy
from scipy.spatial.transform import Rotation as rot


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
        D=np.cross(N1,N2)/np.linalg.norm(np.cross(N1,N2))


        #Coordinate change to the orthogonal plane
        alpha=np.arccos(np.dot(D,[0,0,1]))
         
        r=np.cross(D,np.array([0,0,1]))/np.linalg.norm(np.cross(D,np.array([0,0,1]))) 
        R=rot.from_rotvec(alpha*r)
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
        P3N=P1N[2]
        P1N=P1N[0]
        P1=np.array(P1N[0])
        P2=np.array(P2N[0])
        N1=np.array(P1N[1])
        N2=np.array(P2N[1])

        # Cylinder direction computed from normals
        D=np.cross(N1,N2)/np.linalg.norm(np.cross(N1,N2))


        #Coordinate change to the orthogonal plane
        alpha=np.arccos(np.dot(D,[0,0,1]))
         
        r=np.cross(D,np.array([0,0,1]))/np.linalg.norm(np.cross(D,np.array([0,0,1]))) 
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


#Data acquisition
pcd0 = o3d.io.read_point_cloud("data_23-05-04_14-56-00/pc1.ply")
pcd1 = filterDATA(pcd0)
normal_param=o3d.geometry.KDTreeSearchParamRadius(0.005)
pcd1.estimate_normals()

center = pcd1.get_center()
points = np.asarray(pcd1.points)
normals=np.asarray(pcd1.normals)
PN=np.array([points,normals])
PN=np.swapaxes(PN,0,1)
PNL=np.ndarray.tolist(PN)

Mymodel=RCylModel()
Bestmodel=RCylModel()

###least squares line
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

### 3 cylinder point demonstration
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

###Actual ransac

#Ransac parameters
params=ransac.RansacParams(samples=3, iterations=1000, confidence=0.999, threshold=0.008)
#Ransac application
Inliers,Bestmodel=pyransac.find_inliers(PNL, Mymodel, params)
#Data retrieval
Inliers=np.array(Inliers)
Inliers=Inliers[:,0,:]
Inliers=np.ndarray.tolist(Inliers)
pcdInliers=o3d.geometry.PointCloud()
pcdInliers.points=o3d.utility.Vector3dVector(Inliers)
pcdInliers.paint_uniform_color([1,0,0])
#cylinder generation

D=Bestmodel.direction
alpha=np.arccos(np.dot([0,0,1],D))
r=np.cross(np.array([0,0,1]),D)/np.linalg.norm(np.cross(np.array([0,0,1]),D)) 
Rot=rot.from_rotvec(alpha*r)
Rinv=Rot.inv()
pcdDir=copy.deepcopy(pcdInliers)
pcdDir.rotate(Rinv.as_matrix())
pointDir=np.asarray(pcdDir.points)
Max=pcdDir.select_by_index(np.where(pointDir[:,2] ==max(pointDir[:,2])))
Max=np.asarray(Max.points)
Max=Max[0,:]

Min=pcdDir.select_by_index(np.where(pointDir[:,2] ==min(pointDir[:,2])))
Min=np.asarray(Min.points)
Min=Min[0,:]

center=(Max+Min)/2
print(center)
center=Rot.apply(center)

C=Bestmodel.center
C=C+np.dot((center-C),D)*D
R=Bestmodel.radius



Cylinder=o3d.geometry.TriangleMesh.create_cylinder(radius=R, height=0.5,split=10)
Cylinder=Cylinder.translate(C)
Cylinder=Cylinder.rotate(Rot.as_matrix())

Cylinder=o3d.geometry.LineSet.create_from_triangle_mesh(Cylinder)

#Display
o3d.visualization.draw_geometries([pcd0,Cylinder,pcd1,pcdInliers])
#o3d.visualization.draw_geometries([pcd1,pcdInliers])


###space partition
'''
grid=o3d.geometry.VoxelGrid()
grid=grid.create_from_point_cloud(pcd1,0.5)
disp_grid=o3d.geometry.TriangleMesh()
disp_grid=grid.TriangleMesh
#disp_grid=o3d.geometry.LineSet.create_from_triangle_mesh(disp_grid)
o3d.visualization.draw_geometries([pcd0,grid])
'''