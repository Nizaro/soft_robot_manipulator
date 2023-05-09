import numpy as np
import open3d as o3d
from scipy import stats
import pyransac
from scipy.spatial.transform import Rotation as rot


def findCyl(P1N,P2N,P3N):
    P1=P1N[0,:]
    P2=P2N[0,:]
    P3=P3N[0,:]
    N1=P1N[1,:]
    N2=P2N[1,:]

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

    Cproj=-np.dot(C,D)*D+C
    return D,C,R

#Data acquisition
pcd0 = o3d.io.read_point_cloud("data_23-05-04_14-56-00/pc3.ply")

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


#estimation of spread in each plane
center = pcd1.get_center()
points = np.asarray(pcd1.points)
dists = points-center


##Second method, direct 3D computation

Points_centered=points-center
(u,s,v)=np.linalg.svd((1/4)*np.matmul(np.transpose(Points_centered),Points_centered))
D3=u[:,0]/np.linalg.norm(u[:,0])


pcd2=o3d.geometry.PointCloud()

pcd2.points = o3d.utility.Vector3dVector([center])
linepoints=[center-D3/2,center+D3/2]

line=[[0,1]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)



#print(pcd1.has_normals())
#final visualization
#o3d.visualization.draw_geometries([pcd0,line_set,pcd2])
#o3d.visualization.draw_geometries([pcd1,line_set,pcd2])
#o3d.visualization.draw_geometries([pcd0])
#o3d.visualization.draw_geometries([pcd1])

###Third method, cylinder from 2 points with normals
pcd1.estimate_normals()
normals=np.asarray(pcd1.normals)

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



(D,C,R)=findCyl(P1N, P2N, P3N)
alpha=np.arccos(np.dot(D,[0,0,1]))
 
r=np.cross(D,np.array([0,0,1]))/np.linalg.norm(np.cross(D,np.array([0,0,1]))) 
Rot=rot.from_rotvec(alpha*r)
Rinv=Rot.inv()


Cylinder=o3d.geometry.TriangleMesh.create_cylinder(radius=R, height=0.5)
Cylinder=Cylinder.rotate(Rinv.as_matrix())
Cylinder=Cylinder.translate(C)

Cylinder=o3d.geometry.LineSet.create_from_triangle_mesh(Cylinder)

linepoints=[P1,P1+N1/10,P2,P2+N2/10]

line=[[0,1],[2,3]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)

pcd3=o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector([P1,P2,P3])
pcd3.paint_uniform_color([0,1,0])
pcd4=o3d.geometry.PointCloud()
o3d.visualization.draw_geometries([pcd3,Cylinder,line_set])