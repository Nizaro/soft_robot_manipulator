import numpy as np
import open3d as o3d
from scipy import stats

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

spreadXY=np.mean(np.sqrt(dists[:,0]**2+dists[:,1]**2))
spreadYZ=np.mean(np.sqrt(dists[:,1]**2+dists[:,2]**2))
spreadZX=np.mean(np.sqrt(dists[:,2]**2+dists[:,0]**2))

#We compute two linear regression which give us two plane 
#(should be changed so the linear regression are done in the two plane of maximum spread)
if spreadXY==min([spreadXY,spreadYZ,spreadZX]):
    print(1)
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(points[:,1], points[:,2])    
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(points[:,2], points[:,0])
    #D=np.cross(N1,N2)/np.linalg.norm(np.cross(N1,N2))
    D2=np.array([slope1*slope2,1,slope1])
    D2=D2/np.linalg.norm(D2)
    p0=np.array([slope2*intercept1+intercept2,0,intercept1])
    
elif spreadYZ==min([spreadXY,spreadYZ,spreadZX]):
    print(2)
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(points[:,2], points[:,0])    
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(points[:,0], points[:,1])
    D2=np.array([slope1,slope1*slope2,1])
    D2=D2/np.linalg.norm(D2)
    p0=np.array([intercept1,slope2*intercept1+intercept2,0])
    
elif spreadZX==min([spreadXY,spreadYZ,spreadZX]):
    print(3)
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(points[:,0], points[:,1])    
    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(points[:,1], points[:,2])
    D2=np.array([1,slope1,slope1*slope2])
    D2=D2/np.linalg.norm(D2)
    p0=np.array([0,intercept1,slope2*intercept1+intercept2])
#Computation of the line as the intersection of the plane

##Second method, direct 3D computation

Points_centered=points-center
(u,s,v)=np.linalg.svd((1/4)*np.matmul(np.transpose(Points_centered),Points_centered))
D3=u[:,0]/np.linalg.norm(u[:,0])


pcd2=o3d.geometry.PointCloud()
'''
pcd2.points = o3d.utility.Vector3dVector([p0])
linepoints=[p0-D2,p0+D2]
'''
pcd2.points = o3d.utility.Vector3dVector([center])
linepoints=[center-D3/2,center+D3/2]

line=[[0,1]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)


#pcd1.estimate_normals()
#print(pcd1.has_normals())
#final visualization
o3d.visualization.draw_geometries([pcd0,line_set,pcd2])
#o3d.visualization.draw_geometries([pcd1,line_set,pcd2])
#o3d.visualization.draw_geometries([pcd0])
#o3d.visualization.draw_geometries([pcd1])