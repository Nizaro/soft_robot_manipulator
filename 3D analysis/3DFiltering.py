import numpy as np
import open3d as o3d
from scipy import stats


pcd1 = o3d.io.read_point_cloud("data_23-05-04_14-56-00/pc0.ply")

points = np.asarray(pcd1.points)
z_threshold=-1.1
pcd1 = pcd1.select_by_index(np.where(points[:,2] > z_threshold)[0])

points = np.asarray(pcd1.points)
x1_threshold=0.2
pcd1 = pcd1.select_by_index(np.where(points[:,0] < x1_threshold)[0])

points = np.asarray(pcd1.points)
x2_threshold=-0.2
pcd1 = pcd1.select_by_index(np.where(points[:,0] > x2_threshold)[0])

white_threshold=0.4
colors = np.asarray(pcd1.colors)
color = (colors[:,0]+colors[:,1]+colors[:,2])/3
pcd1 = pcd1.select_by_index(np.where(color < white_threshold)[0])

#o3d.visualization.draw(pcd1)

center = pcd1.get_center()
points = np.asarray(pcd1.points)
dists = points-center
spreadXY=np.mean(np.sqrt(dists[:,0]**2+dists[:,1]**2))
spreadYZ=np.mean(np.sqrt(dists[:,1]**2+dists[:,2]**2))
spreadZX=np.mean(np.sqrt(dists[:,2]**2+dists[:,0]**2))

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(points[:,1], points[:,2])
N1=np.array([0,-slope1,1])/np.linalg.norm(np.array([0,-slope1,1]))

slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(points[:,2], points[:,0])
N2=np.array([1,0,-slope2])/np.linalg.norm(np.array([1,0,-slope2]))

D=np.cross(N1,N2)/np.linalg.norm(np.cross(N1,N2))
D2=np.array([slope1*slope2,1,slope1])
D2=D2/np.linalg.norm(D2)
p0=np.array([slope2*intercept1+intercept2,0,intercept1])
pcd2=o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector([p0])
linepoints=[p0-D,p0+D]
line=[[0,1]]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(linepoints)
line_set.lines = o3d.utility.Vector2iVector(line)
        
o3d.visualization.draw_geometries([pcd1,line_set,pcd2])