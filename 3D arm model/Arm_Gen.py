# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:47:46 2023

@author: anselme
"""
"""
This code aim to demonstrate the capability of creating
a 2D spline of defined length using Pythagorean Hodograph curves
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as Rot
import random
#import open3d as o3d
import copy
import csv


#First equation to solve to find satisfying PH curves
def PHA(x,a,c,k):
    F=np.sqrt(a**2-x**2)-np.sqrt(3)*x-2*c-k*(x+np.sqrt(3)*np.sqrt(a**2-x**2))
    return F

#Second equation to solve to find satisfying PH curves
def PHB(x,a,c,k):
    F=np.sqrt(a**2-x**2)+np.sqrt(3)*x+2*c-k*(x-np.sqrt(3)*np.sqrt(a**2-x**2))
    return F

#solving of the first equation 1 or 2 solutions exists
def PHsolveA(a,b,c,t):
    t1=np.real(t)
    t2=np.imag(t)
    if t2==0:
        X=-np.sqrt(3*a)/2
    else:
        k=(b*t1)/(a*t2)
        args=(a,c,k)
        x1=fsolve(PHA,-a*0.99,args)
        x2=fsolve(PHA,a*0.99,args)
        if np.abs(x1-x2)/a <0.0001:
            X=x1
        else:
        
            X=np.array([x1[0],x2[0]])
    return X
    
#solving of the second equation at most 1 solution exists
def PHsolveB(a,b,c,t):
    t1=np.real(t)
    t2=np.imag(t)
    if t2==0:
        X=np.sqrt(3*a)/2
    else:
        k=(b*t1)/(a*t2)
        args=(a,c,k)
        X=fsolve(PHA,0,args)          
    return X

#exact solution of the first equation (same will be done for second equation)
def PHFindA(a,b,c,t):
    t1=np.real(t)
    t2=np.imag(t)
    if t2==0:
        X=-np.sqrt(3*a)/2
    else:
        k=(b*t1)/(a*t2)
        '''
        A=1/(8*(k**2+1))
        B1=(c**2)*(4*k+4*np.sqrt(3))**2
        B2=-4*(4*k**2+4)*((a**2)*(-3*k**2+2*np.sqrt(3)*k-1)+4*c**2)
        C=-c*k-4*np.sqrt(3)*c
        x1=A*(-np.sqrt(B1+B2)+C)
        x2=A*(np.sqrt(B1+B2)+C)
        '''
        A=4*(1+k**2)
        B=4*c*(k+np.sqrt(3))
        C=4*c**2-(a-a*np.sqrt(3)*k)**2
        
        x1=(-B+np.sqrt(B**2-4*A*C))/(2*A)
        x2=(-B-np.sqrt(B**2-4*A*C))/(2*A)

        if x1==x2:
            X=x1
        else:
            X=np.array([x1,x2])
        for i in range(X.size):
            if X[i]>a or X[i]<-a:
                X=np.delete(X,i)
                break
        for i in range(X.size):
            if X[i]>a or X[i]<-a:
                X=np.delete(X,i)
                break    
            
        return X

#function that return the desired spline and the bezier control point of it's derivative
def FixedLengthPH(p0,p1,L,t):

#Input
#p0 is the starting point of the desired spline
#p1 is th end point of the desired spline
#t represent the direction of the tangent at p1 (tangent at p1 is undefined as cubic PH curve do not have enough degree of freedom)
#L is the desired length of the curve
# /!\  L should not be shorter than |p1-p0|
    
    
    # coordinate change to canonical space
    c=np.abs(p1-p0)/2
    theta=np.angle(p1-p0)
    a=L/2
    b=np.sqrt(a**2-c**2)
    B=(p1+p0)/2
    Mult=np.cos(theta)-np.sin(theta)*1j

    p0c=(p0-B)*Mult
    p1c=(p1-B)*Mult
    tc=t*Mult
    
    #set up to compute upper semi-ellipse solutions
    if np.imag(tc)>0:
        flip=1
        tc=np.conjugate(tc)
    else:
        flip=0
        
    
    #Computation of the rectifying control polygon admiting desired length and tangent
    #XA=PHsolveA(a,b,c,tc)
    XA=PHFindA(a,b,c,tc)
    #XB=PHsolveB(a,b,c,tc)
    
    #X=np.append(XA,XB)
    X=XA
    Y=np.empty(X.size)
    Y=b*np.sqrt(1-(X**2/a**2))
    
    Pc=np.empty(X.size,dtype=complex)
    
    if flip==1:
        Pc=X+Y*1j
    else:
        Pc=X-Y*1j
    
    
    #Coordinate change to initial space
    Inv=np.cos(theta)+np.sin(theta)*1j
    P=Inv*Pc+B
               

    #hodograph Bezier control point calculation
    sqdeltp0=np.sqrt(P-p0)
    sqdeltp1=np.sqrt(p1-P)
    
    z0=(np.sqrt(2)/2)*(+(1+np.sqrt(3))*sqdeltp0+(1-np.sqrt(3))*sqdeltp1) 
    z0=np.append(z0,(np.sqrt(2)/2)*(+(1+np.sqrt(3))*sqdeltp0-(1-np.sqrt(3))*sqdeltp1)) 
    z1=(np.sqrt(2)/2)*(+(1-np.sqrt(3))*sqdeltp0+(1+np.sqrt(3))*sqdeltp1)
    z1=np.append(z1,(np.sqrt(2)/2)*(+(1-np.sqrt(3))*sqdeltp0-(1+np.sqrt(3))*sqdeltp1))
    #z0=np.append(z0,-z0)
    #z1=np.append(z1,-z1)
    #configuration filtering
    m=np.imag(z0**2/t)
    n=np.real(z0**2/t)
    Bad=[]
    
    for j in range(z0.size):
        if n[j]<0:
            Bad.append([j])
            
    m=np.delete(m,Bad)
    z0=np.delete(z0,Bad)
    z1=np.delete(z1,Bad)
    
    for j in range(z0.size):
        if m[j]**2==min(m**2):
            z0=z0[j]
            z1=z1[j]
            break
        
    #Curve computation
    temp=np.linspace(0,1,101,dtype=complex)
    #i=0 #for debuging
    #r0=p0+((-temp*z0[i]+temp*z1[i]+z0[i])**3-z0[i]**3)/(3*(z1[i]-z0[i]))
    t=(z0*(1-temp)+z1*temp)**2
    r0=p0+((-temp*z0+temp*z1+z0)**3-z0**3)/(3*(z1-z0))
    return z0,z1,r0,t  

def PH3D(p0,p1,t0,L):
    
    
    #Centering
    offset=-(p1+p0)/2
    pa0=p0+offset
    pa1=p1+offset
    
    #rotation to plane XY
    A=np.linalg.norm(np.cross(pa1,t0))
    n=np.cross(pa1,t0)/(A)
    alpha=np.arccos(np.dot(n,[0,0,1]))
    
    r=np.cross(n,np.array([0,0,1]))/np.linalg.norm(np.cross(n,np.array([0,0,1]))) 
    R=Rot.from_rotvec(alpha*r)
    Rinv=R.inv()
    pb0=R.apply(pa0)
    pb1=R.apply(pa1)
    tb0=R.apply(t0)
    
    #Formating for 2D function
    p02D=pb0[0]+pb0[1]*1j
    p12D=pb1[0]+pb1[1]*1j
    t02D=tb0[0]+tb0[1]*1j
    
    #2D calculation
    zb0,zb1,rb0,tb=FixedLengthPH(p02D, p12D, L, t02D)
    rb0=np.transpose(np.array([np.real(rb0),np.imag(rb0),np.zeros(rb0.size)]))
    zb0=np.array([np.real(zb0**2),np.imag(zb0**2),0])
    tb1=np.array([np.real(zb1**2),np.imag(zb1**2),0])
    tb=np.array([np.real(tb),np.imag(tb),np.zeros(tb.size)])
    rot=np.empty(101,dtype=rotation)
    t=np.empty([3,101])
    
    #rotation to initial reference
    r0=Rinv.apply(rb0)-offset
    z0=Rinv.apply(zb0)
    t1=Rinv.apply(tb1)
    
    #Rotation computation
    #axerot=np.cross(t0,t1)/np.linalg.norm(np.cross(t0,t1))
    rot[0]=Rot.identity()
    for i in range (101):
        t[:,i]=Rinv.apply(tb[:,i])
    for i in range(100):
        axerot=np.cross(t0,t[:,i])/np.linalg.norm(np.cross(t0,t[:,i]))
        alpha=np.arccos(np.dot(t0,t[:,i+1])/(np.linalg.norm(t0)*np.linalg.norm(t[:,i+1])))
        rotb=Rot.from_rotvec(alpha*axerot)
        rot[i+1]=rotb

    return(t1,r0,rot)

def PCC3D(p1,p2,Q,n):
    Sn=np.array([n,3])
    r=np.linalg.norm(p2-p1)
    Sn=np.empty([n,3])
    Qn=np.empty([n],dtype=rotation)
    ctheta=np.dot((p2-p1),Q.apply([0,0,1]))/np.linalg.norm(p2-p1)
    theta=np.arccos(ctheta)
    L=r*theta/np.sin(theta)
    rho=L/(2*theta)
    rotvec=-np.cross(Q.apply([0,0,1]),(p2-p1))/np.linalg.norm(np.cross(Q.apply([0,0,1]),(p2-p1)))
    rot=Rot.from_rotvec(theta*rotvec)
    O=rho*np.cross(Q.apply([0,0,1]),rotvec)
    for i in range(n):
        thetai=theta*i/(n-1)
        roti=Rot.from_rotvec((-thetai*2)*rotvec)
        Sn[i,:]=p1+O-roti.apply(O)
        Qn[i]=roti*Q
    
    
    return Sn,Qn
    
def MultiPCC3D(P,Q,m):
    n=np.size(P)//3
    S=np.empty([m,3,n])
    Qn=np.empty([m,n],dtype=rotation)
    for i in range(n-1):
        S[:,:,i],Qn[:,i]=PCC3D(P[i],P[i+1],Q[i],m)
        

    return S,Qn

def MultiPH3D(p,L):
    n=p.size//3
    r=np.empty([101,3,n])
    q=np.empty([101,n],dtype=rotation)
    Q=np.empty([101,n],dtype=rotation)
    t=np.empty([3,n+1])
    t[:,0]=np.array([0,0,1])
    Q1=Rot.identity()
    tang=np.empty([3,101,n])
    for i in range(n-1):
        print('p0',p[:,i])
        print('p1',p[:,i+1])
        print('t0',t[:,i])
        (t[:,i+1],r[:,:,i],q[:,i])=PH3D(p[:,i],p[:,i+1],t[:,i],L)
        
        
        for j in range(101):
            Q[j,i]=q[j,i]*Q1

        Q1=Q[100,i]
    return r,Q
rotation=np.dtype(Rot)

def SmoothConstruct(p,L,k,ray):
    Ex=np.array([1,0,0])
    Ey=np.array([0,1,0])
    Ez=np.array([0,0,1])
    n=(p.size//3)-1
    S=np.empty([3,101,n,k])
    rn=np.empty([3,k])
    (r,Q)=MultiPH3D(p,L)
    
    for i in range(k):
        rn[:,i]=ray*np.cos(np.pi*2*i/k)*Ex+ray*np.sin(np.pi*2*i/k)*Ey
        for j in range(n):
            for m in range(101):
                S[:,m,j,i]=r[m,:,j]+Q[m,j].apply(rn[:,i])
                
    
    return S

def SmoothConstructPCC(p,L,k,ray,Q,m):
    Ex=np.array([1,0,0])
    Ey=np.array([0,1,0])
    Ez=np.array([0,0,1])
    n=(p.size//3)-1
    S=np.empty([3,m+1,n,k])
    rn=np.empty([3,k])
    Qn=np.empty([n,k],dtype=rotation)
    (r,Qn)=MultiPCC3D(p,Q,m)
    
    for i in range(k):
        rn[:,i]=ray*np.cos(np.pi*2*i/k)*Ex+ray*np.sin(np.pi*2*i/k)*Ey
        for j in range(n):
            for v in range(m):
                S[:,v,j,i]=r[v,:,j]+Qn[v,j].apply(rn[:,i])
                
    
    return S

def EndConstruct(P,Q,points,m):
    Ex=np.array([1,0,0])
    Ey=np.array([0,1,0])

    N =ray*m//L
    for a in range(int(N)):
        if a>0:
            for b in range(a*7):
                points.append(P[0]+ray*(a/N)*(np.cos(np.pi*2*b/(a*3))*Ex+np.sin(np.pi*2*b/(a*3))*Ey)) 

    Ex=Q[-1].apply(Ex)       
    Ey=Q[-1].apply(Ey) 
              
    for a in range(int(N)):
        if a>0:
            for b in range(a*7):
                points.append(P[-1]+ray*(a/N)*(np.cos(np.pi*2*b/(a*3))*Ex+np.sin(np.pi*2*b/(a*3))*Ey))          
      
    return points

def SegmentedConstruct(p,L,k,ray,nb):
    Ex=np.array([1,0,0])
    Ey=np.array([0,1,0])
    Ez=np.array([0,0,1])
    n=(p.size//3)-1
    S=np.empty([3,101,n,k])
    rn=np.empty([3,k])
    (r,Q)=MultiPH3D(p,L)
    leng=100//nb
    print(leng)
    for i in range(k):
        rn[:,i]=ray*np.cos(np.pi*2*i/k)*Ex+ray*np.sin(np.pi*2*i/k)*Ey
        for j in range(n):
            for l in range(nb) :
                P1=Q[l*leng,j].apply(rn[:,i])+r[l*leng,:,j]
                P2=Q[(l+1)*leng,j].apply(rn[:,i])+r[(l+1)*leng,:,j]
                for m in range(leng+1):
                    S[:,l*leng+m,j,i]=(P1*(leng-m)/leng)+(P2*m/leng)
    
    return S

def SegmentedConstruct2(p,L,k,ray,Q):
    Ex=np.array([1,0,0])
    Ey=np.array([0,1,0])
    Ez=np.array([0,0,1])
    n=(p.size//3)-1
    S=np.empty([3,101,n,k])
    rn=np.empty([3,k])
    (r,Q)=MultiPCC3D(p,Q)
    for i in range(k):
        rn[:,i]=ray*np.cos(np.pi*2*i/k)*Ex+ray*np.sin(np.pi*2*i/k)*Ey
        for j in range(n):
            P1=Q[j].apply(rn[:,i])+r[0,:,j]
            P2=Q[j+1].apply(rn[:,i])+r[99,:,j]
            for m in range(99+1):
                S[:,m,j,i]=(P1*(99-m)/99)+(P2*m/99)
    
    return S

def PCCrandom(N,L,rep):
    theta=np.empty([N])
    phi=np.empty([N])
    r=np.empty([N])
    for i in range(N//rep):
        theta[rep*i]=random.random()*np.pi/3
        phi[rep*i]=random.random()*2*np.pi
        r[rep*i]=L*np.sin(theta[i])/theta[i]
        for j in range(rep-1):
            theta[rep*i+j+1]=theta[rep*i]
            phi[rep*i+j+1]=phi[rep*i]
            r[rep*i+j+1]=r[rep*i]    
    
    return phi,theta,r

def SymPHrandom(N,L,rep):
    theta=np.empty([N])
    phi=np.empty([N])
    r=np.empty([N])
    for i in range(N//rep):
        theta[rep*i]=random.random()*3*np.pi/3
        phi[rep*i]=random.random()*2*np.pi
        r[rep*i]=L*((2*np.cos(theta[i])+1)/(np.cos(theta[i])+2))
        for j in range(rep-1):
            theta[rep*i+j+1]=theta[rep*i]
            phi[rep*i+j+1]=phi[rep*i]
            r[rep*i+j+1]=r[rep*i] 

    return phi,theta,r

def General_Construct(phi,theta,r):
    p=np.empty([len(phi)+1,3])
    P=np.empty([len(phi)+1,3])
    q=np.empty([len(phi)+1],dtype=rotation)
    Q=np.empty([len(phi)+1],dtype=rotation)
    

    p[0,:]=np.array([0,0,0])
    P[0,:]=np.array([0,0,0])
    q[0]=Rot.from_quat([0,0,0,1])
    Q[0]=Rot.from_quat([0,0,0,1])
    for i in range(len(phi)):
        a=Rot.from_euler('ZYX',[phi[i],theta[i],0])
        V=a.apply([0,0,1])
        rotvec=np.cross([0,0,1],V)/np.linalg.norm(np.cross([0,0,1],V))
        q[i+1]=Rot.from_rotvec(2*theta[i]*rotvec)
        #q[i+1]=Rot.from_euler('ZYX',[phi[i],2*theta[i],0])
        p[i+1,:]=r[i]*a.apply([0,0,1])
        Q[i+1]=Q[i]*q[i+1]
        P[i+1,:]=P[i,:]+Q[i].apply(p[i+1,:])   
    return P,Q

def Camera_Sim(pcdin,position,orientation,xdef,Aspect_ratio,FOV,Noise_level):
    out=0
    orientation=orientation/np.linalg.norm(orientation)
    xmax=np.tan(FOV/2)
    ymax=np.tan(FOV/(2/Aspect_ratio))
    ydef=int(xdef*Aspect_ratio)

    xres=2*xmax/xdef
    yres=2*ymax/ydef
    pcd2=copy.deepcopy(pcdin)
    points=np.asarray(pcd2.points)
    points=points-position

    ctheta=np.dot(orientation,[0,0,1])
    theta=np.arccos(ctheta)

    rotvec=np.cross(orientation,[0,0,1])/np.linalg.norm(np.cross(orientation,[0,0,1]))
    rot=Rot.from_rotvec(theta*rotvec)


    points=rot.apply(points)
    #points=points+Camera
    pcd2.points=o3d.utility.Vector3dVector(points)

    pcd3=copy.deepcopy(pcd2)
    PointCam=o3d.geometry.PointCloud()
    PointCam.points=o3d.utility.Vector3dVector([position,position+orientation])
    PointCam.paint_uniform_color([0,0,0])
    #o3d.visualization.draw_geometries([pcd,PointCam,pcd2])

    points=np.asarray(pcd3.points)

    points[:,0]=(points[:,0]/points[:,2])
    points[:,1]=(points[:,1]/points[:,2])

    points[:,0]=(points[:,0]//xres)+xdef/2
    points[:,1]=points[:,1]//yres+ydef/2

    #print('Point sorting')
    SortedPoints= points[(-points[:,2]).argsort()]

    #print('Pixels computation')
    Pixels=np.ones([xdef,ydef,3])
    Pixels=Pixels*20

    l=len(SortedPoints)
    for i in range(l):
        A=SortedPoints[i,:]
        if A[0]>=0 and A[1]>=0 and A[0]<xdef and A[0]<ydef:
            Pixels[int(A[0]),int(A[1]),:]=A
        else :
            out+=1

    #print('pixel reshape')
    
    Pixels=np.reshape(Pixels,(xdef*ydef,3),'F')
    Pixels[:,0]+= np.full((len(Pixels)),-xdef/2)
    Pixels[:,1]+= np.full((len(Pixels)),-ydef/2)

    #print('spatial transform')
    points=Pixels
    pcdCamera=o3d.geometry.PointCloud()
    pcdCamera.points=o3d.utility.Vector3dVector(points)
    pcdCamera = pcdCamera.select_by_index(np.where(points[:,2] !=20)[0])
    points=np.asarray(pcdCamera.points)
    #insert noise

    points[:,2]+=np.random.normal(0,Noise_level,len(points))
    points[:,0]=points[:,0]*xres*points[:,2]
    points[:,1]=points[:,1]*yres*points[:,2]

    


    rot=rot.inv()
    points=rot.apply(points)
    #print('pointcloud creation')

    pcdCamera.points=o3d.utility.Vector3dVector(points)
    #print('2')
    pcdCamera.translate(position)
    #print('3')
    return pcdCamera,out

###Single segment interpolation ===============================================
'''
L=2
p0=np.array([1,0,0])
p1=np.array([1.2,0.2,0.6])
t0=np.array([1,0.5,-1])

(t1,r0)=PH3D(p0,p1,t0,L)
tv=np.array([p0,p0+t0])
t1v=np.array([p1,p1+t1*L/np.linalg.norm(t1)])

color=['b','g','r','c','m','y','k']

ax = plt.axes(projection='3d')
ax.scatter(p0[0],p0[1],p0[2],color=color[0],label='Input points')
ax.scatter(p1[0],p1[1],p1[2],color=color[1])
ax.plot3D(tv[:,0],tv[:,1],tv[:,2],label='Input tangent')
ax.plot3D(r0[:,0],r0[:,1],r0[:,2],label='Solution')
ax.plot3D(t1v[:,0],t1v[:,1],t1v[:,2],label='output direction')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(loc='upper left')
ax.set_aspect('equal')
plt.show()
'''
### Multi-segment interpolation ===============================================
'''
L=2
n=4
p=np.empty([3,n+1])
t=np.empty([3,n+1])
x=np.empty([3,n+1])
y=np.empty([3,n+1])
Ex=np.array([1,0,0])
Ey=np.array([0,1,0])
Ez=np.array([0,0,1])
x[:,0]=np.array([1,0,0])
y[:,0]=np.array([0,1,0])
t[:,0]=np.array([0,0,1])
p[:,0]=np.array([0,0,0])

p[:,1]=np.array([0.5,0.7,1.7])
p[:,2]=np.array([-0.6,0.5,3])
p[:,3]=np.array([0,-0.8,3.8])
p[:,4]=np.array([1,0.5,3.5])
tang=np.empty([3,101,n])
(r,Q)=MultiPH3D(p,L)


color=['b','g','r','c','m','y','k']
ax = plt.axes(projection='3d')
ax.set_aspect('equal')


for i in range(n):
    ax.plot3D(r[:,0,i],r[:,1,i],r[:,2,i])


    for j in range(10):
        Qx=Q[10*j,i].apply(Ex)
        Qy=Q[10*j,i].apply(Ey)
        Qz=Q[10*j,i].apply(Ez)
        #Qz=tang[:,10*j,i]
        Qxv=np.array([r[10*j,:,i],r[10*j,:,i]+Qx/3])
        Qyv=np.array([r[10*j,:,i],r[10*j,:,i]+Qy/3])
        Qzv=np.array([r[10*j,:,i],r[10*j,:,i]+Qz/3])
        ax.plot3D(Qxv[:,0],Qxv[:,1],Qxv[:,2],color=color[1])
        ax.plot3D(Qyv[:,0],Qyv[:,1],Qyv[:,2],color=color[2])
        ax.plot3D(Qzv[:,0],Qzv[:,1],Qzv[:,2],color=color[0])
    Q1=Q[100,i]


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')

plt.show()
'''
###surface construction =======================================================

ax = plt.axes(projection='3d')
L=2
n=1
k=30
ray=0.25
p=np.empty([3,n+1])
t=np.empty([3,n+1])
x=np.empty([3,n+1])
y=np.empty([3,n+1])
Ex=np.array([1,0,0])
Ey=np.array([0,1,0])
Ez=np.array([0,0,1])
x[:,0]=np.array([1,0,0])
y[:,0]=np.array([0,1,0])
t[:,0]=np.array([0,0,1])
p[:,0]=np.array([0,0,0])
'''
p[:,1]=np.array([0.5,0.7,1.7])
p[:,2]=np.array([-0.6,0.5,3])
p[:,3]=np.array([0,-0.8,3.8])
#p[:,4]=np.array([1,0.5,3.5])
'''

phi,theta,r=SymPHrandom(n, L, 1)
p,Q=General_Construct(phi, theta, r)
print('p',p)
print('p t',np.transpose(p))
S1=SmoothConstruct(np.transpose(p), L, k, ray)
#S2=SegmentedConstruct(p, L, k, ray,4)
for j in range(n):
    for l in range(50):
        ax.plot3D(S1[0,l*2,j,:],S1[1,l*2,j,:],S1[2,l*2,j,:],color='b',linestyle='-')
        #ax.plot3D(S2[0,l*5,j,:],S2[1,l*5,j,:],S2[2,l*5,j,:],color='r',linestyle='-')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')
plt.show()       


#print('Arm Generation')









######General parameters _____________________________________________________
'''
m=400
ray=0.22
N=3
L=2
k=int(m*ray*np.pi*2/L)
rep=1
Camera=np.array([-12,0,0])
Cameradir=np.array([2,0,1])

######iteration parameters ___________________________________________________

Ntest=20
Nnoise=4
NoiseStep=0.05
NBatch=4
MainDir='PCC/'
Lout=np.empty([Ntest*Nnoise])
Lname=np.empty([Ntest*Nnoise],dtype=str)
for isegment in range(NBatch):
    N=isegment+1 
    localDir=MainDir+str(N)+'_segments/'
    for iSample in range(Ntest):
        print('generating image ', iSample+1,'/',Ntest,' of serie ',isegment+1,'/',NBatch)
        #####Surface generation ______________________________________________________
        phi,theta,r=PCCrandom(N, L,rep)
        P,Q=General_Construct(phi, theta, r)
        S,Qn=MultiPCC3D(P, Q,m)
        S2=SmoothConstructPCC(P, Q, k, ray,Q,m)
        points=[]
        
        #ax = plt.axes(projection='3d')
        for j in range(N):
            
            for l in range(m):
                
                #ax.plot3D(S2[0,l,j,:],S2[1,l,j,:],S2[2,l,j,:],color='r',linestyle='-')
                for i in range(k):
                    points.append(S2[:,l,j,i])
        
        points=EndConstruct(P, Q, points, m)
              
        pcd=o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([1,0,0])
        pcd.estimate_normals()
        print('      Ground Truth')
        o3d.io.write_point_cloud(localDir+str(iSample)+"-Ground_Truth.ply", pcd,write_ascii=True,print_progress=True)
        
        #####Camera simulation _______________________________________________________
        
        for iNoise in range(Nnoise):
            print('      Noise level ',iNoise+1,'/',Nnoise)
        
            pcdCamera,out=Camera_Sim(pcd, Camera, Cameradir, 1080, 16/9, np.pi/3, iNoise*NoiseStep)
            Lout[iSample*Nnoise+iNoise]=out
            name=str(iSample)+"-Camera-"+str(iNoise*NoiseStep)
            Lname[iSample*Nnoise+iNoise]=name
            print(str(iSample)+"-Camera-"+str(iNoise*NoiseStep))
            o3d.io.write_point_cloud(localDir+str(iSample)+"-Camera-"+str(iNoise*NoiseStep)+'.ply', pcdCamera,write_ascii=True,print_progress=True)
            
            
            #print('display')
            #pcdCamera.paint_uniform_color([0,1,0])
            #o3d.visualization.draw_geometries([pcd,pcdCamera])
        with open(localDir+'Out_of_Camera.csv', 'w', newline='') as file:
             writer = csv.writer(file)
             Data=np.array([Lname,Lout])
             
             writer.writerows(Data)
'''
        

        
