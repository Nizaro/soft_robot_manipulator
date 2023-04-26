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
    r0=p0+((-temp*z0+temp*z1+z0)**3-z0**3)/(3*(z1-z0))
    return z0,z1,r0

def PH3D(p0,p1,t0,L):
    
    
    #Centering
    offset=-(p1+p0)/2
    pa0=p0+offset
    pa1=p1+offset
    
    #rotation to plane XY
    A=np.linalg.norm(np.cross(pa1,t0))
    n=np.cross(pa1,t0)/(A)
    alpha=np.arccos(np.dot(n,[0,0,1]))
    
    r=np.cross(n,np.array([0,0,1]))/np.linalg.norm(np.cross(n,np.array([0,0,1]))) #seems to be the problem
    R=Rot.from_rotvec(alpha*r)
    Rinv=R.inv()
    pb0=R.apply(pa0)
    pb1=R.apply(pa1)
    tb0=R.apply(t0)
    
    
    
    #tbv=np.array([[0,0,0],tb0])
    #rv=np.array([[0,0,0],r])
    #nv=np.array([[0,0,0],n])
    
    
    #ax.scatter(pb0[0],pb0[1],pb0[2],marker="*",color=color[0],label='rotated points')
    #ax.scatter(pb1[0],pb1[1],pb1[2],marker="*",color=color[1])
    #ax.scatter(pb0[0],pa0[1],pa0[2],marker="+",color=color[0],label='Offset points')
    #ax.scatter(pb1[0],pa1[1],pa1[2],marker="+",color=color[1])
    
    #ax.plot3D(rv[:,0],rv[:,1],rv[:,2],label='Rotation axis')
    #ax.plot3D(nv[:,0],nv[:,1],nv[:,2],label='Normal to the plane')
    #ax.plot3D([0,0],[0,0],[0,1])
    
    #Formating for 2D function
    p02D=pb0[0]+pb0[1]*1j
    p12D=pb1[0]+pb1[1]*1j
    t02D=tb0[0]+tb0[1]*1j
    
    #2D calculation
    zb0,zb1,rb0=FixedLengthPH(p02D, p12D, L, t02D)
    rb0=np.transpose(np.array([np.real(rb0),np.imag(rb0),np.zeros(rb0.size)]))
    zb0=np.array([np.real(zb0**2),np.imag(zb0**2),0])
    tb1=np.array([np.real(zb1**2),np.imag(zb1**2),0])
    #rotation to initial reference
    r0=Rinv.apply(rb0)-offset
    z0=Rinv.apply(zb0)
    t1=Rinv.apply(tb1)

    return(t1,r0)
    
#Input point

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
