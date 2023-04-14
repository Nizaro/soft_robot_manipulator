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
        A=1/(8*(k**2+1))
        B1=(c**2)*(4*k+4*np.sqrt(3))**2
        B2=-4*(4*k**2+4)*((a**2)*(-3*k**2+2*np.sqrt(3)*k-1)+4*c**2)
        C=-c*k-4*np.sqrt(3)*c
        x1=A*(-np.sqrt(B1+B2)+C)
        x2=A*(np.sqrt(B1+B2)+C)
        if x1==x2:
            X=x1
        else:
            X=np.array([x1,x2])
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
    XB=PHsolveB(a,b,c,tc)
    
    X=np.append(XA,XB)
    #X=XA
    print(flip)
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
    z0=np.append(z0,(np.sqrt(2)/2)*(+(1+np.sqrt(3))*sqdeltp0-(1-np.sqrt(3))*sqdeltp1) ) 
    z1=(np.sqrt(2)/2)*(+(1-np.sqrt(3))*sqdeltp0+(1+np.sqrt(3))*sqdeltp1)
    z1=np.append(z1,(np.sqrt(2)/2)*(+(1-np.sqrt(3))*sqdeltp0-(1+np.sqrt(3))*sqdeltp1))
    
    #configuration filtering
    m=np.imag(z0**2/t)
    
    for j in range(z0.size):
        if m[j]**2==min(m**2):
            z0=z0[j]
            z1=z1[j]
    #Curve computation
    temp=np.linspace(0,1,101,dtype=complex)
    #i=0  #for debuging
    #r0=p0+((-temp*z0[i]+temp*z1[i]+z0[i])**3-z0[i]**3)/(3*(z1[i]-z0[i]))
    r0=p0+((-temp*z0+temp*z1+z0)**3-z0**3)/(3*(z1-z0))

    return z0,z1,r0

L=2
p0=0
p1=1.1j
t=1

z0,z1,r0=FixedLengthPH(p0,p1,L,t)

#Plot of the resulting curve
plt.clf()
toffsetx=np.array([np.real(p0),np.real(L*t/np.abs(t)+p0)])
toffsety=np.array([np.imag(p0),np.imag(L*t/np.abs(t)+p0)])               
plt.scatter(np.real(p1),np.imag(p1))    #draw the final point
plt.scatter(np.real(p0),np.imag(p0))    #draw the initial point
plt.plot(toffsetx,toffsety)             #draw the "relaxed" shaped
plt.plot(np.real(r0),np.imag(r0))       #plot the computed PH spline
plt.axis('equal')
plt.grid(True)


#Plot of multiple segment
plt.clf()
n=5 #nombre de section
L=2
p=np.empty(n+1,dtype=complex)
z=np.empty(2*n+1,dtype=complex)
p[0]=0
p[1]=1.96+0.35j
p[2]=3.94+0.35j
p[3]=5.25-0.95j
p[4]=3.93-2.1j
p[5]=2.55-0.95j
t=1

plt.scatter(np.real(p),np.imag(p))
plt.axis('equal')
z[0]=t
r=np.empty((101,n),dtype=complex)
ta=np.empty((2,n),dtype=complex)
color=['b','g','r','c','m','y','k']

for i in range(n):
    ta[:,i]=np.array([p[i],p[i]+L*z[2*i]**2/np.abs(z[2*i]**2)])
    #plt.plot(np.real(ta[:,i]),np.imag(ta[:,i]),color=color[i]) #affichage des segment "sans flexion"
    
    z[2*i+1],z[2*i+2],r[:,i]=FixedLengthPH(p[i],p[i+1],L,z[2*i]**2)
    plt.plot(np.real(r[:,i]),np.imag(r[:,i]),color=color[i])


    