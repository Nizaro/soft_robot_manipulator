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
import timeit

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


start = timeit.default_timer()
#Input
#p0 is the starting point of the desired spline
#p1 is th end point of the desired spline
#t represent the direction of the tangent at p1 (tangent at p1 is undefined as cubic PH curve do not have enough degree of freedom)
#L is the desired length of the curve
# /!\  L should not be shorter than |p1-p0|
L=2
p0=0
p1=1.95+0.4j
t=1


#display of the input data
plt.clf()
toffsetx=np.array([np.real(p0),np.real(L*t/np.abs(t)+p0)])
toffsety=np.array([np.imag(p0),np.imag(L*t/np.abs(t)+p0)])               
plt.scatter(np.real(p1),np.imag(p1))
plt.scatter(np.real(p0),np.imag(p0))
plt.plot(toffsetx,toffsety)


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

#Computation of the rectifying control polygon admiting desired length and tangent
XA=PHsolveA(a,b,c,tc)
XB=PHsolveB(a,b,c,tc)

#X=np.append(XA,XB)
X=XA
Y=np.empty(X.size)
Y=b*np.sqrt(1-(X**2/a**2))

Pc=np.empty(2*X.size,dtype=complex)
for i in range(X.size):
    Pc[2*i]=X[i]+Y[i]*1j
    Pc[2*i+1]=X[i]-Y[i]*1j

Pc=np.append(Pc,-Pc)

#Coordinate change to initial space
Inv=np.cos(theta)+np.sin(theta)*1j
P=Inv*Pc+B
           

#Bezier control point calculation
sqdeltp0=np.sqrt(P-p0)
sqdeltp1=np.sqrt(p1-P)

z0=(np.sqrt(2)/2)*(+(1+np.sqrt(3))*sqdeltp0+(1-np.sqrt(3))*sqdeltp1) 
z0=np.append(z0,(np.sqrt(2)/2)*(+(1+np.sqrt(3))*sqdeltp0-(1-np.sqrt(3))*sqdeltp1) ) 
z1=(np.sqrt(2)/2)*(+(1-np.sqrt(3))*sqdeltp0+(1+np.sqrt(3))*sqdeltp1)
z1=np.append(z1,(np.sqrt(2)/2)*(+(1-np.sqrt(3))*sqdeltp0-(1+np.sqrt(3))*sqdeltp1))

#configuration filtering
m=np.angle(z0**2/t)

Bad=np.array([-1])
for j in range(z0.size):
    if m[j]>0.01 or m[j]<-0.01 :
        Bad=np.append(Bad,[j])
Bad=np.delete(Bad,0)         
z0=np.delete(z0,Bad)
z1=np.delete(z1,Bad)

#Curve computation
temp=np.linspace(0,1,101,dtype=complex)
i=0
r0=p0+((-temp*z0[i]+temp*z1[i]+z0[i])**3-z0[i]**3)/(3*(z1[i]-z0[i]))

#Plot of the reulsting curve
plt.plot(np.real(r0),np.imag(r0))
plt.axis('equal')
plt.grid(True)

stop = timeit.default_timer()

print('Time: ', stop - start)