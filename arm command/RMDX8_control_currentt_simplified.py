## Code modified from a template for PCAN protocol with RMDX motor
## Author of the original template Guillaume SAMAIN
# Needed Imports
from PCANBasic import *
from Biblio.Bib import Initialisation
from Biblio.Bib import Moteur
import os
import sys
import time
import numpy as np
import random

#VERSION QUI FONCTIONNE avec CSV et courant filtré
Init = Initialisation()

N=10
Theta=[]
Phi=[]

# Definition of trajectory in spherical coordinate


'''
for i in range(12):
    Theta.append([30-5*i,150])
    Phi.append([270,180])

for i in range(13):
    Theta.append([0,5*(i)])
    Phi.append([90,0])
'''


Theta.append([0,0])
Phi.append([0,0])

for i in range(10):
    Theta.append([random.random()*60,random.random()*90])
    Phi.append([random.random()*360,random.random()*360])
print(Theta)
print(Phi)


Trajectoire=[]
Theta.append([0,0])
Phi.append([0,0])
D=4.22/3.05

# Transformation of spherical coordinate to motor angular position according to PCC Model 
for i in range(len(Theta)):
    Trajectoire.append([-D*Theta[i][1]*np.cos(Phi[i][1]*np.pi/180)+D*Theta[i][0]*np.cos(Phi[i][0]*np.pi/180)/3,-D*Theta[i][1]*np.sin(Phi[i][1]*np.pi/180)+D*Theta[i][0]*np.sin(Phi[i][0]*np.pi/180)/3,D*Theta[i][0]*np.cos(Phi[i][0]*np.pi/180),D*Theta[i][0]*np.sin(Phi[i][0]*np.pi/180)])
print(Trajectoire)

# Trajectory save
file= open("Random_trajectory_2cam.txt","a")
file.write("Theta1;Theta2;Phi1;Phi2;Delta1;Delta2;Delta3;Delta4")
for i in range(len(Theta)):
    file.write("\n"+str(Theta[i][0])+";"+str(Theta[i][1])+";"+str(Phi[i][0])+";"+str(Phi[i][1])+";"+str(Trajectoire[i][2])+";"+str(Trajectoire[i][3]))
file.close()


# Command of motors
ListeMoteurs = [[0x141,-36],[0x142,-36],[0x143,6],[0x144,6]] #[ID,GearRatio]
Moteur1 = Moteur(ListeMoteurs,Trajectoire,Theta,Phi)

