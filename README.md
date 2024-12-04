# soft_robot_manipulator
3D perception for the control of a soft robot manipulator

The main concept is to reconstruct in 3D a soft robot manipulator with partial occlusions.
Our approach is based on computer vision and utilize RGBD camera. 
We use multiple pass of RANSAC algorithm to detect the shape of our manipulator in poinclouds.
We used the PCC modelisation of soft robotic arm in our algorithm.
This repository also includes tests for using pythogorean odograph curves as a model for the manipulator.


You will find the two libraries we developed in the 3D analysis folder ;
- Arm_Gen.py is focused on tools for simulating our manipulator and RGBD camera
- Filtering3D.py contain all the tools used for the manipulator reconstruction and pose estimation

As an example, you will find the script 3D analysis/Test_PCCreconstruction_Multisegment.py 
You will need to bring your own pointcloud in ply format and manually change ligne 19
You then need to change the parameter to match your own arm at line 44,45,46,52,54,55




