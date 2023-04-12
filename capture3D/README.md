# Allows to capture 3D data from realsense and extract calibration data
## Developped by Nizar Ouarti 

## The dependcies are the folowing, on linux use apt-get:
	cmake
	opencv
	pcl version 1.8
	eigen3
eigen is a a header library

##  Compilation
	mkdir build
	cd build
	cmake ..
	make

##  usage

	./capture3D
press 's' to save a frame (rgb and depth) and 'q' to quit
