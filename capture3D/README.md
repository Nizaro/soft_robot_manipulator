# Allows to capture 3D data from realsense and extract calibration data
## Developped by Nizar Ouarti 

## The dependcies are the folowing, on linux use apt-get:
	cmake 
		linux: sudo apt-get install cmake
	opencv
		linux: script d'intallation: https://github.com/milq/milq/blob/master/scripts/bash/install-opencv.sh
	pcl version >= 1.8  
		linux: sudo apt install libpcl-dev
	eigen3
		eigen is a a header library: download the .h

##  Compilation
	mkdir build
	cd build
	cmake ..
	make

##  usage

	./capture3D
press 's' to save a frame (rgb and depth) and 'q' to quit
