# Developped by Nizar Ouarti
#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(capture3D)

set (prg_name "capture3D")


add_executable(${prg_name} capture3D.cpp)
set_property(TARGET ${prg_name} PROPERTY CXX_STANDARD 11)
target_link_libraries(${prg_name} 
	
    	-lopencv_core
    	-lopencv_imgcodecs
      	-lopencv_imgproc
      	-lopencv_highgui 
	-lopencv_video
	-lopencv_videoio
  	-lrealsense2
  	-lboost_system
  	-lpcl_io    	
      )
    
INCLUDE_DIRECTORIES(
  /usr/include/pcl-1.12/
  /usr/include/eigen3
  /usr/local/include/opencv4/ 
  )

    
set_target_properties(${prg_name} PROPERTIES
    COMPILE_FLAGS "-O3 -Wno-deprecated -std=c++14 `pkg-config --libs opencv` "
    )

