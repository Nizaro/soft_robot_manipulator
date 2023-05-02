from plyfile import PlyData, PlyElement
import numpy as np

image=PlyData.read('data_23-05-02_13-05-03/pc0.ply')
Vert=image['vertex'].data
Face=image['face'].data

