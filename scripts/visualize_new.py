# -*- coding: utf-8 -*-
import os 
#import imageio
import numpy as np
#import mayavi.mlab
import open3d as o3d

point_cloud_path = "out\pc_original_0.npy"
cloud_im = np.load(point_cloud_path).T

# xx= cloud_im[0,:]
# yy= cloud_im[1,:]
# zz= cloud_im[2,:]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud_im) 
o3d.visualization.draw_geometries([pcd], window_name='Open3D downSample', width=1920, height=1080, left=50, top=50,
                                        point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)
# mayavi.mlab.figure(fgcolor=(0.5, 0.5, 0.5), bgcolor=(1, 1, 1))
# nodes = mayavi.mlab.points3d(-xx, -yy, zz ,mode="cube", scale_factor="10")
# nodes.glyph.scale_mode = 'scale_by_vector'

# mayavi.mlab.view(azimuth= 00, elevation=185,distance=9800,roll = None)
# mayavi.mlab.show()