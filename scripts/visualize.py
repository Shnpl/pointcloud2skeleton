import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')


import os
def draw_line_between_joints(ax,start_id,end_id,data,color):
    start_point = data[start_id-1,:]
    end_point = data[end_id-1,:]
    x = [start_point[0],end_point[0]]
    y = [start_point[1],end_point[1]]
    z = [start_point[2],end_point[2]]
    ax.plot(x,y,z,c=color)
def draw_skeleton(ax,sk_data,color):
    draw_line_between_joints(ax,1,2,sk_data,color)
    draw_line_between_joints(ax,2,21,sk_data,color)
    draw_line_between_joints(ax,21,3,sk_data,color)
    draw_line_between_joints(ax,3,4,sk_data,color)

    draw_line_between_joints(ax,21,5,sk_data,color)
    draw_line_between_joints(ax,5,6,sk_data,color)
    draw_line_between_joints(ax,6,7,sk_data,color)
    draw_line_between_joints(ax,7,8,sk_data,color)
    #draw_line_between_joints(ax,8,22,sk_data,color)
    #draw_line_between_joints(ax,8,23,sk_data,color)

    draw_line_between_joints(ax,21,9,sk_data,color)
    draw_line_between_joints(ax,9,10,sk_data,color)
    draw_line_between_joints(ax,10,11,sk_data,color)
    draw_line_between_joints(ax,11,12,sk_data,color)
    #draw_line_between_joints(ax,12,25,sk_data,color)
    #draw_line_between_joints(ax,12,24,sk_data,color)

    draw_line_between_joints(ax,1,13,sk_data,color)
    draw_line_between_joints(ax,13,14,sk_data,color)
    draw_line_between_joints(ax,14,15,sk_data,color)
    draw_line_between_joints(ax,15,16,sk_data,color)

    draw_line_between_joints(ax,1,17,sk_data,color)
    draw_line_between_joints(ax,17,18,sk_data,color)
    draw_line_between_joints(ax,18,19,sk_data,color)
    draw_line_between_joints(ax,19,20,sk_data,color)


def visualize(np_pred_sk_data,np_gt_sk_data,pc_original = None):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    ax.set_aspect('equal')

    x_gt_skeleton = np_gt_sk_data[:,0]
    y_gt_skeleton = np_gt_sk_data[:,1]
    z_gt_skeleton = np_gt_sk_data[:,2]


    x_pred_skeleton = np_pred_sk_data[:,0]
    y_pred_skeleton = np_pred_sk_data[:,1]
    z_pred_skeleton = np_pred_sk_data[:,2]

    # x_pc_calibrated = pc_calibrated[:,0]
    # y_pc_calibrated = pc_calibrated[:,1]
    # z_pc_calibrated = pc_calibrated[:,2]


    x_pc_original = pc_original[:,0]
    y_pc_original = pc_original[:,1]
    z_pc_original = pc_original[:,2]

    # scale_x = max(x_pc_original)-min(x_pc_original)
    # scale_y = max(y_pc_original)-min(y_pc_original)
    # scale_z = max(z_pc_original)-min(z_pc_original)
    # scale = 1/max(scale_x,scale_y,scale_z)

    # x_pc_original = x_pc_original*scale
    # y_pc_original = y_pc_original*scale
    # z_pc_original = z_pc_original*scale

    ax.view_init(elev=180, azim=-90)

    #gt
    ax.scatter(x_gt_skeleton, y_gt_skeleton, z_gt_skeleton,'blue',label='GT')
    for i in range(21):
        ax.text(x_gt_skeleton[i],y_gt_skeleton[i],z_gt_skeleton[i],i,size=5,color='blue')
    draw_skeleton(ax,np_gt_sk_data,'blue')
    #pred
    ax.scatter(x_pred_skeleton, y_pred_skeleton, z_pred_skeleton,'red',label='PRED')
    for i in range(21):
        ax.text(x_pred_skeleton[i],y_pred_skeleton[i],z_pred_skeleton[i],i,size=5,color = 'red')
    draw_skeleton(ax,np_pred_sk_data,'red')
    #
    # ax.scatter(x_pc_calibrated, y_pc_calibrated, z_pc_calibrated,color = 'yellow')
    ax.scatter(x_pc_original, y_pc_original, z_pc_original,color = 'cyan',alpha=0.3)

    plt.show()

if __name__ == "__main__":
    # for i in range(1):
    i = 0
    pred_path = f"out/pred_{i}.npy"
    # gt_path = f"out/gt_{i}.npy"
    # pc_calibrated = f"out/pc_calibrated_{i}.npy"
    pc_original = f"out/pc_original_{i}.npy"
    np_pred_sk_data = np.load(pred_path)
    # np_gt_sk_data = np.load(gt_path)
    # pc_calibrated = np.load(pc_calibrated).T
    pc_original = np.load(pc_original).T