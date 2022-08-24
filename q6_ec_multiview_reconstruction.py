import numpy as np
import matplotlib.pyplot as plt

import os

from helper import visualize_keypoints, plot_3d_keypoint, connections_3d, colors
from q3_2_triangulate import triangulate
from q5_bundle_adjustment import bundleAdjustment
# Insert your package here

'''
Q6.1 Multi-View Reconstruction of keypoints.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx3 matrix with the 2D image coordinates and confidence per row
            C2, the 3x4 camera matrix
            pts2, the Nx3 matrix with the 2D image coordinates and confidence per row
            C3, the 3x4 camera matrix
            pts3, the Nx3 matrix with the 2D image coordinates and confidence per row
    Output: P, the Nx3 matrix with the corresponding 3D points for each keypoint per row
            err, the reprojection error.
'''
def MultiviewReconstruction(C1, pts1, C2, pts2, C3, pts3, Thres = 100):
    # Replace pass by your implementation
    P, err = triangulate(C1, pts1[:, :2], C2, pts2[:, :2])
    return P, err


'''
Q6.2 Plot Spatio-temporal (3D) keypoints
    :param car_points: np.array points * 3
'''
def plot_3d_keypoint_video(pts_3d_video):
    # Replace pass by your implementation
    pass


#Extra Credit
if __name__ == "__main__":
         
        
    pts_3d_video = []
    for loop in range(10):
        print(f"processing time frame - {loop}")

        data_path = os.path.join('data/q6/','time'+str(loop)+'.npz')
        image1_path = os.path.join('data/q6/','cam1_time'+str(loop)+'.jpg')
        image2_path = os.path.join('data/q6/','cam2_time'+str(loop)+'.jpg')
        image3_path = os.path.join('data/q6/','cam3_time'+str(loop)+'.jpg')

        im1 = plt.imread(image1_path)
        im2 = plt.imread(image2_path)
        im3 = plt.imread(image3_path)

        data = np.load(data_path)
        pts1 = data['pts1']
        pts2 = data['pts2']
        pts3 = data['pts3']

        K1 = data['K1']
        K2 = data['K2']
        K3 = data['K3']

        M1 = data['M1']
        M2 = data['M2']
        M3 = data['M3']

        #Note - Press 'Escape' key to exit img preview and loop further 
        img = visualize_keypoints(im2, pts2)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # YOUR CODE HERE
        for i in range(10):
            time = np.load('./data/q6/time'+str(i)+'.npz')
            pts1 = time['pts1']
            pts2 = time['pts2']
            pts3 = time['pts3']
            M1_0 = time['M1']
            M2_0 = time['M2']
            M3_0 = time['M3']
            K1_0 = time['K1']
            K2_0 = time['K2']
            K3_0 = time['K3']
            C1_0 = np.dot(K1_0, M1_0)
            C2_0 = np.dot(K1_0, M2_0)
            C3_0 = np.dot(K1_0, M3_0)
            Thres = 200
            P_mv, err_mv = MultiviewReconstruction(
                C1_0, pts1, C2_0, pts2, C3_0, pts3, Thres)
            M2_opt, pts_3d, _ , _ = bundleAdjustment(
                K2_0, M2_0, pts2[:, :2], K3_0, M3_0, pts3[:, :2], P_mv)
            num_points = pts_3d.shape[0]
            for j in range(len(connections_3d)):
                index0, index1 = connections_3d[j]
                xline = [pts_3d[index0, 0], pts_3d[index1, 0]]
                yline = [pts_3d[index0, 1], pts_3d[index1, 1]]
                zline = [pts_3d[index0, 2], pts_3d[index1, 2]]
                ax.plot(xline, yline, zline, color=colors[j])
            np.set_printoptions(threshold=1e6, suppress=True)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    plot_3d_keypoint(P_mv)
    np.savez('q6_1.npz', M=M2_opt, w=P_mv)
    img = plt.imread('./data/q6/cam3_time0.jpg')
    visualize_keypoints(img, pts3, Thres)