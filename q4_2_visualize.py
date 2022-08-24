import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import camera2
from q3_2_triangulate import triangulate

from q2_1_eightpoint import eightpoint
from q3_2_triangulate import findM2
from q4_1_epipolar_correspondence import epipolarCorrespondence
from q3_1_essential_matrix import essentialMatrix
# Insert your package here


'''
Q4.2: Finding the 3D position of given points based on epipolar correspondence and triangulation
    Input:  temple_pts1, chosen points from im1
            intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
            F, the fundamental matrix
            im1, the first image
            im2, the second image
    Output: P (Nx3) the recovered 3D points
    
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's best error is around ~2000 on the 3D points. 
'''
def compute3D_pts(temple_pts1, intrinsics, F, im1, im2):

    # ----- TODO -----
    # YOUR CODE HERE
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    E = essentialMatrix(F, K1, K2)
    x2 = np.empty((x1.shape[0], 1))
    y2 = np.empty((x1.shape[0], 1))
    for i in range(x1.shape[0]):
        correspondences = epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
        x2[i] = correspondences[0]
        y2[i] = correspondences[1]
    temple_pts2 = np.hstack((x2, y2))
    M2s = camera2(E)
    C1 = np.dot(K1, M1)
    cur_err = np.inf
    for i in range(M2s.shape[2]):
        C2 = np.dot(K2, M2s[:, :, i])
        w, err = triangulate(C1, temple_pts1, C2, temple_pts2)

        if err<cur_err and np.min(w[:, 2])>=0:
            cur_err = err
            M2 = M2s[:, :, i]
            C2_opt = C2
            w_opt = w
    np.savez('q4_2', F = F, M1 = M1, M2 = M2, C1 = C1, C2 = C2)

    return w_opt



'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
if __name__ == "__main__":

    temple_coords_path = np.load('data/templeCoords.npz')
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')
    # ----- TODO -----
    # YOUR CODE HERE
    x1 = temple_coords_path['x1']
    y1 = temple_coords_path['y1']
    temple_pts1 = np.hstack((x1, y1))


    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    w_opt = compute3D_pts(temple_pts1, intrinsics, F, im1, im2)
    fig = plt.figure()
    res = Axes3D(fig)
    res.set_xlim3d(np.min(w_opt[:,0]),np.max(w_opt[:,0]))
    res.set_ylim3d(np.min(w_opt[:,1]),np.max(w_opt[:,1]))
    res.set_zlim3d(np.min(w_opt[:,2]),np.max(w_opt[:,2]))
    res.set_xlabel('X')
    res.set_ylabel('Y')
    res.set_zlabel('Z')
    res.scatter(w_opt[:,0],w_opt[:,1],w_opt[:,2])
    plt.show()





