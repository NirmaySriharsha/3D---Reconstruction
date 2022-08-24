import numpy as np
import matplotlib.pyplot as plt
from q4_1_epipolar_correspondence import epipolarMatchGUI

from helper import displayEpipolarF, calc_epi_error, toHomogenous, camera2
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2, triangulate
from q3_1_essential_matrix import essentialMatrix
import scipy
from mpl_toolkits.mplot3d import Axes3D
# Insert your package here
import sys
from helper import refineF

# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    max_inliers = -1

    p1_hom = np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))
    p2_hom = np.vstack((pts2.T, np.ones((1, pts1.shape[0]))))

    for idx in range(nIters):
        total_inliers = 0
        rand_idx = np.random.choice(pts1.shape[0], 8)
        rand1 = pts1[rand_idx, :]
        rand2 = pts2[rand_idx, :]

        F = eightpoint(rand1, rand2, M)
        pred_x2 = np.dot(F, p1_hom)
        pred_x2 = pred_x2 / np.sqrt(np.sum(pred_x2[:2, :]**2, axis=0))

        err = abs(np.sum(p2_hom*pred_x2, axis=0))
        n_inliers = err < tol
        # print(n_inliers)
        total_inliers = n_inliers[n_inliers.T].shape[0]
        if total_inliers > max_inliers:
            F_opt = F
            max_inliers = total_inliers
            inliers = n_inliers
        print(idx)

    return F_opt, inliers



'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    d = r.shape[0]
    theta = np.linalg.norm(r)
    if theta == 0:
        R = np.identity(d)
    else:
        u = r/theta
        u1 = u[0]
        u2 = u[1]
        u3 = u[2]

        u_x = np.array([[0,-u3,u2],[u3,0,-u1],[-u2,u1,0]])
        R = np.identity(d)*np.cos(theta) + (1-np.cos(theta))*np.matmul(u,u.T) + np.sin(theta)*u_x

    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''


def s_half(r):
    theta = np.linalg.norm(r)

    if theta == np.pi and ( (r[2]< 0 and r[0] == r[1] and r[0] == 0 and r[1] == 0) or  (r[0] == 0 and r[1] < 0) or (r[0] < 0) ):
        return -1*r
    return r

def arctan2(y,x):
    if x > 0:
        return np.arctan(y/x)
    elif x<0:
        return np.pi + np.arctan(y/x)
    elif x == 0 and y < 0:
        return -np.pi/2
    elif x ==0 and y > 0:
        return np.pi/2

def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - R.T)/2
    s = np.linalg.norm(np.array([A[2,1], A[0,2], A[1,0]]))
    c = (R[0,0]+R[1,1]+R[2,2]-1)/2
    r = []
    if s == 0 and c == 1:
        r = np.zeros((3,1))
    elif s == 0 and c == -1:
        Z = np.add(R, np.identity(3))
        r_1 = Z[:,0]
        r_2 = Z[:,1]
        r_3 = Z[:,2]
        if len(np.nonzero(r_1)) > 0:
            v = r_1
        elif len(np.nonzero(r_2)) > 0:
            v = r_2
        elif len(np.nonzero(r_3)) > 0:
            v = r_3
        u = v/np.linalg.norm(v)
        r_hat = u*np.pi
        r = s_half(r_hat)
    elif s != 0:
        u = np.array([A[2,1], A[0,2], A[1,0]])/s
        theta = arctan2(s,c)
        r = u*theta

    return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    C1 = np.dot(K1, M1)
    P = x[:-6].reshape(-1, 3)
    r2 = x[-6:-3].reshape(3, 1)
    t2 = x[-3:].reshape(3, 1)

    R2 = rodrigues(r2)
    M2 = np.hstack((R2, t2)).reshape(3, 4)  # Extrinsics of camera 2
    C2 = np.dot(K2, M2)
    P_hom = np.vstack((P.T, np.ones((1, P.shape[0]))))

    p1_hat = np.zeros((2, P_hom.shape[1]))
    p2_hat = np.zeros((2, P_hom.shape[1]))

    x1_hom = np.dot(C1, P_hom)
    x2_hom = np.dot(C2, P_hom)

    p1_hat[0, :] = (x1_hom[0, :] / x1_hom[2, :])
    p1_hat[1, :] = (x1_hom[1, :]/x1_hom[2, :])
    p2_hat[0, :] = (x2_hom[0, :]/x2_hom[2, :])
    p2_hat[1, :] = (x2_hom[1, :] / x2_hom[2, :])
    p1_hat = p1_hat.T
    p2_hat = p2_hat.T

    residuals = np.concatenate(
        [(p1 - p1_hat).reshape(-1), (p2 - p2_hat).reshape(-1)])
    return residuals





'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    

    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    R2_0 = M2_init[:, 0:3]
    t2_0 = M2_init[:, 3]
    r2_0 = invRodrigues(R2_0)
    def fun(M): return (rodriguesResidual(K1, M1, p1, K2, p2, M))
    #obj_start = rodriguesResidual(K1, M1, p1, K2, p2, M2_init.flatten())
    x_0 = P_init.flatten()
    x_0 = np.append(x_0, r2_0.flatten())
    x_0 = np.append(x_0, t2_0.flatten())

    x_opt, _ = scipy.optimize.leastsq(fun, x_0)
    Pnew = x_opt[0:-6].reshape(-1, 3)
    rnew = x_opt[-6:-3].reshape(3, 1)
    tnew = x_opt[-3:].reshape(3, 1)

    R2 = rodrigues(rnew)
    M2 = np.hstack((R2, tnew))
    #obj_end = rodriguesResidual(K1, M2, p1, K2, p2, M2.flatten())
    return M2, Pnew, obj_start, obj_end



if __name__ == "__main__":
              
    #np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')
    M=np.max([*im1.shape, *im2.shape])

    FRansac, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    F = eightpoint(noisy_pts1, noisy_pts2, M)
    #epipolarMatchGUI(im1, im2, FRansac)
    #displayEpipolarF(im1, im2, FRansac)
    #displayEpipolarF(im1, im2, F)
    # YOUR CODE HERE
    print(inliers)
    print(np.count_nonzero(inliers)/len(inliers))
    print(np.count_nonzero(inliers))
    print(len(inliers))

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    #assert(F.shape == (3, 3))
    #assert(F[2, 2] == 1)
    #assert(np.linalg.matrix_rank(F) == 2)
    

    # YOUR CODE HERE
    


    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    #rotVec = sRot.random()
    #mat = rodrigues(rotVec.as_rotvec())
    #print(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)))
    #assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    #assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)
    E = essentialMatrix(F, K1, K2)
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    M2_all = camera2(E)

    C1 = np.dot(K1, M1)
    err_min = np.inf

    for i in range(M2_all.shape[2]):
         M2_i = M2_all[:, :, i]
         C2 = np.dot(K2, M2_i)
         w, err = triangulate(C1, noisy_pts1, C2, noisy_pts2)

         if err < err_min:
             err_val = err
             M2 = M2_i
             C2_opt = C2
             w_best = w

    P_init, err_orig = triangulate(C1, noisy_pts1, C2_opt, noisy_pts2)
    print('Original reprojection error: ', err_orig)

    
    #5.3
    M2_opt, P2, _, _ = bundleAdjustment(K1, M1, noisy_pts1, K2, M2, noisy_pts2, P_init)

    C2_opt = np.dot(K2, M2_opt)
    w_hom = np.hstack((P2, np.ones([P2.shape[0], 1])))
    C2 = np.dot(K2, M2)
    err_opt = 0

    for i in range(noisy_pts1[inliers, :].shape[0]):
        pts1hat = np.dot(C1, w_hom[i, :].T)
        pts2hat = np.dot(C2_opt, w_hom[i, :].T)

        # Normalizing
        p1_hat_norm = (np.divide(pts1hat[0:2], pts1hat[2])).T
        p2_hat_norm = (np.divide(pts2hat[0:2], pts2hat[2])).T
        err1 = np.square(noisy_pts1[:, 0] - p1_hat_norm[0]) + \
            np.square(noisy_pts1[:, 1] - p1_hat_norm[0])
        err2 = np.square(noisy_pts2[:, 0] - p2_hat_norm[0]) + \
             np.square(noisy_pts2[:, 1] - p2_hat_norm[0])
        err_opt += np.sum((p1_hat_norm - noisy_pts1[i])
                          ** 2 + (p2_hat_norm - noisy_pts2[i])**2)

    print('Error with optimized 3D points: ', err_opt)

    plot_3D_dual(P_init, P2)

    fig1 = plt.figure()
    ax1 = Axes3D(fig1)
    ax1.set_xlim3d(np.min(P_init[:, 0]), np.max(P_init[:, 0]))
    ax1.set_ylim3d(np.min(P_init[:, 1]), np.max(P_init[:, 1]))
    ax1.set_zlim3d(np.min(P_init[:, 2]), np.max(P_init[:, 2]))
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.scatter(P_init[:, 0], P_init[:, 1], P_init[:, 2])
    plt.show()

    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    ax2.set_xlim3d(np.min(P2[:, 0]), np.max(P2[:, 0]))
    ax2.set_ylim3d(np.min(P2[:, 1]), np.max(P2[:, 1]))
    ax2.set_zlim3d(np.min(P2[:, 2]), np.max(P2[:, 2]))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.scatter(P2[:, 0], P2[:, 1], P2[:, 2])
    plt.show()




    # YOUR CODE HERE