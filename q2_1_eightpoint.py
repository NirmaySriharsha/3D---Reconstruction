import numpy as np
import matplotlib.pyplot as plt
#from pyrsistent import T
import helper
from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF

# Insert your package here



'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    A = np.empty((pts1.shape[0], 9))

    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    pts1 = pts1 / M
    pts2 = pts2 / M
    x_1 = pts1[:, 0]
    y_1 = pts1[:, 1]
    x_2 = pts2[:, 0]
    y_2 = pts2[:, 1]

    A = np.vstack((x_2 * x_1, x_2 * y_1, x_2, y_2 * x_1,  y_2 * y_1, y_2, x_1, y_1, np.ones(pts1.shape[0]))).T
    u, s, vh = np.linalg.svd(A)

    F = vh[-1].reshape(3, 3)
    F = helper.refineF(F, pts1, pts2)
    F = helper._singularize(F)

    F = np.dot((np.dot(T.T, F)), T)
    #return F/F[2,2]
    return F



if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    # Q2.1
    # Write your code here
    #np.savez('q2_1.npz', F = F, M = np.max([*im1.shape, *im2.shape]))
    print(F, np.linalg.matrix_rank(F))
    #helper.displayEpipolarF(im1, im2, F)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)
    #print(F, np.linalg.matrix_rank(F))
    #np.savez('q2_1.npz', F = F, M = np.max([*im1.shape, *im2.shape]))

    assert(F.shape == (3, 3))
    #assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)