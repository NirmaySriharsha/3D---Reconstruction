import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here


'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):

    A = np.empty((pts1.shape[0], 9))

    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    pts1 = pts1 / M
    pts2 = pts2 / M
    x_1 = pts1[:, 0]
    y_1 = pts1[:, 1]
    x_2 = pts2[:, 0]
    y_2 = pts2[:, 1]

    A = np.vstack((x_2 * x_1, x_2 * y_1, x_2, y_2 * x_1,  y_2 * y_1,y_2, x_1, y_1, np.ones(pts1.shape[0]))).T
    u, s, vh = np.linalg.svd(A)

    f1 = vh[-1].reshape(3, 3)
    f2 = vh[-2].reshape(3, 3)

    def dF(a): return np.linalg.det(a*f1 + (1-a)*f2)

    a0 = dF(0)
    a1 = 2*(dF(1) - dF(-1))/3 - (dF(2) - dF(-2))/12
    a2 = (dF(1) + dF(-1)) / 2 - a0
    a3 = (dF(1) + dF(-1)) / 2 - a1

    roots = np.roots([a3, a2, a1, a0])

    mat = [root*f1 + (1-root)*f2 for root in roots]
    mat = [refineF(F, pts1, pts2) for F in mat]

    Farray = [np.dot((np.dot(T.T, F)), T) for F in mat]

    return Farray



if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE
    np.random.seed(4)
    r = [56, 26, 104, 62, 53 , 92, 22] #Choices obtained using the code below
    F = sevenpoint(pts1[r, :], pts2[r, :], np.max([*im1.shape, *im2.shape]))
    #After running it I saw that there are three Fs and the third one was the best so I saved and used that.
    #np.savez('q2_2', F = F[2])
    #print(F)
    displayEpipolarF(im1, im2, F[2])
            


    """"
    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("Error:", ress[min_idx])
    print(min_idx)
    print(choices[min_idx])
    #F = F/F[2, 2]
    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)
    """