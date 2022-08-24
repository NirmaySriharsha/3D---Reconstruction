import numpy as np
import matplotlib.pyplot as plt

from helper import _epipoles

from q2_1_eightpoint import eightpoint

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        # x, y = plt.ginput(1, mouse_stop=2)[0]

        out = plt.ginput(1, timeout=3600, mouse_stop=2)

        if len(out) == 0:
            print(f"Closing GUI")
            break
        
        x, y = out[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            print('Zero line vector in displayEpipolar')

        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', MarkerSize=8, linewidth=2)
        plt.draw()


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2
            
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    x1 = int(x1)
    y1 = int(y1)
    rect_size = 20  # window size to adjust
    im1sect = im1[(y1 - rect_size//2): (y1 + rect_size//2 + 1),
                   (x1 - rect_size//2): (x1 + rect_size//2 + 1), :]

    im2_h, im2_w, _ = im2.shape

    p1 = np.array([x1, y1, 1])

    epi_line = np.dot(F, p1)
    epi_l = epi_line / np.linalg.norm(epi_line)
    a, b, c = epi_l

    ep2_y = np.arange(im2_h)
    ep2_x = np.rint(-(epi_l[1]*ep2_y + epi_l[2]) / epi_l[0])

    # Do the gaussian thingy
    r_v = np.arange(-rect_size//2, rect_size//2+1, 1)
    r_x, r_y = np.meshgrid(r_v, r_v)
    d = 7
    weight = np.sum(np.dot((np.exp(-((r_x**2 + r_y**2) / (2 * (d**2))))), 1)/ np.sqrt(2*np.pi*d**2))
    tol = 1e5

    for y2 in range((y1 - rect_size//2), (y1 + rect_size//2 + 1)):
        x2 = int((-b*y2-c)/a)
        if (x2 >= rect_size//2 and x2 + rect_size//2 < im2_w and y2 >= rect_size//2 and y2 + rect_size//2 < im2_h):
            im2_sect = im2[y2-rect_size//2:y2+rect_size //2 +1, x2-rect_size//2:x2+rect_size//2 +1, :]
            err = np.linalg.norm((im1sect - im2_sect) * weight)
            if err < tol:
                tol = err
                x2_opt = x2
                y2_opt = y2
    return x2_opt, y2_opt




if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')


    # ----- TODO -----
    # YOUR CODE HERE
    
    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))
    np.savez('q4_1', F = F, pts1 = pts1, pts2 = pts2)
    
    # Simple Tests to verify your implementation:
    x2, y2 = epipolarCorrespondence(im1, im2, F, 119, 217)
    epipolarMatchGUI(im1, im2, F)
    print(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])))
    #assert(np.linalg.norm(np.array([x2, y2]) - np.array([118, 181])) < 10)