import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    
    t = -R_wc.T.dot(t_wc)
    R = R_wc.T
    intermid = np.matmul(K, np.hstack((R[:,0].reshape(3,1), R[:,1].reshape(3,1), t.reshape(3,1))))
    pc = np.hstack((pixels, np.ones(pixels.shape[0]).reshape(-1,1)))
    P = np.linalg.inv(intermid) @ pc.T
    Pw = np.zeros((pixels.shape[0],3))
    for i in range(len(pixels)):
        pc_ = P[:,i]
        pw_1 = pc_[1]/pc_[2]
        pw_2 = pc_[0]/pc_[2]
        Pw[i] = np.array([pw_2, pw_1,0])
    return Pw
