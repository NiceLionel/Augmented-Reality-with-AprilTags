from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    Pw = Pw[:,0:-1]
    H = est_homography(Pw,Pc)
    H = H/H[2,2]
    K_inverse = np.linalg.inv(K)
    KH = K_inverse @ H
    h1 = KH[:,0]
    h2 = KH[:,1]
    h3 = KH[:,2]
    USV = np.concatenate(([h1],[h2],[np.cross(h1,h2)]),axis=0)
    USV = np.transpose(USV)
    [U, S, Vt] = np.linalg.svd(USV)
    UVt = np.eye(3)
    UVt[-1,-1] = np.linalg.det(U @ Vt)
    R = U @ UVt @ Vt
    t = h3/np.linalg.norm(h1)
    R = np.linalg.inv(R)
    t = np.transpose(-R @ t)
    return R, t
