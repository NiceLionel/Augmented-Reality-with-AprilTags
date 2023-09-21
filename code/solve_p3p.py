import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    p1 = Pw[0]
    p2 = Pw[1]
    p3 = Pw[2]
    Pc_ = np.hstack(((Pc[:,0] - K[0,2]).reshape(4,1),(Pc[:,1] - K[1,2]).reshape(4,1),K[0,0]*np.ones((4,1))))
    a = np.linalg.norm(p2-p3)
    b = np.linalg.norm(p1-p3)
    c = np.linalg.norm(p1-p2)
    q1 = Pc_[0]
    q2 = Pc_[1]
    q3 = Pc_[2]
    j1 = q1/np.linalg.norm(Pc_,2,axis=1)[0]
    j2 = q2/np.linalg.norm(Pc_,2,axis=1)[1]
    j3 = q3/np.linalg.norm(Pc_,2,axis=1)[2]
    #calculate the angles
    alpha = np.arccos(np.dot(j2,j3))
    beta = np.arccos(np.dot(j1,j3))
    lamb = np.arccos(np.dot(j1,j2))
    A4 = ((a**2-c**2)/b**2 - 1)**2 - (4 * c**2 / b**2) * np.cos(alpha)**2
    A3 = 4*(((a**2 - c**2)/b**2)*(1-(a**2-c**2)/b**2)*np.cos(beta) - (1-(a**2+c**2)/b**2)*np.cos(alpha)*np.cos(lamb) + 2*c**2*np.cos(alpha)**2*np.cos(beta)/b**2)
    A2 = 2*(((a**2 - c**2)/b**2)**2 - 1 + 2*(((a**2 - c**2)/b**2)**2)*np.cos(beta)**2 + 2*((b**2 - c**2)/b**2)*np.cos(alpha)**2 - 4*((a**2 + c**2)/b**2)*np.cos(alpha)*np.cos(beta)*np.cos(lamb) + 2*((b**2 - a**2)/b**2)*np.cos(lamb)**2)
    A1 = 4*(-((a**2 - c**2)/b**2)*((1+(a**2 - c**2)/b**2))*np.cos(beta) + 2*a**2*np.cos(lamb)**2*np.cos(beta)/b**2 - (1-((a**2 + c**2)/b**2))*np.cos(alpha)*np.cos(lamb))
    A0 = (1+(a**2-c**2)/b**2)**2 - 4*a**2*np.cos(lamb)**2/b**2
    coeff = [A4,A3,A2,A1,A0]
    roots = np.roots(coeff)
    real_root = np.isreal(roots)
    v_root = []
    #determine the correct roots
    for i in range(len(roots)):
        #only keep the positive and real root
        if real_root[i] > 0 and roots[i] > 0:
            v_root.append(np.real(roots[i]))
    v_root = np.array(v_root)
    
    uv_root_pair = []
    for i in range(len(v_root)):
        #calculate the corresponding u root
        u = ((-1 + (a**2-c**2)/b**2)*v_root[i]**2 - 2*((a**2-c**2)/b**2)*np.cos(beta)*v_root[i] + 1 + ((a**2-c**2)/b**2)) / (2 * (-v_root[i] * np.cos(alpha) + np.cos(lamb)))
        if u > 0:
            #pair up the u and v roots
            uv_root_pair.append([u,v_root[i]])
    
    dist_min = 20000000000 #set the initial value of the dist_min to get started
    for u,v in uv_root_pair:
        s1 = np.sqrt(c**2/(1 + u**2 - 2*u*np.cos(lamb)))
        s2 = u * s1
        s3 = v * s1
        s = np.array([s1,s2,s3])
        P_ = np.zeros((3,3))
        P_[0] = j1*s[0]
        P_[1] = j2*s[1]
        P_[2] = j3*s[2]
        R_pre,t_pre = Procrustes(P_,Pw[0:3,:])
        t_ = -R_pre.T.dot(t_pre)
        R_ = R_pre.T
        coord = R_ @ Pw[3].T + t_
        dist = np.linalg.norm(coord/coord[-1]*K[0,0] - Pc_[-1])
        #compare the distance
        if dist<dist_min:
            dist_min = dist
            R = R_pre
            t = t_pre
    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    
    X_set = X - X.mean(0)
    Y_set = Y - Y.mean(0)
    R_hat = np.matmul(Y_set.T,X_set)
    [U,S,Vt] = np.linalg.svd(R_hat)
    VUt = np.eye(3)
    VUt[-1,-1] = np.linalg.det(np.transpose(Vt) @ np.transpose(U))
    R = U @ VUt @ Vt
    t = Y.mean(0).T - R @ X.mean(0).T

    return R, t


