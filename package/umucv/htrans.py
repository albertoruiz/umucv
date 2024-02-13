import numpy        as np
import numpy.linalg as la
import cv2          as cv


def vec(*argn):
    return np.array(argn)


def col(*args):
    a = args[0]
    n = len(args)
    if n==1 and type(a) == np.ndarray and len(a.shape) == 1:
        return a.reshape(-1,1)
    return np.array(args).reshape(n,1)


def row(*args):
    return col(*args).T


def jc(*args):
    return np.hstack(args)


def jr(*args):
    return np.vstack(args)


def homog(x):
    ax = np.array(x)
    uc = np.ones(ax.shape[:-1]+(1,))
    return np.append(ax,uc,axis=-1)


def inhomog(x):
    ax = np.array(x)
    return ax[..., :-1] / ax[...,[-1]]



def htrans(h,x):
    return inhomog(np.dot(homog(x) , h.T))


def kgen(sz,f):
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2,0,   w2],
                     [0,  f*w2, h2],
                     [0,   0,   1 ]])


# matriz de calibración sencilla dada la
# resolución de la imagen y el fov horizontal en grados
def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    # print(f)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])



def desp(d):
    n = len(d)
    D = np.eye(n+1)
    D[:n,-1] = d
    return D


def scale(s):
    return np.diag(np.append(s,1))


def unitary(v):
    return v / la.norm(v)


def rotation(v, a=None, homog=False):
    if a==None:
        R = cv.Rodrigues(v)[0]
    else:
        R = cv.Rodrigues(unitary(v)*a)[0]
    if homog:
        Rh = np.eye(4)
        Rh[:3,:3] = R
        return Rh
    else:
        return R


def rot3(ang):
    c = np.cos(ang)
    s = np.sin(ang)
    return np.array([[c,-s, 0]
                    ,[s, c, 0]
                    ,[0, 0, 1]])


def lookat(eye, target, up = (0, 0, 1)):
    z = np.asarray(target, np.float64) - eye  # target-eye
    z /= la.norm(z)
    x = np.cross(up, z)
    x /= la.norm(x)
    y = np.cross(z, x)
    R = np.float64([x, y, z])  # filas
    tvec = -np.dot(R, eye)
    return jc(R, col(tvec))


def lookat2(eye, target, up = (0, 0, 1)):
    z = np.asarray(target, np.float64) - eye  # target-eye
    z /= la.norm(z)
    x = np.cross(-np.array(up), z)  # "arriba" con y creciendo hacia abajo
    x /= la.norm(x)
    y = np.cross(z, x)
    R = np.float64([x, y, z])  # filas
    tvec = -np.dot(R, eye)
    return jc(R, col(tvec))



def rmsreproj(view,model,transf):
    err = view - htrans(transf,model)
    return np.sqrt(np.mean(err.flatten()**2))



class Pose:
    def __init__(self,K,image,model):
        is2D = model.shape[1] == 2
        if is2D:
            okmodel = np.hstack([model,np.zeros((len(model),1))])
        else:
            okmodel = model

        ok,rvec,tvec = cv.solvePnP(okmodel,image,K,vec(0.,0,0,0))
        if not ok:
            self.rms = 1e6
            return
        self.R,_  = cv.Rodrigues(rvec)
        self.M    = np.dot(K , jc(self.R,tvec))        
        self.view = htrans(self.M,okmodel)
        self.rms  = np.sqrt(np.mean((image - self.view).flatten()**2))
        self.t    = tvec.flatten()
        self.C    = np.dot(-self.R.T, self.t)


    #rms,M,R,t,C = ht.pose(K,image,okmodel)
    #assert(depthOfPoint(M,vec(0,0,0)>0))
    #R = M[:,:3]
    #t = np.dot(la.inv(K), M[:,3])
    #C = ht.inhomog(ht.null1(M))
    #assert C[2]<0, "bad side of the world"
    #return rms, M, R, t, htrans(M,okmodel)



def depthOfPoint(M,x):
    a = M[:,:3]
    m3 = a[2,:]
    w = np.dot(M , col(np.append(x,1)))
    w3 = w[2,0]
    return np.sign(la.det(a))/la.norm(m3)*w3


# espacio nulo de una matriz que sirve para obtener el centro
# de la cámara
def null1(M):
    u,s,vt = la.svd(M)
    return vt[-1,:]


def rq(M):
    Q,R = la.qr(np.flipud(np.fliplr(M)).T)
    R   = np.fliplr(np.flipud(R.T))
    Q   = np.fliplr(np.flipud(Q.T))
    return R,Q


# descomposición de la matriz de cámara como K,R,C
def sepcam(M):

    K,R = rq (M[:,:3])

    # para corregir los signos de f dentro de K
    s = np.diag(np.sign(np.diag(K)))

    K = np.dot(K , s)
    K = K/K[2,2]

    R = np.dot(s , R)
    R = R*np.sign(la.det(R))

    C = inhomog(null1(M))

    return K,R,C


##########################################################

def rot1b(a):
    T3 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,1]])
    P  = np.eye(4)[:3]
    return P @ desp((0,0,1)) @ rotation((1,0,0),a,homog=True) @ desp((0,0,-1)) @ T3


def tiltH(K,d,a,b=0):
    H = K @ rot3(np.radians(-d)) @ rot1b(np.radians(-a)) @ rot3(np.radians(d+b)) @ la.inv(K)
    return H


def f_from_hfov(hfov):
    return 1/np.tan(hfov/2)


def tilt(fov,d,a,b,x):
    h,w = x.shape[:2]
    K = kgen((w,h),f_from_hfov(np.radians(fov)))
    H = tiltH(K,d,a,b)
    return cv.warpPerspective(x,H,(w,h))


def untilt(fov,d,a,b,x):
    h,w = x.shape[:2]
    K = kgen((w,h),f_from_hfov(np.radians(fov)))
    H = tiltH(K,d,a,b)
    return cv.warpPerspective(x,la.inv(H),(w,h))

