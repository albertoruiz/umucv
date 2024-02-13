import numpy        as np
import cv2          as cv
from umucv.htrans import vec, scale, desp, htrans, rot3
import numpy.linalg as la

def orientation(x):
    return cv.contourArea(x.astype(np.float32),oriented=True)

def fixOrientation(x):
    if orientation(x) >= 0:
        return x
    else:
        return np.flipud(x)

def redondez(c):
    p = cv.arcLength(c.astype(np.float32),closed=True)
    oa = orientation(c)
    if p>0:
        return oa, 100*4*np.pi*abs(oa)/p**2
    else:
        return 0,0

def boundingBox(c):
    (x1, y1), (x2, y2) = c.min(0), c.max(0)
    return (x1, y1), (x2, y2)

def internal(c,h,w):
    (x1, y1), (x2, y2) = boundingBox(c)
    return x1>1 and x2 < w-2 and y1 > 1 and y2 < h-2

def redu(c,eps=0.5):
    red = cv.approxPolyDP(c,eps,True)
    n = len(red)
    return red.reshape(n,2)


def extractContours(g, minarea=10, minredon=25, reduprec=1, approx=True):
    #gt = cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,101,-10)
    ret, gt = cv.threshold(g,189,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    app_mode = cv.CHAIN_APPROX_SIMPLE if approx else cv.CHAIN_APPROX_NONE
    
    # gt.copy() not needed
    # [-2] seems to work in different opencv versions
    contours = cv.findContours(gt, cv.RETR_TREE, app_mode)[-2]

    if approx:
        contours = [ redu(c.reshape(-1,2), reduprec) for c in contours ]
    else:
        contours = [ c.reshape(-1,2) for c in contours ]

    h,w = g.shape
    
    tharea = (min(h,w)*minarea/100.)**2 
    
    def good(c):
        oa,r = redondez(c)
        black = oa > 0 # and positive orientation
        return black and abs(oa) >= tharea and r > minredon

    ok = [c for c in contours if good(c) and internal(c,h,w) ]
    return ok


def eig22(cxx,cyy,cxy):
    l,v = la.eigh(np.array([[cxx,cxy],[cxy,cyy]]))
    return l[1], l[0], np.arctan2(v[1][1],v[1][0])


def eig22(cxx,cyy,cxy):
    ra = np.sqrt(abs ( cxx*cxx + 4*cxy*cxy -2*cxx*cyy + cyy*cyy))
    l1 = max(0.5*(cxx+cyy+ra), 0)
    l2 = max(0.5*(cxx+cyy-ra), 0)
    a = np.arctan2 (2*cxy,(cxx-cyy+ra))
    eps = 1E-12
    if (abs(cxy) < eps and cyy > cxx):
        ap = np.pi/2
    else:
        ap = a
    return (l1,l2,ap)





def mymoments(c):
    cf = c.astype(float)
    cr = np.roll(cf,-1,0)
    x1 = cf[:,0]; y1 = cf[:,1]
    x2 = cr[:,0]; y2 = cr[:,1]
    dx = x2-x1; dy = y2-y1
    x12 = x1**2; x22 = x2**2
    y12 = y1**2; y22 = y2**2
    q1 = 2*y1 + y2; q2 = y1 + 2*y2

    s0  = x1*y2-x2*y1
    sx  = np.sum(2*x1*x2*dy-x22*q1+x12*q2)/12
    sy  = np.sum(-2*y1*y2*dx+y22*(2*x1+x2)-y12*(2*x2+x1))/12
    sx2 = np.sum( (x12*x2+x1*x22)*dy + (x1**3-x2**3)*(y1+y2))/12
    sy2 = np.sum(-(y12*y2+y1*y22)*dx - (y1**3-y2**3)*(x1+x2))/12
    sxy = np.sum(s0*(x1*q1+x2*q2))/24
    s   = np.sum(s0)/2
    
    mx  = sx / s
    my  = sy / s
    cx  = sx2 / s - mx**2
    cy  = sy2 / s - my**2
    cxy = sxy / s - mx*my
    
    (l1,l2,ad) = eig22(cx,cy,cxy)
    return (vec(mx,my),(np.sqrt(l1),np.sqrt(l2),ad))


def mymoments(c):
    m = cv.moments(c.astype(np.float32))  # int32, float32, but not float64!
    s = m['m00']
    mx = m['m10']/s
    my = m['m01']/s
    cx = m['mu20']/s
    cy = m['mu02']/s
    cxy = m['mu11']/s
    (l1,l2,ad) = eig22(cx,cy,cxy)
    return (vec(mx,my),(np.sqrt(l1),np.sqrt(l2),ad))


def center(x):
    return mymoments(x)[0]


def whitener(cont):
    (c,(s1,s2,a)) = mymoments(cont)
    return np.dot(np.dot(rot3(a) , scale(1/vec(s1,s2))) , np.dot(rot3(-a) , desp(-c)))

def whiten(c):
    return htrans(whitener(c),c)


def autoscale(cont):
    (x1, y1), (x2, y2) = cont.min(0), cont.max(0)
    s = max(x2-x1,y2-y1)
    c = vec(x1+x2,y1+y2)/2
    h = np.dot(scale(1/vec(s,s)) , desp(-c))
    return htrans(h,cont)


def fourierPL(x):
    z = x[:,0] + x[:,1]*1j
    z = np.append(z,z[0])
    dz = z-np.roll(z,1)
    dta = abs(dz)
    t = np.cumsum(dta)
    tot = t[-1]
    t = t / tot
    dt = dta / tot
    alpha = dz[1:]/dt[1:]
    A = alpha - np.roll(alpha,-1)
    H = np.exp(-2*np.pi*1j*t[1:])
    mz = z+np.roll(z,-1)
    f0 = np.sum (mz[:-1]*dt[1:]) / 2
    def g(w):
        if w == 0: return f0
        return np.dot(H**w,A) / (2*np.pi * w)**2
    return g




def moments_2(c):
    m = cv.moments(c.astype(np.float32))  # int32, float32, but not float64!
    s = m['m00']
    return (m['m10']/s, m['m01']/s, m['mu20']/s, m['mu02']/s, m['mu11']/s)


def errorEllipse(e,c, mindiam = 20, minratio = 0.2):
    (cx,cy), (A,B), ang = e
    if A < mindiam: return 2000
    if B/A < minratio: return 1000
    T = la.inv(desp((cx,cy)) @ rot3(np.radians(ang)) @ scale((A/2,B/2)))
    cc = htrans(T,c)
    r = np.sqrt((cc*cc).sum(axis=1))
    return abs(r-1).max()


def detectEllipses(contours, mindiam = 20, minratio = 0.2, tol = 5):
    res = []
    for c in contours:
        mx,my,cxx,cyy,cxy = moments_2(c)
        v1,v2,a = eig22(cxx,cyy,cxy)
        p = mx,my
        s1, s2 = np.sqrt(v1), np.sqrt(v2)
        e = p, (4*s1,4*s2), a/np.pi*180
        err = errorEllipse(e, c, mindiam=mindiam, minratio=minratio)
        if err < tol/100:
            res.append(e)
    return sorted(res, key = lambda x: - x[1][0] * x[1][1])

