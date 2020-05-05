#!/usr/bin/env python

# Detección de elipses (segundo método)
# pruébalo con:

# ./elipses0a.py --dev=dir:*.png


# paquetes habituales
from umucv.stream   import autoStream
import cv2 as cv
import numpy as np
from umucv.contours import extractContours
from umucv.util import mkParam, putText
from umucv.htrans import htrans, desp, rot3, scale

def eig22(cxx,cyy,cxy):
    l,v = np.linalg.eigh(np.array([[cxx,cxy],[cxy,cyy]]))
    return l[1], l[0], np.arctan2(v[1][1],v[1][0])

def moments_2(c):
    m = cv.moments(c.astype(np.float32))  # int32, float32, but not float64!
    s = m['m00']
    return (m['m10']/s, m['m01']/s, m['mu20']/s, m['mu02']/s, m['mu11']/s)

def errorEllipse(e,c, mindiam = 20, minratio = 0.2):
    (cx,cy), (A,B), ang = e
    if A < mindiam: return 2000
    if B/A < minratio: return 1000
    T = np.linalg.inv(desp((cx,cy)) @ rot3(np.radians(ang)) @ scale((A/2,B/2)))
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




cv.namedWindow("elipses")
param = mkParam("elipses")
param.addParam("err",30,50)
param.addParam("area",5,20)




for key,frame in autoStream():
    g = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    print(param.area)
    cs = extractContours(g, minarea= max(1,param.area) ) 

    els = detectEllipses(cs, tol=param.err)

    for e in els:            

        cv.ellipse(frame,e, color=(0,0,255), thickness=2, lineType=cv.LINE_AA)

    cv.imshow('elipses',frame)


