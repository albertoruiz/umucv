#!/usr/bin/env python

# > ./deque.py --dev=file:../images/rot4.mjpg

import cv2         as cv
import numpy       as np
import argparse
from umucv.stream  import mkStream
from umucv.util    import sourceArgs
from collections   import deque


def bgr2gray(x):
    return cv.cvtColor(x,cv.COLOR_BGR2GRAY).astype(float)/255

def history(cam, n, fun = lambda x: x):
    d = deque(maxlen=n)
    for k in range(n-1):
        d.appendleft(fun(next(cam)))
    while True:
        d.appendleft(fun(next(cam)))
        yield np.array(d)


parser = argparse.ArgumentParser()
sourceArgs(parser)
args = parser.parse_args()

stream = mkStream(args.size, args.dev)

for h in history(stream, 10, bgr2gray):
    if cv.waitKey(1) & 0xFF == 27: break

    cv.imshow('input',h[0])
    #print(h.dtype)
    m = np.mean(h,axis=0)
    #print(m)
    cv.imshow('ghost', m)

cv.destroyAllWindows()

