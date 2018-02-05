#!/usr/bin/env python

import cv2 as cv
import datetime

def play(f,dev=0):
    cap = cv.VideoCapture(dev)

    pausa = False
    while(True):
        key = cv.waitKey(1) & 0xFF
        if key == 27: break
        if key == ord(' '): pausa = not pausa
        if pausa: continue
        ret, frame = cap.read()
        r = f(frame)
        cv.imshow('original',frame)
        cv.imshow('result',r)
        if key == ord('s'):
            fname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            cv.imwrite(fname+'.png',frame)
    cv.destroyAllWindows()

def fun(x):
    return 255 - cv.cvtColor(x, cv.COLOR_RGB2GRAY)

if __name__ == "__main__":
    play(fun)

