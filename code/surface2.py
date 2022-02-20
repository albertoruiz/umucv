#!/usr/bin/env python

import cv2   as cv

# basado en la demo pyqtgraph example: GLSurfacePlot
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

from umucv.stream import autoStream

class KeyPressWindow(gl.GLViewWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.sigKeyPress.emit(ev)

sigma = 1

def keyPressed(event):
    global sigma
    code = event.key()
    print("Key pressed:", code)
    if code == ord('B'):
        sigma += 1
        print('sigma:',sigma)

## Create a GL View widget to display data
app = QtGui.QApplication([])

#w = gl.GLViewWidget()
w = KeyPressWindow()
w.sigKeyPress.connect(keyPressed)


w.show()
w.setWindowTitle('gray level surface')
w.setCameraPosition(distance=50)

## Add a grid to the view
g = gl.GLGridItem()
w.addItem(g)

source = autoStream()

def getframe():
    key, frame = next(source)
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(float)/255

p = gl.GLSurfacePlotItem(z = getframe(), shader='heightColor', computeNormals=False, smooth=False)
p.scale(20/640,20/640,10)
p.translate(-8,-10,0)
w.addItem(p)

def update():
    newimage = getframe()
    cv.imshow("input",newimage)
    smooth = cv.GaussianBlur(newimage,(-1,-1), sigma)
    cv.imshow("smooth",smooth)
    p.setData(z=smooth)
    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)

QtGui.QApplication.instance().exec_()

