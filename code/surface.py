#!/usr/bin/env python

import cv2   as cv

# basado en la demo pyqtgraph example: GLSurfacePlot
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np


## Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('gray level surface')
w.setCameraPosition(distance=50)

## Add a grid to the view
g = gl.GLGridItem()
w.addItem(g)


cap = cv.VideoCapture(0)
assert cap.isOpened()

def getframe():
    ret, frame = cap.read()    
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(float)/255

p = gl.GLSurfacePlotItem(z = getframe(), shader='heightColor', computeNormals=False, smooth=False)
p.scale(20/640,20/640,10)
p.translate(-8,-10,0)
w.addItem(p)

def update():    
    p.setData(z=getframe())
    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)

QtGui.QApplication.instance().exec_()

