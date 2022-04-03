#!/usr/bin/env python

import cv2   as cv

# basado en la demo pyqtgraph example: GLSurfacePlot
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
from umucv.stream import autoStream
from umucv.htrans   import desp, scale


## Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('gray level surface')
w.setCameraPosition(distance=50)

## Add a grid to the view
g = gl.GLGridItem()
w.addItem(g)


# floor
imagegl = gl.GLImageItem(data=np.zeros([100,100,4]))
w.addItem(imagegl)

def img2tex(img):
    x = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    texture,_ = pg.makeARGB(x, useRGBA=True)
    return texture

def transform(H,obj):
    obj.setTransform(QtGui.QMatrix4x4(*(H.flatten())))


stream = autoStream()


def update():
    key, frame = next(stream)
    #p.setData(z=getframe())
    imagegl.setData(data=img2tex(frame))
    s = 1/50
    transform( scale((s,s,1)) @ desp((0,0,1)) , imagegl)
    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)

QtGui.QApplication.instance().exec_()

