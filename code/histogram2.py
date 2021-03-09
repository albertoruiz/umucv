#!/usr/bin/env python

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from umucv.stream import autoStream

fin = False
def on_press(event):
    global fin
    if event.key == 'escape':
        fin = True

plt.ion()
fig = plt.figure(figsize=(10,3))
fig.canvas.mpl_connect('key_press_event', on_press)

fig.suptitle('histogram')

ax = fig.add_axes([-0.25,0,1,1])
im = ax.imshow(np.zeros((480,640)),'gray',vmin=0,vmax=255)
ax.set_axis_off();

ax2 = fig.add_axes([0.55,0.1,0.4,0.8])
l1, = ax2.plot([], [], '-r',lw=2)
ax2.set_xlim(0,255)
ax2.set_ylim(0,10000)

plt.show()

for key, frame in autoStream():

    x = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    im.set_data(x)

    h,b = np.histogram(x, np.arange(257))

    l1.set_data(b[1:],h);

    #plt.pause(0.001)
    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.001)
    if fin: break

