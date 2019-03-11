#!/usr/bin/env python

# adapted from 
# https://github.com/opencv/opencv/blob/master/samples/python/lk_track.py

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
'''

import numpy as np
import cv2 as cv
import time

from umucv.stream import autoStream
from umucv.util import putText


lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

track_len = 20
detect_interval = 5
tracks = []
frame_idx = 0


for key, frame in autoStream():
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    vis = frame.copy()

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        t0 = time.time()
        p1,  _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        t1 = time.time()
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < 1
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)
            cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
       
        tracks = new_tracks
        cv.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
        putText(vis, 'tracks: {}, {:.0f}ms'.format(len(tracks), 1000*(t1-t0)) )
        
    #for t in tracks:
    #    print( t[0],t[-1] )

    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv.circle(mask, (x, y), 5, 0, -1)
        p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])

    frame_idx += 1
    prev_gray = frame_gray
    cv.imshow('lk_track', vis)

cv.destroyAllWindows()

