#!/usr/bin/env python

# original file: http://dlib.net/face_landmark_detection.py.html

# modified for opencv gui

# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   The face detector we use is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset (see
#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#      300 faces In-the-wild challenge: Database and results. 
#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
#   You can get the trained model file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#   Note that the license for the iBUG 300-W dataset excludes commercial use.
#   So you should contact Imperial College London to find out if it's OK for
#   you to use this model file in a commercial product.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#

import dlib
predictor_path = "../../data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

import cv2          as cv
import numpy        as np
from umucv.stream import autoStream

for key,frame in autoStream():
    img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 0)
    for k, d in enumerate(dets):
        cv.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255,128,64) )
        
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        
        L = []
        for p in range(68):
            x = shape.part(p).x
            y = shape.part(p).y
            L.append(np.array([x,y]))
            cv.circle(frame, (x,y), 2,(255,0,0), -1)
        L = np.array(L)
        
        cv.line(frame, tuple(L[27]), tuple(L[8]), (0,0,255))

    cv.imshow("face landmarks",frame)

