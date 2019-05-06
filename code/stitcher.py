#!/usr/bin/env python

# ./stitcher.py '../images/pano/*.jpg' --output=campus.jpg


'''
Stitching sample
================

Show how to use Stitcher API from python in a simple way to stitch panoramas
or scans.
'''

import cv2 as cv
import numpy as np
import argparse
import sys
import glob

if True:
    modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

    parser = argparse.ArgumentParser(description='Stitching sample.')
    parser.add_argument('--mode',
     type = int, choices = modes, default = cv.Stitcher_PANORAMA,
     help = 'Determines configuration of stitcher. The default is `PANORAMA` (%d), '
          'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
          'for stitching materials under affine transformation, such as scans.' % modes)
    parser.add_argument('--output', default = 'result.jpg',
     help = 'Resulting image. The default is `result.jpg`.')
    parser.add_argument('img', help = 'input images')
    args = parser.parse_args()

# read input images
imgs = []
for img_name in glob.glob(args.img):
 print(img_name)
 img = cv.imread(img_name)
 if img is None:
     print("can't read image " + img_name)
     sys.exit(-1)
 imgs.append(img)

stitcher = cv.createStitcher(args.mode)
status, pano = stitcher.stitch(imgs)

if status != cv.Stitcher_OK:
 print("Can't stitch images, error code = %d" % status)
 sys.exit(-1)

cv.imwrite(args.output, pano);
print("stitching completed successfully. %s saved!" % args.output)

