#!/usr/bin/env python

import cv2 as cv
import sys

if len(sys.argv) < 2:
    # with no arguments, use this default file
    filename = "../images/cube3.png"
elif len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    sys.exit("Expecting a single image file argument")

image = cv.imread(filename)
print(image.shape)

image_small = cv.resize(image, (800, 600))

textColor = (0, 0, 255)  # red
cv.putText(image_small, "Hello World!!!", (200, 200),
           cv.FONT_HERSHEY_PLAIN, 3.0, textColor,
           thickness=4)

cv.imshow('Hello World GUI', image_small)
cv.waitKey()
cv.destroyAllWindows()

