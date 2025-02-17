#!/usr/bin/env python

# Cómo definir argumentos en la línea de órdenes junto con los que
# ya tiene autoStream (--dev, --size, etc.)

import cv2 as cv
from umucv.stream import autoStream
from umucv.util import read_arguments

def my_arguments(parser):
    parser.add_argument('--fov', help='horizontal field of view (degrees)', type=float, default=60)
    parser.add_argument('--only', help='stop after this number of frames', type=int, default=None)

args = read_arguments(my_arguments)


print(f"FOV = {args.fov:.1f}º")

for k, (key,frame) in enumerate(autoStream()):
    cv.imshow('input',frame)
    if args.only and k == args.only: break

