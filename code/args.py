#!/usr/bin/env python

# Cómo añadir más argumentos en la línea de órdenes


import cv2 as cv
from umucv.stream import autoStream, sourceArgs
import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('--fov', help='horizontal field of view (degrees)', type=float, default=None)
parser.add_argument('--steps', help='number of repetitions', type=int, default=5)

# añade los parámetros de autoStream
sourceArgs(parser)
args, rest = parser.parse_known_args(sys.argv)
assert len(rest)==1, 'unknown parameters: '+str(rest[1:])

# comprobamos que se han capturado bien
print(f"steps = {args.steps}")

if args.fov is not None:
    print(f"FOV = {args.fov:.1f}º")



for key,frame in autoStream():
    cv.imshow('input',frame)


