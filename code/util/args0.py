#!/usr/bin/env python

# Cómo definir argumentos en la línea de órdenes junto con los que
# ya tiene autoStream (--dev, --size, etc.)

import cv2 as cv
from umucv.stream import autoStream, sourceArgs

import argparse, sys
parser = argparse.ArgumentParser()
parser.add_argument('--fov', help='horizontal field of view (degrees)', type=float, default=60)
parser.add_argument('--only', help='stop after this number of frames', type=int, default=None)

# añade los parámetros de autoStream
# (si no lo hacemos también funcionará, pero habría que quitar
#  el assert más abajo y no podríamos comprobar si hay parámetros erróneos
sourceArgs(parser)
 
args, rest = parser.parse_known_args(sys.argv)
assert len(rest)==1, 'unknown parameters: '+str(rest[1:])

print(f"FOV = {args.fov:.1f}º")

for k, (key,frame) in enumerate(autoStream()):
    cv.imshow('input',frame)
    if args.only and k == args.only: break

