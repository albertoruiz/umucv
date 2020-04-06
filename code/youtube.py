#!/usr/bin/env python

# pip install pafy youtube-dl

uso ="""
Muestra los streams disponibles de un vídeo:
> ./youtube.py C6LuDdY-KrQ

Imprime la url del stream seleccionado:
> ./youtube.py 5 C6LuDdY-KrQ

Con backquote se lo podemos pasar a --dev
> ./stream.py --dev=`./youtube.py 5 C6LuDdY-KrQ`

Si lo guardamos en una carpeta que esté en el path
podemos usarlo desde cualquier sitio (quitando ./ ).
"""

import pafy
import sys

args = sys.argv

if len(args) in [2,3]:
    url = args[-1]
    video = pafy.new(url)
    streams = video.videostreams
    if len(args) == 3:
        n = int(args[1])
        print(print(streams[n].url))
    else:
        for k,s in enumerate(streams):
            print(k,s)
else:
    print(uso)

