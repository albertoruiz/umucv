#!/usr/bin/env python

# adaptado de 
# https://fadeit.dk/blog/2015/04/30/python3-flask-pil-in-memory-image/

#  $ ./server.py
#  En navegador:  http://localhost:5000/320x240/image.png

from io import BytesIO
from flask import Flask, send_file
from PIL import Image
import cv2 as cv

if False:
    cap = cv.VideoCapture(0)
    def getframe():
        ret, frame = cap.read()
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
else:
    # captura asíncrona en un hilo
    from umucv.stream import Camera
    cam = Camera()
    def getframe():
        return cv.cvtColor(cam.frame, cv.COLOR_BGR2RGB)


app = Flask(__name__)

@app.route('/<ancho>x<alto>/image.png')
def generate_image(ancho,alto):
    
    r = cv.resize(getframe(),(int(ancho),int(alto)))

    # para evitar usar un archivo intermedio en disco
    image = Image.fromarray(r, mode = 'RGB')
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')
        
app.run(debug=False)       # debug=True parece incompatible con V4L2 !?
# app.run(host='0.0.0.0')  # Para que sea accesible públicamente

cam.stop()

