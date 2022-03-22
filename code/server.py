#!/usr/bin/env python

# adaptado de 
# https://fadeit.dk/blog/2015/04/30/python3-flask-pil-in-memory-image/

#  $ ./server.py
#  En navegador:  http://localhost:5000/resize/320x240/via.png

from io import BytesIO
from flask import Flask, send_file
from PIL import Image
import cv2 as cv

if False:
    cap = cv.VideoCapture(0)
    def getframe():
        ret, frame = cap.read()
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
elif False:
    from umucv.stream import autoStream
    stream = autoStream()
    def getframe():
        _,frame = next(stream)
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
else:
    # captura asíncrona en un hilo
    # para devolver siempre el frame más reciente
    from umucv.stream import Camera
    cam = Camera()
    def getframe():
        return cv.cvtColor(cam.frame, cv.COLOR_BGR2RGB)


def send_image(imgRGB):
    # para evitar un archivo intermedio en disco
    image = Image.fromarray(imgRGB, mode = 'RGB')
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')


app = Flask(__name__)

@app.route('/resize/<ancho>x<alto>/via.png')
def generate_image(ancho,alto):
    x = getframe()
    r = cv.resize(x,(int(ancho),int(alto)))
    return send_image(r)

# podemos añadir rutas para servir html, templates, etc.

app.run(debug=False)       # debug=True parece incompatible con V4L2 !?
# app.run(host='0.0.0.0')  # Para que sea accesible públicamente

cam.stop()

