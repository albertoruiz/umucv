#!/usr/bin/env python

# adaptado de 
# https://fadeit.dk/blog/2015/04/30/python3-flask-pil-in-memory-image/

#  $ ./server.py
#  En navegador:  http://localhost:5000/320x240/image.png

import re
from io import BytesIO
from flask import Flask, abort, send_file
from PIL import Image, ImageDraw

import cv2 as cv

cap = cv.VideoCapture(0)
assert cap.isOpened()

def getframe():
    ret, frame = cap.read()
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

print(getframe().shape)


app = Flask(__name__)

@app.route('/<dimensions>/image.png')
def generate_image(dimensions):
    #Extract digits from request variable e.g 200x300
    try:
        [width, height] = [int(s) for s in re.findall(r'\d+', dimensions)]
    except:
      abort(400)
    
    g = cv.resize(getframe(),(width,height))

    # para evitar usar un archivo intermedio en disco
    image = Image.fromarray(g, mode = 'RGB')
    
    #draw = ImageDraw.Draw(image)
    #draw.text((50, 50), str(g.shape) )

    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)

    return send_file(byte_io, mimetype='image/png')
        
if __name__ == '__main__':
    app.run(debug=False)       # debug=True parece incompatible con V4L2 !?
    # app.run(host='0.0.0.0')  # Para que sea accesible públicamente

