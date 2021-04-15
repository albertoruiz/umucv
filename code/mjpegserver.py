#!/usr/bin/env python

# Adapted from:

'''
    Author: Igor Maculan - n3wtron@gmail.com
    A Simple mjpg stream http server
'''
# https://gist.github.com/n3wtron/4624820

import cv2
from PIL import Image
import threading
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
from io import StringIO,BytesIO
import time
import subprocess


from umucv.stream import Camera, sourceArgs
import signal
import sys
import argparse

parser = argparse.ArgumentParser()
sourceArgs(parser)
parser.add_argument('--quality', help='jpeg quality', type=int, default=30)
args = parser.parse_args()

QUALITY = args.quality

def encode(frame):
    imgRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    jpg = Image.fromarray(imgRGB)
    tmpFile = BytesIO()
    jpg.save(tmpFile,'JPEG',quality=QUALITY)
    #print('encoded', tmpFile.getbuffer().nbytes)
    return tmpFile

cam = Camera(args.size, args.dev, debug=False)

stop = False

def signal_handler(signal, frame):
        global stop
        cam.stop()
        stop = True
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            t = 0
            while not stop:
                try:
                    while cam.time == t:
                        time.sleep(0.01)
                        ##print('.')
                    t = cam.time
                    result = 255 - cam.frame
                    tmpFile = encode(result)
                    #print('sent',tmpFile.getbuffer().nbytes)
                    self.wfile.write("--jpgboundary\r\n".encode())
                    self.send_header('Content-type','image/jpeg')
                    self.send_header('Content-length',str(tmpFile.getbuffer().nbytes))
                    self.end_headers()
                    self.wfile.write(tmpFile.getvalue())
                except:
                    break

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def main():
    try:
        
        server = ThreadedHTTPServer(('', 8087), CamHandler)
        #ip = subprocess.check_output(["hostname", "-I"]).decode('utf-8')[:-2]
        print(f"server started at http://localhost:8087/cam.mjpg")
        server.serve_forever()
    except KeyboardInterrupt:
        server.socket.close()

if __name__ == '__main__':
    main()

