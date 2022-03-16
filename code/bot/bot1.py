#! /usr/bin/env python

# envía una imagen cuando ocurre algo

# (en este caso cuando se pulsa la tecla b,
# pero la idea es enviarla cuando se detecta actividad, etc.)
import os

from telegram.ext import Updater

from io import BytesIO
from PIL import Image
import cv2 as cv
from umucv.stream import autoStream

from dotenv import load_dotenv

load_dotenv()

Bot = Updater(os.environ['TOKEN']).bot


def sendImage(userid, frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame, mode='RGB')
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    Bot.sendPhoto(chat_id=userid, photo=byte_io)


for key, frame in autoStream():
    cv.imshow('image', frame)
    if key == ord('b'):
        sendImage(os.environ['USER_ID'], frame)
