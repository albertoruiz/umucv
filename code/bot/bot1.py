#! /usr/bin/env python

# envía una imagen cuando ocurre algo

from telegram.ext import Updater

from io import BytesIO
from PIL import Image
import cv2 as cv
from umucv.stream import autoStream

updater = Updater('api token del bot')

Bot = updater.bot

myid = "id del destinatario"


def sendImage(bot, cid, frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame, mode = 'RGB')
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    bot.sendPhoto(chat_id=cid, photo=byte_io)


for key, frame in autoStream():
    cv.imshow('image',frame)
    if key == ord('b'):
        sendImage(Bot, myid, frame)

