#! /usr/bin/env python

# envía una imagen cuando se la pides

from telegram.ext import Updater, CommandHandler

from io import BytesIO
from PIL import Image
import cv2 as cv
from umucv.stream import Camera

from dotenv import load_dotenv
import os
load_dotenv('token.env')
updater = Updater(os.environ['TOKEN'])
Bot = updater.bot

cam = Camera()

def hello(update, _):
    update.message.reply_text('Hello {}'.format(update.message.from_user.first_name))

def sendImage(userid, frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame, mode = 'RGB')
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    Bot.sendPhoto(chat_id=userid, photo=byte_io)

def image(update,_):
    cid = update.message.chat.id
    img = cam.frame
    sendImage(cid, img)

updater.dispatcher.add_handler(CommandHandler('hello', hello))
updater.dispatcher.add_handler(CommandHandler('image', image))

updater.start_polling()

