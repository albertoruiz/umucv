#! /usr/bin/env python

# responde a comandos

from telegram.ext import Updater, CommandHandler
import threading

from io import BytesIO
from PIL import Image
import cv2 as cv
from umucv.stream import Camera

updater = Updater('api token del bot')

cam = Camera(dev='0',sz=(640,480))

def shutdown():
    updater.stop()
    updater.is_idle = False
    cam.stop()

def stop(bot, update):
    update.message.reply_text('Bye!')
    threading.Thread(target=shutdown).start()

def hello(bot, update):
    update.message.reply_text('Hello {}'.format(update.message.from_user.first_name))

def sendImage(bot, cid, frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame, mode = 'RGB')
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    bot.sendPhoto(chat_id=cid, photo=byte_io)

def image(bot, update):
    cid = update.message.chat_id
    img = cam.frame
    sendImage(bot, cid, img)


updater.dispatcher.add_handler(CommandHandler('stop',  stop))
updater.dispatcher.add_handler(CommandHandler('hello', hello))
updater.dispatcher.add_handler(CommandHandler('image', image))

updater.start_polling()
updater.idle()

