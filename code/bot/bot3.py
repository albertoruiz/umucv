#! /usr/bin/env python

# comando con argumentos
# y procesamiento de una imagen
# enviada por el usuario

from dotenv import load_dotenv
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from io import BytesIO
from PIL import Image
import cv2 as cv
import skimage.io as io
import numpy as np

load_dotenv()


################################################################################

def process(img):
    r = cv.cvtColor(cv.cvtColor(img, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)
    r = np.fliplr(r)
    return r


################################################################################


Bot = updater.bot

def sendImage(userid, frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame, mode = 'RGB')
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    Bot.sendPhoto(chat_id=userid, photo=byte_io)

def hello(update,_):
    update.message.reply_text('Hello {}'.format(update.message.from_user.first_name))

def argu(update, _):
    print(update)
    args = update.message.text.split()
    print('arguments:')
    for arg in args:
       print(arg)
    update.message.reply_text(str(sum([int(a) for a in args[1:]])))

def work(update,_):
    file_id = update.message.photo[-1].file_id
    path = Bot.get_file(file_id)['file_path']
    # print(path)
    img = io.imread(path)
    print(update.message.from_user.first_name, img.shape)
    update.message.reply_text('{}x{}'.format(img.shape[1], img.shape[0]))
    r = process(img)
    sendImage(update.message.chat.id, r)


updater.dispatcher.add_handler(CommandHandler('hello', hello))
updater.dispatcher.add_handler(CommandHandler('argu' , argu))
updater.dispatcher.add_handler(MessageHandler(Filters.photo, work))


updater.start_polling()
updater.idle()

