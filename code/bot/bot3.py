#! /usr/bin/env python

# comando con argumentos
# y procesamiento de una imagen
# enviada por el usuario

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from io import BytesIO
from PIL import Image
import cv2 as cv
import skimage.io as io

updater = Updater('api token del bot')

def sendImage(bot, cid, frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame, mode = 'RGB')
    byte_io = BytesIO()
    image.save(byte_io, 'PNG')
    byte_io.seek(0)
    bot.sendPhoto(chat_id=cid, photo=byte_io)

def hello(bot, update):
    update.message.reply_text('Hello {}'.format(update.message.from_user.first_name))

def argu(bot, update, args):
    print('arguments:')
    for arg in args:
       print(arg)

def work(bot, update):
    file_id = update.message.photo[-1].file_id
    path = bot.get_file(file_id)['file_path']
    img = io.imread(path)
    print(img.shape)
    update.message.reply_text('{}x{}'.format(img.shape[1],img.shape[0]))
    r = cv.cvtColor(cv.cvtColor(img, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)
    sendImage(bot, update.message.chat_id, r)


updater.dispatcher.add_handler(CommandHandler('hello', hello))
updater.dispatcher.add_handler(CommandHandler('argu' , argu, pass_args=True))
updater.dispatcher.add_handler(MessageHandler(Filters.photo, work))


updater.start_polling()
updater.idle()

