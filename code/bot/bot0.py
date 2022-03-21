#! /usr/bin/env python

# envía a tu cuenta de telegram un mensaje

# >pip install python-telegram-bot

# Con BotFather creas el bot y te dará el api token IDBot te da tu id de usuario Estos valores podemos almacenamos en
# .env Este fichero se puede cargar mediante linea de comando o opciones de ejecucion del IDE ** Importante,
# el fichero .env no subirlo al repositorio si es publico. Lo más recomendable es añadir la excepcion a gitignore

from telegram.ext import Updater, CommandHandler
from telegram import Update, Bot
import subprocess
from os import environ
from dotenv import load_dotenv # pip install python-dotenv

load_dotenv()
# En caso de que vuestro IDE no tenga soporte para .env files, esta llamada a la libreria las carga si estan en ./

MI_ID = environ.get('USER_ID', None)

UPDATER = Updater(environ.get('TOKEN'), use_context=True)
dispatcher = UPDATER.dispatcher
Bot: Bot = UPDATER.bot


def get_ID(update: Update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=f'Chat ID:`{update.effective_chat.id}`\nUser ID: `{update.effective_user}`')


start_handler = CommandHandler('start', get_ID) # Al enviarle al bot /start, te da la id del chat por el le has hablado
dispatcher.add_handler(start_handler)


def shutdown():
    UPDATER.stop()
    UPDATER.is_idle = False


start_handler = CommandHandler('shutdown', shutdown)
dispatcher.add_handler(start_handler)


def myip():
    if True:
        # Linux
        # Convierte a string utf-8 y elimina el espacio + salto de linea del final
        return subprocess.check_output(["hostname", "-I"]).decode('utf-8')[:-2].split()[0]
    else:
        # Windows
        return '???'


UPDATER.start_polling()
if MI_ID:
    Bot.sendMessage(chat_id=MI_ID, text=f'Hello! My IP is {myip()}')
    Bot.send_message(MI_ID, f'Hey, I am Online')
