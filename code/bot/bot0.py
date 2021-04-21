#! /usr/bin/env python

# envía a tu cuenta de telegram un mensaje

# >pip install python-telegram-bot

# Con BotFather creas el bot y te dará el api token
# IDBot te da tu id de usuario
# los almacenamos en mybotid.py

from telegram.ext import Updater
import subprocess

from mybotid import myid, mybot

Bot = Updater(mybot).bot


def myip():
    if True:
        # Linux
        # Convierte a string utf-8 y elimina el espacio + salto de linea del final
        return subprocess.check_output(["hostname", "-I"]).decode('utf-8')[:-2].split()[0]
    else:
        # Windows
        return '???'

Bot.sendMessage(chat_id=myid, text=f'Hello! My IP is {myip()}')

