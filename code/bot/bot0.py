#! /usr/bin/env python

# envía a tu cuenta de telegram la ip de la máquina

# >pip install python-telegram-bot
# Con BotFather creas el bot y te dará el api token

from telegram.ext import Updater
import subprocess

myid = "el id de tu usuario, te lo da IDBot"

updater = Updater('el api token de tu bot')

Bot = updater.bot


def myip():
    # Convierte a string utf-8 y elimina el espacio + salto de linea del final
    return subprocess.check_output(["hostname", "-I"]).decode('utf-8')[:-2]

Bot.sendMessage(chat_id=myid, text='Hello! My IP is {}'.format(myip()))

