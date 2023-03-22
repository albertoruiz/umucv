#! /usr/bin/env python

# envía a tu cuenta de telegram un mensaje

# pip install python-telegram-bot==13.0
# pip install python-dotenv

# Con BotFather creas el bot y te dará el api token
# IDBot te da tu id de usuario

# Almacenamos estos valores en token.env y se leen como variables de entorno
# ** Importante: no subir el fichero .env al repositorio si es publico  
# Lo más recomendable es añadir la excepcion a gitignore

from telegram.ext import Updater, CommandHandler
from telegram import Update, Bot

from os import environ
from dotenv import load_dotenv
load_dotenv('token.env')

MI_ID = environ.get('USER_ID', None)
UPDATER = Updater(environ.get('TOKEN'), use_context=True)
dispatcher = UPDATER.dispatcher
Bot: Bot = UPDATER.bot

Bot.send_message(MI_ID, f'Hey, I am Online')

