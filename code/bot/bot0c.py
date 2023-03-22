#! /usr/bin/env python

# responde a comandos

from telegram.ext import Updater, CommandHandler
from telegram import Update, Bot

from os import environ
from dotenv import load_dotenv
load_dotenv('token.env')

MI_ID = environ.get('USER_ID', None)
UPDATER = Updater(environ.get('TOKEN'), use_context=True)
dispatcher = UPDATER.dispatcher
Bot: Bot = UPDATER.bot


def hola(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=f'Hola {update.effective_user.first_name}!')
hola_handler = CommandHandler('hola', hola)
dispatcher.add_handler(hola_handler)


def get_ID(update: Update, context):
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=f'Chat ID:`{update.effective_chat.id}`\nUser ID: `{update.effective_user}`')
info_handler = CommandHandler('info', get_ID)
dispatcher.add_handler(info_handler)


UPDATER.start_polling()

