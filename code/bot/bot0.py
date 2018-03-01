#! /usr/bin/env python

# envía a tu cuenta de telegram la ip de la máquina

# >pip install python-telegram-bot
# Con BotFather creas el bot y te dará el api token

from telegram.ext import Updater
import os

updater = Updater('el api token de tu bot')

Bot = updater.bot

myid = "el id de tu usuario, te lo da IDBot"

def myip():
    os.system("hostname -I > myip.txt")
    ips = open('myip.txt', 'r').read().replace('\n',' ')
    return ips

Bot.sendMessage(chat_id=myid, text='Hello! My IP is {}'.format(myip()))

