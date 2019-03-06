#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @author: josem
    Bot de Telegram que devuelve la IP del servidor
    Adaptado de los bots de Alberto disponibles en umucv
    Version simple para mostrar la autenticacion del usuario.
"""

# Libreria Telegram
from telegram.ext import Updater, CommandHandler
# Hilos
import threading
# Ejecutar comandos
import subprocess


# Updater con el token del bot (otorogado por BotFather)
updater = Updater('TOKEN')

# Mi id de usuario de Telegram (otorogado por IDBot)
my_id = 0


# Comando para iniciar el bot
# Recibe el bot y el evento nuevo
def start(bot, update):
    # Responde al mensage recibido
    update.message.reply_text("Estoy vivo!")


# Comando para saludar en el chat
# Recibe el bot y el evento nuevo
def hello(bot, update):
    # Responde al mensage recibido
    update.message.reply_text("Hola {}".format(update.message.from_user.first_name))


# Cierra la camara y el hilo del updater
def shutdown():
    updater.stop()
    updater.is_idle = False


# Comando para detener el bot, lanza shutdown
# Recibe el bot y el evento nuevo
def stop(bot, update):
    # Id del chat
    cid = update.message.chat_id
    if cid == my_id:
        # Responde al mensage recibido
        update.message.reply_text("Bye!")
        # Lanza shutdown
        threading.Thread(target=shutdown).start()
    else:
        update.message.reply_text("Bot privado!")


# Ejecuta el comando para obtener la ip
def ip():
    # Convierte a string utf-8 y elimina el espacio + salto de linea del final
    return subprocess.check_output(["hostname", "-I"]).decode('utf-8')[:-2]


# Comando para enviar la ip
# Recibe el bot y el evento nuevo
def get_ip(bot, update):
    # Id del chat
    cid = update.message.chat_id
    if cid == my_id:
        # Envia el mensaje con la ip
        update.message.reply_text("Mi IP es {}".format(ip()))
    else:
        update.message.reply_text("Bot privado!")


# Main
def main():
    # Manejadores
    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('hello', hello))
    updater.dispatcher.add_handler(CommandHandler('stop', stop))
    updater.dispatcher.add_handler(CommandHandler('ip', get_ip))
    # Comienza el bot
    updater.start_polling()
    # Mensaje inicial
    bot = updater.bot
    bot.sendMessage(chat_id=my_id, text="Mi IP es {}".format(ip()))
    # Bloquea la ejecucion hasta que se para el bot
    updater.idle()


# Programa principal
if __name__ == '__main__':
    main()
