#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @author: josem
    Bot de Telegram que devuelve la IP del servidor
    Adaptado de los bots de Alberto disponibles en umucv
"""

# Libreria Telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
# OpenCV
import cv2 as cv
# Enviar bytes
from io import BytesIO
# Para cargar una imagen dados los bytes
from PIL import Image
# Para cargar una imagen dado el path
import skimage.io as io
# Stream de la camara
from umucv.stream import Camera
# Hilos
import threading
# Ejecutar comandos
import subprocess
# Manejo del sistema
import os


# Updater con el token del bot (otorogado por BotFather)
updater = Updater('TOKEN')

# Mi id de usuario de Telegram (otorogado por IDBot)
my_id = 0

# Camara activada
cam_enabled = False

# Camara
if cam_enabled:
    cam = Camera(dev='0', size=(640, 480))

# Directorio actual (para execute)
dir_path = subprocess.check_output("pwd").decode('utf-8')[:-1]


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
    if cam_enabled:
        cam.stop()


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


# Envia imagen
# Recibe el bot, el id del chat y el frame
def send_image(bot, cid, frame):
    # Convierte a RGB
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Carga imagen dada el array que es el frame
    image = Image.fromarray(frame, mode='RGB')
    # Crea el stream de bytes
    byte_io = BytesIO()
    # Guarda la imagen en formato PNG en el stream
    image.save(byte_io, 'PNG')
    # Pone el stream en la posicion 0
    byte_io.seek(0)
    # El bot envia la foto al chat
    bot.sendPhoto(chat_id=cid, photo=byte_io)


# Comando para enviar una imagen (captura)
# Recibe el bot y el evento nuevo
def get_image(bot, update):
    # Id del chat
    cid = update.message.chat_id
    if cid == my_id:
        if cam_enabled:
            # Captura de la camara
            img = cam.frame
            # Envia la imagen
            send_image(bot, cid, img)
        else:
            update.message.reply_text("No hay webcam")
    else:
        update.message.reply_text("Bot privado!")


# Comando que imprime los argumentos (echo)
# Recibe el bot, el evento nuevo y los argumentos
def echo(bot, update, args):
    # Une los elementos de args y separa con espacios
    update.message.reply_text(" ".join(args))


# Comando que ejecuta un comando en la terminal
# Recibe el bot, el evento nuevo y los argumentos
def execute(bot, update, args):
    global dir_path
    # Id del chat
    cid = update.message.chat_id
    if cid == my_id:
        # Prompt = whoami + hostname + pwd
        user = subprocess.check_output("whoami").decode('utf-8')[:-1]
        host = subprocess.check_output("hostname").decode('utf-8')[:-1]
        prompt = user + "@" + host + ":" + dir_path + " $ " + " ".join(args) + "\n"
        # Actualiza el directorio actual si es cd
        if args[0] == "cd":
            # Path es el home del usuario
            if len(args) < 2:
                dir_path = "/home/"+user
                out = ""
            # Si hay path, debe existir
            elif os.path.isdir(args[1]):
                dir_path = args[1]
                out = ""
            # Si no, error
            else:
                out = "No existe el directorio"
        # Ejecuta el comando (argumentos) y convierte a string utf-8
        else:
            # Cambia el directorio actual
            command = " ".join(["cd", dir_path, "&&"] + args)
            out = subprocess.check_output(command, shell=True).decode('utf-8')[:-1]
        # Envia el mensaje con la salida y elimina el salto de linea del final
        update.message.reply_text(prompt+out)
    else:
        update.message.reply_text("Bot privado!")


# Comando que procesa un mensaje del usuario y lo devuelve invertido
# Recibe el bot y el evento nuevo
def process_text(bot, update):
    update.message.reply_text(update.message.text[::-1])


# Comando que procesa una imagen del usuario y la devuelve en gris
# Recibe el bot y el evento nuevo
def process_image(bot, update):
    # El identificador de la ultima foto
    file_id = update.message.photo[-1].file_id
    # Path para el archivo
    path = bot.get_file(file_id)['file_path']
    # Carga la imagen dado el path
    img = io.imread(path)
    # Responde con las dimensiones de la imagen
    update.message.reply_text("{}x{}".format(img.shape[1], img.shape[0]))
    # Imagen en gris
    r = cv.cvtColor(cv.cvtColor(img, cv.COLOR_RGB2GRAY), cv.COLOR_GRAY2RGB)
    # Envia la imagen
    send_image(bot, update.message.chat_id, r)


# Main
def main():
    # Manejadores
    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('hello', hello))
    updater.dispatcher.add_handler(CommandHandler('stop', stop))
    updater.dispatcher.add_handler(CommandHandler('ip', get_ip))
    updater.dispatcher.add_handler(CommandHandler('image', get_image))
    updater.dispatcher.add_handler(CommandHandler('echo', echo, pass_args=True))
    updater.dispatcher.add_handler(CommandHandler('exec', execute, pass_args=True))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, process_text))
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, process_image))
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
