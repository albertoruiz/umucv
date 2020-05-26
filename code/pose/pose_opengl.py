#!/usr/bin/env python

# Reproducimos el ejemplo de realidad aumentada pose3.py usando
# el motor gráfico OpenGL (a través de las utilidades de pyqtgraph).
# Las texturas (imágenes) y otras primitivas gráficas se ven con ocultación
# automática. La clave es situar correctamente la cámara de opengl
# y transformar los objetos con la pose estimada con el marcador.

# ./pose_opengl.py --dev=../../images/rot4.mjpg

import cv2   as cv

import numpy as np
from umucv.stream import autoStream
from umucv.htrans   import desp, scale, Pose, sepcam, jr, jc, col, row, rotation
from umucv.contours import extractContours, redu
from umucv.util import cube

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

################################################################################

# detección del marcador poligonal como en los ejemplos anteriores.
# (No es lo ideal, los puntos de referencia son inestables)

def polygons(cs,n,prec=2):
    rs = [ redu(c,prec) for c in cs ]
    return [ r for r in rs if len(r) == n ]

def rots(c):
    return [np.roll(c,k,0) for k in range(len(c))]

def bestPose(K,view,model):
    poses = [ Pose(K, v.astype(float), model) for v in rots(view) ]
    return sorted(poses,key=lambda p: p.rms)[0]

# encapsulamos aqui toda la funcionalidad de estimación de pose
def find(img, poly):
    g = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cs = extractContours(g, minarea=5, reduprec=2)
    good = polygons(cs,len(poly),3)
    poses = []
    for g in good:
        p = bestPose(K,g, poly)
        if p.rms < 2:
            poses += [p]
    return poses

# matriz de calibración sencilla a partir del fov horizontal en grados
def Kfov(sz,hfovd):
    hfov = np.radians(hfovd)
    f = 1/np.tan(hfov/2)
    # print(f)
    w,h = sz
    w2 = w / 2
    h2 = h / 2
    return np.array([[f*w2, 0,    w2],
                     [0,    f*w2, h2],
                     [0,    0,    1 ]])

################################################################################

# En esta sección definimos varias utilidades para facilitar el uso de pyqtgraph

# transforma un objeto adaptando el tipo de array de numpy al usado por pyqtgraph
def transform(H,obj):
    obj.setTransform(QtGui.QMatrix4x4(*(H.flatten())))

# para mostrar imágenes en la escena 3D deben convertirse en "texturas"
def img2tex(img):
    x = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    texture,_ = pg.makeARGB(x, useRGBA=True)
    return texture

# con canal alpha
def img2texA(img):
    x = cv.cvtColor(img, cv.COLOR_BGRA2RGBA)
    texture,_ = pg.makeARGB(x, useRGBA=True)
    return texture


# construye la matriz 4x4 de transformación de la escena a partir de una matriz de cámara
def cameraTransf2(M):
    K,R,C = sepcam(M)
    rt = jr(jc(R, -R @ col(C)),
            row(0,0,0,1))
    return rt

# construye la misma matriz de transformación de la escena a partir de un objeto Pose
# de forma un poco más eficiente, reaprovechamos R y t que ya están calculados
def cameraTransf(p):
    return jr(jc(p.R, col(p.t)), row(0,0,0,1))

# distancia de la cámara al origen (para evitar el valor cero)
DC = 1

# Esta función crea un objeto imagen y devuelve una función que actualiza su posicion
# con la transformación de la escena deducida de la matriz de cámara.
# Recibe también una transformación T que lleva la posición inicial de la imagen,
# en el plano z=0, y con anchura = 1, a la posición deseada.
def mkTexture(w,img,T):
    W = img.shape[1]
    if img.shape[2] == 3:
        obj = gl.GLImageItem(data=img2tex(img.transpose(1,0,2)))
    else:
        obj = gl.GLImageItem(data=img2texA(img.transpose(1,0,2)))
    w.addItem(obj)
    S = scale((1/W,1/W,1))
    D = desp((0,0,-DC))
    def update(H):
        transform( D @ H @ T @ S , obj )
    return update

# Esta función hace lo mismo con un objeto línea. No hace falta la transformación,
# ya que podemos crearla con los objetos ya transformados.
def mkLine(w,pts,color,width):
    obj = gl.GLLinePlotItem(pos=pts,color=color,antialias=True,width=width)
    obj.setGLOptions('opaque')
    w.addItem(obj)
    D = desp((0,0,-DC))
    def update(H):
        transform( D @ H, obj )
    return update

# Rotaciones homogéneas (4x4) en los 3 ejes, para construir fácilmente las
# transformaciones de los objetos y las texturas.
def rx(ad):
    return rotation((1,0,0),np.radians(ad),homog=True)

def ry(ad):
    return rotation((0,1,0),np.radians(ad),homog=True)

def rz(ad):
    return rotation((0,0,1),np.radians(ad),homog=True)

# pyqtgraph tiene una función parecida pero más limitada y alguna opción no funciona
# esta versión resuelve el problema.
def setCameraPosition(w, pos=None, distance=None, elevation=None, azimuth=None, fov=None):
    if pos is not None:
        w.opts['center'] = pos
    if distance is not None:
        w.opts['distance'] = distance
    if elevation is not None:
        w.opts['elevation'] = elevation
    if azimuth is not None:
        w.opts['azimuth'] = azimuth
    if fov is not None:
        w.opts['fov'] = fov
    w.update()      


# La clave para que los objetos virtuales se vean correctamente sobre la escena real
# es situar correctamente la "cámara" opengl y mostrar la imagen de cámara a la
# distancia justa.
# dist es la distancia a la que se proyecta la imagen de fondo. Si se queda corta
# respecto a las medidas del marcador los objetos pueden quedar ocultos por detrás
def prepare_RA(w, size, fov, dist=20):
    WIDTH,HEIGHT = size 
    # Ponemos el punto de vista de la visualización gráfica en el origen del sistema 
    # de referencia, con elevación y azimuth de modo que el eje x apunte hacia la 
    # derecha, el eje y hacia abajo y el eje z mirando hacia delante.
    # Es la posición "inicial" de una cámara en el origen.
    # (ponemos distancia > 0 que luego se compensa porque cero da problemas)
    setCameraPosition(w, distance=DC, elevation=-90, azimuth=-90, fov=fov)
    
    # Preparamos el objeto textura que contendrá la imagen de cámara en vivo. La vamos
    # a situar centrada delante del punto de vista del visor gráfico, a la distancia
    # justa para que ocupe toda la ventana, teniendo en cuenta el FOV. Es el fondo
    # de la escena, delante pondremos los objetos virtuales.
    W2 = WIDTH/2
    d = dist
    s = (d + DC)/W2 * np.tan(np.radians(fov)/2)
    camera_image = gl.GLImageItem(data=np.zeros([100,100,4]))
    transform( scale((s,s,1)) @ desp((-WIDTH//2,-HEIGHT//2, d)) , camera_image)
    w.addItem(camera_image)
    
    def update(frame):
        camera_image.setData(data=img2tex(frame.transpose(1,0,2)))
    
    return update

################################################################################


# leemos una imagen para usar como textura de paredes virtuales
bricks = cv.imread('bricks.jpg')


# añadimos canal alpha a la imagen y hacemos transparente una parte
puerta = cv.imread('bricks.jpg')
puerta = cv.cvtColor(puerta, cv.COLOR_RGB2RGBA)
puerta[:,:,3] = 255
puerta[0:200,100:200,3] = 0

## Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('3D')

# nuestro marcador de siempre
marker = np.array(
       [[0,   0,   0],
        [0,   1,   0],
        [0.5, 1,   0],
        [0.5, 0.5, 0],
        [1,   0.5, 0],
        [1,   0,   0]])


# preparamos la fuente de imágenes para que se puedan leer las imágenes con next
# de esta forma podemos consultar el tamaño de la imagen, y hacer la captura
# dentro del callback de pyqtgraph, donde no podemos usar for.
stream = autoStream()

HEIGHT, WIDTH = next(stream)[1].shape[:2]
size = WIDTH,HEIGHT
print(size)

# necesitamos el FOV de la cámara 
fov = 65

# para construir la matriz de calibración aproximada para la estimación de pose
K = Kfov(size, fov)

# y para generar función de opengl para visualización la imagen de cámara en vivo
update_view = prepare_RA(w,size,fov)


# Ya solo queda transformar la posición de los objetos virtuales con la rotación
# y desplazamiento que indica la matriz de cámara (pose) estimada.
# La proyección final la hace el motor gráfico.
# (En los ejemplos anteriores la hacíamos nosotros con htrans(M, obj) ) 
# Las funciones mkTexture y mkLine permiten generar cómodamente objetos virtuales.
# Como ejemplo ponemos cuatro paredes sobre los lados de un cuadrado unidad, centrado
# en el sistema de referencia que nos da el marcador. Ponemos también un "suelo" de color liso.
# Descomentando las líneas comentadas se muestran también el perfil del marcador y un cubo.

# para cambiar si queremos el alto y ancho de las paredes (deformando)
S = scale((1,1,1))  # (lo dejamos igual)

# Si hay objetos transparentes primero definimos los sólidos
# (aún así, si hay varios transparentes la escena puede quedar mal, 
# deben renderizarse de atrás hacia delante)

objects = [ mkLine(w, cube/2+(1.25,0.25,0), color=(128,255,255,1), width=2)
     #    , mkLine(w, marker + (0,0,0.05), color=(255,255,0,1), width=2)    
    #     , mkTexture(w, np.zeros((500,500,3),np.uint8) + np.array([64,92,64]).astype(np.uint8), desp((0,0,.3)) )

          , mkTexture(w, bricks, rx(90) @ S )
          , mkTexture(w, bricks, rz(90) @ rx(90) @ S )
          , mkTexture(w, bricks, desp((1,0,0)) @ rz(90) @ rx(90) @ S)
    
          , mkTexture(w, puerta, desp((0,1,0)) @ rx(90) @ S)
          ]

# Las transformaciones de cada pared sirven para ponerlas "de pie",
# lo que se consigue con una o dos rotaciones y un posible desplazamiento.
# Recuerda que la posición inicial "es el suelo" y que las operaciones se
# aplican de derecha a izquierda.


# callback de pyqtgraph que nos sirve de bucle de captura
def update():
    key, frame = next(stream)

    # actualizamos la imagen de cámara en la escena
    update_view(frame)
    
    # estimamos la pose
    poses = find(frame, marker)

    for p in poses:
        # extraemos la transformación de la escena de la matriz de cámara
        H = cameraTransf(p)
        
        for x in objects:
            # movemos los objetos virtuales a la posición adecuada
            # para que se vean con la perspectiva correcta.
            x(H)
        
        break
        # solo podemos responder a un marcador (solo hay una escena virtual).


# Arrancamos la aplicación
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(30)
QtGui.QApplication.instance().exec_()

