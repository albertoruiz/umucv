#!/usr/bin/env python

# 4) Construimos una red convolucional

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util   import putText
import time

##############################################################################
# pip install tensorflow keras
import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Softmax, Flatten

# empezamos a definir la red neuronal, exactamente igual que en el notebook
model = Sequential()
# la primera capa es convolucional con 32 filtros 5x5
model.add(Conv2D(input_shape=(28,28,1), filters=32, kernel_size=(5,5), strides=1,
                 padding='same', use_bias=True, activation='relu'))
# la segunda reduce la resolución a la mitad
model.add(MaxPool2D(pool_size=(2,2)))
# la tercera es convolucional, con 64 filtros 5x5
model.add(Conv2D(filters=64, kernel_size=(5,5), strides=1,
                 padding='same', use_bias=True, activation='relu'))
# volvemos a reducir la resolución a la mitad
model.add(MaxPool2D(pool_size=(2,2)))
# ponemos los resultados de todos los filtros como un vector
model.add(Flatten())
# añadimos una capa densa (completamente conectada) que produce 1024 features
model.add(Dense(1024,activation='relu'))
# una etapa de regularización
model.add(Dropout(rate=0.5))
# y finalmente la capa de salida con un elemento para cada clase
model.add(Dense(10, activation='softmax'))

# preparamos la red (aunque no la vamos a entrenar ahora)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# leemos los pesos preentrenados
# están aquí:
# wget https://robot.inf.um.es/material/va/digits.keras
model.load_weights('digits.keras')

def classifyN(xs):
    # ponemos la estructura de array que espera la red: una lista de imágenes de un canal
    t = np.array(xs).reshape(-1,28,28,1)
    # y hacemos lo mismo de antes, devolvemos la clase más probable y su probabilidad
    p = model.predict(t)
    r = np.argmax(p,axis=1)
    pm = np.max(p,axis=1)
    return r,pm

# (mas abajo elegiremos esta función de clasifación para comparar con la anterior)

########################################################################################
from sklearn import decomposition, discriminant_analysis

# pon el path correcto, el archivo está en repo/umucv/data
mnist = np.load("../../../data/mnist.npz")
xl,yl,xt,yt = [mnist[d] for d in ['xl', 'yl', 'xt', 'yt']]
cl = np.argmax(yl,axis=1)
ct = np.argmax(yt,axis=1)

transformer = decomposition.PCA(n_components=40).fit(xl)

xrl = transformer.transform(xl)
xrt = transformer.transform(xt)

maq = discriminant_analysis.QuadraticDiscriminantAnalysis(store_covariance=True).fit(xrl,cl)

print( (maq.predict(xrt) == ct ).mean() )

def classifyG(xs):
    t = np.array(xs).reshape(-1,28*28)
    p = maq.predict_proba(transformer.transform(t))
    r = np.argmax(p,axis=1)
    pm = np.max(p,axis=1)
    return r,pm
########################################################################################


# elegimos la red convolucional
classify = classifyN

# el resto del código es exactamente igual


def binarize(gray):
    _, r = cv.threshold(gray, 128, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return r

def ccs(mask):
    n, cc, st, cen = cv.connectedComponentsWithStats(mask)
    trozos = []
    LMIN=10
    LMAX=100
    for x in range(n):
        if LMIN**2 < st[x][cv.CC_STAT_AREA] < LMAX**2:
            x1 = st[x][cv.CC_STAT_LEFT]
            y1 = st[x][cv.CC_STAT_TOP]
            x2 = st[x][cv.CC_STAT_WIDTH] + x1
            y2 = st[x][cv.CC_STAT_HEIGHT] + y1
            trozos.append( ( cen[x], (cc[y1:y2+1, x1:x2+1]==x).astype(np.uint8)*255 ) )
    return trozos

def extractThings(image):
    g = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if black:
        g = 255-g
    b = binarize(g)  
    return ccs(b)


def center(p):
    r,c = p.shape
    rs = np.outer(range(r),np.ones(c))
    cs = np.outer(np.ones(r),range(c))
    s = np.sum(p)
    my  = np.sum(p*rs) / s
    mx  = np.sum(p*cs) / s
    return mx,my

def adaptsize(x):
    h,w = x.shape
    s = max(h,w)
    h2 = (s-h)//2
    w2 = (s-w)//2
    y = x
    if w2>0:
        z1 = np.zeros([s,w2])
        z2 = np.zeros([s,s-w-w2])
        y  = np.hstack([z1,x,z2])
    if h2>0:
        z1 = np.zeros([h2,s])
        z2 = np.zeros([s-h-h2,s])
        y  = np.vstack([z1,x,z2])
    y = cv.resize(y,(20,20))/255
    mx,my = center(y)
    H = np.array([[1.,0,4-(mx-9.5)],[0,1,4-(my-9.5)]])
    return cv.warpAffine(y,H,(28,28))


black = True

for key, frame in autoStream():

    cosas = extractThings(frame)

    nor = [adaptsize(c) for (x,y), c in cosas]
    
    loc = [ c[0] for c in cosas ]

    t0 = time.time()
    if nor:
        clas,prob = classify(nor)
    else:
        clas,prob = [],[]
    t1 = time.time()

    for (x,y), label,pr in zip(loc, clas, prob):
        col = (0,255,255)
        if pr < 0.5:
            label = '?'
        if pr < 0.9:
            col = (0,160,160)
        putText(frame, str(label), (int(x),int(y)), color=col, div=1, scale=2, thickness=2)


    caben = frame.shape[1]//28
    for k,x in enumerate(nor[:caben]):
        frame[-28:,28*k:28*(k+1),:] = x.reshape(28,28,1)*255

    cv.imshow('digits',frame)

cv.destroyAllWindows()


