#!/usr/bin/env python

# 3) Construimos el clasificador gaussiano con reducción PCA

import cv2 as cv
import numpy as np
from umucv.stream import autoStream
from umucv.util   import putText
import time

########################################################################################

# nos apoyamos en scikit-learn
from sklearn import decomposition, discriminant_analysis

# sacamos los ejemplos de entrenamiento de keras

from tensorflow.keras.datasets import mnist

(kxl,cl), (kxt,ct) = mnist.load_data()
xl = kxl.reshape(len(kxl),-1)/255
xt = kxt.reshape(len(kxt),-1)/255



# fabricamos la función de reducción de dimensión
transformer = decomposition.PCA(n_components=40).fit(xl)

# reducimos la dimensión de los ejemplos
xrl = transformer.transform(xl)
xrt = transformer.transform(xt)

# entrenamos la máquina de clasificación con los ejemplos reducidos
maq = discriminant_analysis.QuadraticDiscriminantAnalysis(store_covariance=True).fit(xrl,cl)

# imprimimos la tasa de aciertos sobre los ejemplos de test
print( (maq.predict(xrt) == ct ).mean() )

# construimos una función que clasifica un conjunto de imágenes normalizadas
def classifyG(xs):
    # las ponemos como una matriz, donde cada fila es una imagen
    t = np.array(xs).reshape(-1,28*28)
    # esto nos da la clase más probable
    # r = maq.predict(transformer.transform(t))
    # pero es mejor hacer lo siguiente:
    
    # esto nos da las probabilidades de cada una
    p = maq.predict_proba(transformer.transform(t))
    # de ellas deducimos la más probable
    r = np.argmax(p,axis=1)
    pm = np.max(p,axis=1)
    # para cada objeto devolvemos la clase mas probable y su probabilidad
    # esto nos permite rechazar decisiones
    return r,pm

########################################################################################


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
        z1 = np.zeros([s,w2],np.uint8)
        z2 = np.zeros([s,s-w-w2],np.uint8)
        y  = np.hstack([z1,x,z2])
    if h2>0:
        z1 = np.zeros([h2,s],np.uint8)
        z2 = np.zeros([s-h-h2,s],np.uint8)
        y  = np.vstack([z1,x,z2])
    y = cv.resize(y,(20,20))
    mx,my = center(y)
    H = np.array([[1.,0,4-(mx-9.5)],[0,1,4-(my-9.5)]])
    return cv.warpAffine(y,H,(28,28))/255


black = True

for key, frame in autoStream():

    cosas = extractThings(frame)

    nor = [adaptsize(c) for (x,y), c in cosas]
    
    # extraemos las posiciones
    loc = [ c[0] for c in cosas ]

    # clasificamos los objetos midiendo el tiempo
    t0 = time.time()
    if nor:
        clas,prob = classifyG(nor)
    else:
        clas,prob = [],[]
    t1 = time.time()


    # recorremos las posiciones de cada objeto y escribimos ahí la clase predicha
    # tenemos también la probabilidad de la decisión
    # si no estamos seguros atenuamos el color o ponemos "?"
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


