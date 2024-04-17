#!/usr/bin/env python

import numpy as np
import cv2 as cv
import sys
import glob
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--examples', help='folder with the input images and masks', type=str, default='caras')
parser.add_argument('--show', help="show a few cases and don't write the output file", action='store_true')
parser.add_argument('--nomirror', help="don't add flipped examples", action='store_true')
args = parser.parse_args()


def see_masked(img,mask):
    print(img.shape)
    print(mask.shape)
    return np.where(mask,img,0)


filenames = glob.glob(args.examples+"/mask_*.*")

print(filenames)

dataset = [(cv.cvtColor(cv.imread(filename.replace('mask_','')), cv.COLOR_BGR2RGB),
            cv.imread(filename)>128) for filename in filenames]


if args.show:
    plt.imshow(see_masked(*dataset[np.random.randint(len(dataset))]))
    plt.show()


SZ     = 128
EXPAND = 20
MIRROR = not args.nomirror

def extract_region(img, pos, sz):
    # Extraemos un cuadrado de lado sz centrado en pos,
    # desplazánndolo si es necesario para que no se salga

    H,W = img.shape[:2]
    x,y = pos
    r = sz//2
    x = min(max(x,r),W-r)
    y = min(max(y,r),H-r)
    return img[y-r:y+r, x-r:x+r]


def expand(sample, N, size):
    # Generamos N recortes cuadrados de la imagen y su máscara
    # en posiciones aleatorias alrededor de cada componente conexa
    # de la máscara, y otros N más aleatoriamente en toda la imagen

    source, mask = sample
    mask = (mask[:,:,0]*255).astype(np.uint8)
    
    def getSampleAt(x,y):
        size0 = size//2 + np.random.randint(0,size)
        src = extract_region(source,(x,y),size0)
        msk = extract_region(mask,(x,y),size0)
            
        src = cv.resize(src,(size,size))
        msk = cv.resize(msk,(size,size))
        return src,msk

    H,W = mask.shape
    S = min(H,W)/10
    
    n, cc, st, cen = cv.connectedComponentsWithStats(mask)
    #print(n)
    #print(cc)
    samples = []
    
    for x in range(1,n):
        x1 = st[x][cv.CC_STAT_LEFT]
        y1 = st[x][cv.CC_STAT_TOP]
        x2 = st[x][cv.CC_STAT_WIDTH] + x1
        y2 = st[x][cv.CC_STAT_HEIGHT] + y1
        xc = (x1+x2)//2
        yc = (y1+y2)//2
        
        for _ in range(N):
            x,y = (np.random.randn(2) * S + (xc,yc)).astype(int)
            samples.append( getSampleAt(x,y) )
            #if MIRROR:
            #    samples.append((np.fliplr(samples[-1][0]), np.fliplr(samples[-1][1])))

    for _ in range(N):
        x,y = np.random.randint([0,0],[W,H],2)
        samples.append( getSampleAt(x,y) )
        #if MIRROR:
        #    samples.append((np.fliplr(samples[-1][0]), np.fliplr(samples[-1][1])))
            
    return samples


#expandtest = expand(dataset[2],20,128)


training_set = []
for s in dataset:
    training_set += expand(s, EXPAND, SZ)
    if MIRROR:
        sm = (np.fliplr(s[0]), np.fliplr(s[1]))
        training_set += expand(sm, EXPAND, SZ)
    

ax = np.array([s[0] for s in training_set])
print(ax.shape)
ay = np.array([s[1] for s in training_set])
print(ay.shape)


def show_example(s):
        img, mask = s
        r = (img/255)*(mask.reshape(*mask.shape,1)/255)
        return r

if args.show:
    for _ in range(5):
        k = np.random.randint(len(training_set))
        s = training_set[k]
        plt.imshow(show_example(s))
        plt.show()


if not args.show:
    np.savez('samples.npz', x=ax, y=ay)

