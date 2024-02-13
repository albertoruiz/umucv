import cv2          as cv
import numpy        as np
import numpy.linalg as la
from umucv.htrans import htrans,row,jr,jc,col,sepcam,inhomog

import time
import datetime
from collections import deque

import atexit

atexit.register(cv.destroyAllWindows)


class mkParam:
    def __init__(self,wname,width=500):
        self.wname = wname
        cv.namedWindow(wname,cv.WINDOW_NORMAL)
        cv.resizeWindow(wname,width,5)
    
    def supdate(self,pname,v):
            exec('self.{}={}'.format(pname,v))
    
    def addParam(self,pname,v,vmax,action=None):
        def update(v):
            self.supdate(pname,v)
        if not action:
            action = update
        cv.createTrackbar(pname, self.wname, v, vmax, action)
        action(v)


def putText(img, string, orig=(5,16), color=(255,255,255), div=2, scale=1, thickness=1):
    (x,y) = orig
    if div > 1:
        (w,h), b = cv.getTextSize(string, cv.FONT_HERSHEY_PLAIN, scale, thickness)
        img[y-h-4:y+b, x-3:x+w+3] //= div
    cv.putText(img, string, (x,y), cv.FONT_HERSHEY_PLAIN, scale, color, thickness, cv.LINE_AA)



class ROI2:
    """
    if self.roi:
        [x1,y1,x2,y2] = self.roi
        (x1,y1) top left
        (x2,y2) bottom right
        all coords are included in the roi
    """
    def __init__(self,window):
        self.roi = []
        self.DOWN = False
        self.xt = True
        self.yt = True
        
        def fun(event, x, y, flags, param):
            if   event == cv.EVENT_LBUTTONDOWN:
                self.roi = [x,y,x,y]
                self.DOWN = True
            elif event == cv.EVENT_LBUTTONUP:
                self.DOWN = False
            elif self.DOWN:
                x1,y1,x2,y2 = self.roi
                if self.xt:
                    if x > x1:
                        x2 = x
                    else:
                        x1 = x
                        self.xt = False
                else:
                    if x < x2:
                        x1 = x
                    else:
                        x2 = x
                        self.xt = True
                if self.yt:
                    if y > y1:
                        y2 = y
                    else:
                        y1 = y
                        self.yt = False
                else:
                    if y < y2:
                        y1 = y
                    else:
                        y2 = y
                        self.yt = True
                self.roi = [x1,y1,x2,y2]
                
        cv.setMouseCallback(window, fun)



class ROI:
    def __init__(self,window):
        self.roi = []
        self.DOWN = False
        def poly():
            x1,y1,x2,y2 = self.roi
            self.box = np.array([[x1-1,y1-1],[x1-1,y2],[x2,y2],[x2,y1-1]])
        def fun(event, x, y, flags, param):
            if   event == cv.EVENT_LBUTTONDOWN:
                self.roi = [x,y,x+1,y+1]
                poly()
                self.DOWN = True
            elif event == cv.EVENT_LBUTTONUP:
                self.DOWN = False
            elif self.DOWN:
                x1,y1,_,_ = self.roi 
                x2 = x+1
                y2 = y+1
                self.roi = [min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)]
                poly()
        cv.setMouseCallback(window, fun)


class zoomWindow:
    def __init__(self, wname, W=800, H=600, zink=ord('+'), zoutk=ord('-')):
        self.S = 0
        self.pos = [(0,0)]
        self.W = W
        self.H = H
        self.W2 = W//2
        self.H2 = H//2
        self.wname = wname
        self.zink = zink
        self.zoutk = zoutk
        def fun(event, x, y, flags, param):
            [(X,Y)] = param
            if event == cv.EVENT_LBUTTONDOWN:
                X = X-(x-self.W2)
                Y = Y-(y-self.H2) 
                param[0] = X,Y
        cv.namedWindow(wname)
        cv.setMouseCallback(wname, fun, self.pos)
    def show(self, image):
        s = 2**self.S
        [(X,Y)] = self.pos
        T = np.array([[s,0.,X],
                      [0.,s,Y]])
        scaled = cv.warpAffine(image,T,(self.W,self.H),flags=cv.INTER_NEAREST)
        putText(scaled,f'{2**self.S}x')
        cv.line(scaled, (0,self.H2),(self.W,self.H2),(0,0,255))
        cv.line(scaled, (self.W2,0),(self.W2,self.H),(0,0,255))
        cv.imshow(self.wname, scaled)
    def update(self, key, image=None):
        [(X,Y)] = self.pos
        s = 2**self.S
        W2,H2 = self.W2, self.H2
        if key == self.zink: xc=(W2-X)/s; yc=(H2-Y)/s; self.S += 1; s=2**self.S; self.pos[0]=(W2-xc*s, H2-yc*s)
        if key == self.zoutk: xc=(W2-X)/s; yc=(H2-Y)/s; self.S -= 1; s=2**self.S; self.pos[0]=(W2-xc*s, H2-yc*s)
        if key == ord('0'): self.pos[0]=(0,0); self.S=0
        if image is not None:
            self.show(image)


class Video:
    def __init__(self, ext='mp4', fourcc=None, codec='X264', fps=None):
        self.ext = ext
        self.codec = codec
        if fourcc == None:
            fourcc = cv.VideoWriter_fourcc(*self.codec)
        self.fourcc = fourcc
        self.times = deque(maxlen=20)
        self.ON = False
        self.Open = False
        self.nf = 0
        self.fps = fps
    def write(self, frame, key=0, keypressed=ord('v')):
        if key==keypressed:
            self.ON = not self.ON
            msg = {True:"recording",False:"paused"}
            print(f'video: {msg[self.ON]}')
        self.times.append(time.time())
        if self.ON:
            if not self.Open:
                h,w = frame.shape[:2]
                #print(self.times)
                if self.fps is None:
                    ts = np.array(self.times)
                    deltas = ts[1:] - ts[:-1]
                    #print(deltas)
                    fps = int(0.5 + 1/deltas.mean())
                else:
                    fps = self.fps
                fname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.") + self.ext
                self.out = cv.VideoWriter(fname, self.fourcc, fps, (w,h))
                self.Open = True
                print(f'recording {self.codec} video {fname} with resolution {w}x{h} at {fps}fps')
            self.out.write(frame)
            self.nf += 1
    def release(self):
        if self.Open:
            self.out.release()
            print(f'{self.nf} frames written to video')



videos = dict()

def imshow(winname, image, key, code=ord('v')):
    if winname not in videos:
        videos[winname] = Video()
        atexit.register(videos[winname].release)
    cv.imshow(winname, image)
    videos[winname].write(image, key, code)



class Help:
    
    def __init__(self, s, scale=1.5, thickness=1):
        import numpy as np
        from umucv.util import putText
        
        self.active = False
        lines = s.strip().split('\n')
        

        wt, ht = 0, 0        
        for l in lines:
            (w,h), b = cv.getTextSize(l, cv.FONT_HERSHEY_PLAIN, scale, thickness)
            wt = max(wt,w)
            ht = max(ht,h)

        #print(wt,ht)

        dh = ht*2
        H = np.zeros((dh*(1+len(lines))-ht,wt +2*ht),np.uint8) + 64
  
        for k, l in enumerate(lines):        
            putText(H, l, orig= (ht, dh*(k+1)), color=128+64, div=1, scale=scale, thickness=thickness)
        self.himg = H
    
    def show(self):
        cv.namedWindow('Help',cv.WINDOW_GUI_NORMAL | cv.WINDOW_AUTOSIZE)
        cv.imshow('Help',self.himg)
        self.active = True
    
    def show_if(self, key, chosen):
        if key !=chosen:
            return
        if not self.active:
            self.show()
        else:
            cv.destroyWindow('Help')
            self.active = False



try:
    lineType = cv.LINE_AA
except AttributeError:
    lineType = 0

def showCalib(K,image,dg=5,col=(128,128,128)):
    HEIGHT, WIDTH = image.shape[:2]
    cv.line(image,(0,HEIGHT//2),(WIDTH,HEIGHT//2),col,1,lineType)
    cv.line(image,(WIDTH//2,0),(WIDTH//2,HEIGHT),col,1,lineType)
    AW = WIDTH*1//100
    for ang in range(-45,46,dg):
        alpha = np.radians(ang)
        pos3d = [np.sin(alpha), 0, np.cos(alpha)]
        x,_ = inhomog(K @ pos3d).astype(int)
        x += HEIGHT//2 - WIDTH//2
        cv.line(image,(WIDTH//2-AW,x),(WIDTH//2+AW,x),col,1,lineType)
        x += -HEIGHT//2 + WIDTH//2
        cv.line(image,(x,HEIGHT//2-AW),(x,HEIGHT//2+AW),col,1,lineType)



cube = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [0,0,0],
    
    [0,0,1],
    [1,0,1],
    [1,1,1],
    [0,1,1],
    [0,0,1],
        
    [1,0,1],
    [1,0,0],
    [1,1,0],
    [1,1,1],
    [0,1,1],
    [0,1,0]
    ])

ejeX = np.array([[0.,0,0],[1,0,0]])    
ejeY = np.array([[0.,0,0],[0,1,0]])
ejeZ = np.array([[0.,0,0],[0,0,1]])

def augmented(img,camera,thing,color,thick=1):
    cv.drawContours(img, [htrans(camera,thing).astype(int)] , -1, color, thick, lineType)

def showAxes(img,camera,scale=1):
    p = camera
    s = scale
    augmented(img,p,ejeX*s, (0,0,255), 1)
    augmented(img,p,ejeY*s, (0,210,0), 1)
    augmented(img,p,ejeZ*s, (255,0,0), 1)
    for axis,name,color in zip(htrans(p,np.eye(3)*s).astype(int), ['x','y','z'],[(0,0,255),(0,210,0),(255,0,0)]):
        putText(img,name,axis,color,div=1)


import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)
    
def digits(n):
    return printoptions(precision=n, suppress=True)



# esquema en 3d de una cámara
def cameraOutline(M, sc=0.3):

    K,R,C = sepcam(M)
    
    # formamos una transformación 3D para mover la cámara en el origen a la posición de M
    rt = jr(jc(R, np.dot(-R , col(C))),
            row(0,0,0,1))
    
    x = 1
    y = x
    z = 0.99
    
    ps =[x,    0,    z,
         (-x), 0,    z,
         0,    0,    z,
         0,    1.3*y,z,
         0,    (-y), z,
         x,    (-y), z,
         x,    y,    z,
         (-x), y,    z,
         (-x), (-y), z,
         x,    (-y), z,
         x,    y,    z,
         0,    y,    z,
         0,    0,    z,
         0,    0,    0,
         1,    1,    z,
         0,    0,    0,
         (-1), 1,    z,
         0,    0,    0,
         (-1), (-1), z,
         0,    0,    0,
         (1), (-1),  z,
         0,    0,    0,
         0,    0,    (2*x)]
    
    ps = np.array(ps).reshape(-1,3)
    return htrans(la.inv(rt), sc * ps) #   ps @ la.inv(rt).transpose())


# otra versión
def cameraOutline2(M, sc = 0.3):

    K,R,C = sepcam(M)
    
    # formamos una transformación 3D para mover la cámara en el origen a la posición de M
    rt = jr(jc(R, -R @ col(C)),
            row(0,0,0,1))
    
    x = 1;
    y = x;
    z = 0.99;
    
    ps =[x,    0,    z,
         (-x), 0,    z,
         0,    0,    z,
         0,    1.3*y,z,
         0,    (-y), z,
         x,    (-y), z,
         x,    y,    z,
         (-x), y,    z,
         (-x), (-y), z,
         x,    (-y), z,
         x,    y,    z,
         0,    y,    z,
         0,    0,    z,
         0,    0,    0,
         1,    1,    z,
         0,    0,    0,
         (-1), 1,    z,
         0,    0,    0,
         (-1), (-1), z,
         0,    0,    0,
         (1), (-1),  z,
         0,    0,    0,
         0,    0,    (2*x)]
    
    ps = np.array(ps).reshape(-1,3)
    return htrans(la.inv(rt), sc * ps)



try:
    from matplotlib.pyplot import plot
except ImportError:
    pass


# muestra un polígono cuyos nodos son las filas de un array 2D
def shcont(c, color='blue', nodes=True):
    x = c[:,0]
    y = c[:,1]
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    plot(x,y,color)
    if nodes: plot(x,y,'.',color=color,ms=10)
    

# dibuja una recta "infinita" dadas sus coordenadas homogéneas
def shline(l, color='red', xmin=-2000,xmax=2000):
    a,b,c = l / la.norm(l)
    if abs(b) < 1e-6:
        x = -c/a
        r = np.array([[x,xmin],[x,xmax]])
    else:
        y0 = (-a*xmin - c) / b
        y1 = (-a*xmax - c) / b
        r = np.array([[xmin,y0],[xmax,y1]])
    shcont(r,color=color,nodes=False)


def warpOn(dst,src,pts):
    # ul,ur,dl,dr = pts
    h,w = src.shape[:2]
    H,W = dst.shape[:2]
    psrc = np.array([[0,0],[w,0],[0,h],[w,h]]).astype(np.float32)
    pdst = pts.astype(np.float32)
    T = cv.getPerspectiveTransform(psrc,pdst)
    cv.warpPerspective(src,T,(W,H),dst,0,cv.BORDER_TRANSPARENT)

import time

class Clock:
    def __init__(self):
        self.t0 = time.time()
    def time(self):
        t = time.time()
        return 1000*(t-self.t0)

##########################################################################

def mkCov2p(p1,p2,thick=0.2):
    d = np.array([p1,p2])
    m = np.mean(d,axis=0)
    c = np.cov(d,rowvar=False)
    c = c + np.eye(len(p1)) * la.norm(np.array(p1)-np.array(p2))*thick
    return m,c

u = np.linspace(0.0, 2.0 * np.pi, 20)
v = np.linspace(0.0, np.pi, 20)
x_0 = np.outer(np.cos(u), np.sin(v))
y_0 = np.outer(np.sin(u), np.sin(v))
z_0 = np.outer(np.ones_like(u), np.cos(v))

def ellip3d(mc):
    m,c = mc
    l,r =la.eigh(c)
    tran = np.diag(np.sqrt(abs(l))) @ r.T
    return (np.array([x_0.flatten(),y_0.flatten(),z_0.flatten()]).T @ tran + m).T.reshape(3,len(u),len(v))

