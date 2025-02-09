import numpy as np
import cv2   as cv
from threading import Thread
from umucv.util import Clock, putText, findParent
import datetime
import time
import argparse
import sys

cv.setNumThreads(1)

def mkStream(sz=None, dev='0', loop=False):
    if dev=='help':
        print(helpstring)
        sys.exit(0)


    if dev[:5] == 'glob:':
        dev = dev[5:]
        import glob
        files = sorted(glob.glob(dev))
        if not files:
            print('Empty source!')
            sys.exit(1)
        for f in files:
            yield cv.imread(f,cv.IMREAD_COLOR)


    elif dev[:4] == 'dir:':
        dev = dev[4:]
        import glob
        import time
        files = sorted(glob.glob(dev))
        if not files:
            print('Empty source!')
            sys.exit(1)
            #images = [ np.random.randint(256, size=(480,640,3), dtype= np.uint8) ]
        
        images = [ cv.imread(f,cv.IMREAD_COLOR) for f in files ]
        icons  = [ None for f in files ]
        
        k = 0
        n = len(images)

        def fun(event, x, y, flags, param):
            nonlocal k
            #print(event)
            if event == cv.EVENT_LBUTTONDOWN:
                k = (k+1)%n
            if event == cv.EVENT_RBUTTONDOWN:
                k = (k-1)%n

        cv.namedWindow(dev,cv.WINDOW_NORMAL| cv.WINDOW_GUI_NORMAL )
        cv.resizeWindow(dev,300,200)

        cv.setMouseCallback(dev, fun)
        while True:
            if icons[k] is None:
                h,w = images[k].shape[:2]
                icons[k] = cv.resize(images[k], (int(200*w/h),200) )
                putText(icons[k], files[k].split('/')[-1])
            cv.imshow(dev, icons[k])
            yield images[k].copy()
            time.sleep(0.1)


    elif dev[:9] == 'snapshot:':
        import time
        dev = dev[9:]
        t,dev = dev.split(':',1)
        t = int(t)
        for frame in mkShot(dev,timeout=3,debug=False):
            yield frame
            time.sleep(t)


    elif dev[:5] == 'mjpg:':
        try:
            import urllib.request as url
        except:
            import urllib as url
        stream=url.urlopen(dev[5:])
        bytes=b''
        okread = False
        while True:
            bytes+=stream.read(1024)
            if(len(bytes)==0):
                if loop and okread:
                    stream=url.urlopen(dev[5:])
                    bytes=b''
                else:
                    break
            else:
                a = bytes.find(b'\xff\xd8')
                b = bytes.find(b'\xff\xd9')
                if a!=-1 and b!=-1:
                    jpg = bytes[a:b+2]
                    bytes= bytes[b+2:]
                    i = cv.imdecode(np.fromstring(jpg, dtype=np.uint8),cv.IMREAD_COLOR)
                    okread = True
                    yield i

    else:
        if dev in ['0','1','2','3','4']:
            dev = int(dev)
            try:
                cap = cv.VideoCapture(dev, cv.CAP_V4L)
                if not cap.isOpened():
                    cap = cv.VideoCapture(dev)    
            except:
                cap = cv.VideoCapture(dev)
        
        elif dev[:3] == 'gs:':
            cap = cv.VideoCapture(dev[3:], cv.CAP_GSTREAMER)
        
        else:
            cap = cv.VideoCapture(dev)

        if not cap.isOpened():
            print(f'Cannot open source "{dev}"')
            sys.exit(1)

        if sz is not None:
            w,h = sz
            cap.set(cv.CAP_PROP_FRAME_WIDTH,w)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT,h)
        w   = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        h   = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv.CAP_PROP_FPS)
        print(f'{w:.0f}x{h:.0f} {fps}fps')
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                if loop:
                    cap = cv.VideoCapture(dev)
                else:
                    break


def withKey(stream, t=1):
    pausa = False
    key = 0
    exit = False
    for frame in stream:
        while True:
            key = cv.waitKey(t) & 0xFF
            if key == 27 or key == ord('q'):
                exit = True
                break
            if key == ord('.'):
                t = 1-t
            if key == 32:
                pausa = not pausa
                if pausa:
                    frozen = frame.copy()            
            if pausa:
                yield key, frozen.copy()
            else:
                yield key, frame
            if key == ord('s'):
                fname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                if frame.ndim == 3:
                    cv.imwrite(fname+'.png',frame)
                else:
                    cv.imwrite(fname+'.png',cv.cvtColor(frame,cv.COLOR_GRAY2RGB))
            if not pausa: break
        if exit: break



typical = {'CGA':'320x200', 'QVGA':'320x240' ,'VGA':'640x480', 'PAL':'768x576', 'SVGA':'800x600',
           'XGA':'1024x768', '720':'1280x720', 'HD':'1920x1080'}

def isize(s):
    s = typical.get(s,s)
    k = s.find('x')
    return (int(s[:k]),int(s[k+1:]))

def sourceArgs(p):
    p.add_argument('--dev', type=str, default='None', help='image source')
    p.add_argument('--size', help='desired image size', type=isize, default=None)
    p.add_argument('--resize', help='force image size', type=isize, default=None)
    p.add_argument('--step', help='frame by frame', action='store_true')
    p.add_argument('--loop', help='repeat video forever', action='store_true')


import os
import re

def replace_env_variables(text: str) -> str:
    def replacer(match):
        var_name = match.group(1)
        value = os.getenv(var_name)
        if value is None:
            print(f"Warning: Environment variable '{var_name}' not found.")
            return f"${var_name}"
        return value
    
    return re.sub(r'\$(\w+)', replacer, text)

# Example usage:
#input_text = "Path: $HOME, User: $USER, Missing: $UNKNOWN"
#print(replace_env_variables(input_text))



def readAlias():
    aliasfile = findParent('alias.txt')
    try:
        with open(aliasfile, 'r') as f:
            x = f.read()
    except:
        return {'default':'0'}
    lines = [ l.strip() for l in x.split('\n')]
    lines = [ l for l in lines if len(l)>0 and l[0] != '#' ]
    D = {k.strip(): v.strip() for k,v in map(lambda l: l.split('=',1), lines) }
    for k in D.keys():
        D[k] = replace_env_variables(D[k]).format(**D)
    if 'default' not in D:
        D['default'] = '0'
    return D


original_shape = [None]


def mkResize(args):

    def noresize(x):
        original_shape[0] = x.shape
        return x

    if args.resize is None:
        return noresize
    
    wd,hd = args.resize

    def resize(x):
        nonlocal wd, hd
        original_shape[0] = x.shape
        h, w  = x.shape[:2]
        if h!=hd or w!=wd:
            if hd==0:
                hd = int(h/w*wd)
            if wd==0:
                wd = int(w/h*hd)
            return cv.resize(x, (wd,hd))
        else:
            return x
    
    return resize


def autoStream(transf = lambda x: x):
    parser = argparse.ArgumentParser()
    sourceArgs(parser)
    args, _ = parser.parse_known_args()
    
    D = readAlias()
    dev = D.get(args.dev, D['default'])

    if ' ' in dev:
        dev, other = dev.split(' ',2)    
        other = other.split(' ')
        args, _ = parser.parse_known_args(other + sys.argv)
    
    resize = mkResize(args)
    
    stream = transf( map(resize, mkStream(args.size, dev, args.loop) ) )
    return withKey(stream, 0 if args.step else 1)




def mkShot(ip, user=None, password=None, timeout=1, retries=3, debug=False):
    import requests
    while True:
        img = None
        for _ in range(retries):
            try:
                t0 = time.time()
                if user is None:
                    imgResp = requests.get(ip, timeout=timeout)
                else:
                    auth = requests.auth.HTTPDigestAuth(user, password)
                    imgResp = requests.get(ip, timeout=timeout, auth=auth)
                img = cv.imdecode(np.array(bytearray(imgResp.content),dtype=np.uint8),cv.IMREAD_COLOR)
                t1 = time.time()
                if debug: print(f'{(t1-t0)*1000:.0f}ms')
                break
            except:
                if debug: print('timeout')
        else:
            break
        yield img



class Camera:
    def __init__(self, size=None, dev=None, debug=False, transf = lambda x: x):
        self.clock = Clock()
        parser = argparse.ArgumentParser()
        sourceArgs(parser)
        args, _ = parser.parse_known_args()
        if size is None: size = args.size
        if dev  is None:
            dev  = args.dev
            D = readAlias()
            dev = D.get(args.dev, args.dev)
        resize = mkResize(args)
        self.stream = transf(map(resize,mkStream(size,dev)))
        self.frame = None
        self.time  = 0
        self.goon  = True
        self.drop  = False
        
        def fun():
            while self.goon:
                if self.drop:
                    next(self.stream)
                    if debug:
                        print('Frame dropped: {:.0f}'.format(self.clock.time()))
                else:    
                    self.frame = next(self.stream)
                    t = self.clock.time()
                    dt = t - self.time
                    self.time = t
                    if debug:
                        print('Frame ready: {:.0f} ({:.0f})'.format(self.time,dt))
        
        t = Thread(target=fun,args=())
        t.start()
        while self.frame is None: pass
    
    def stop(self):
        self.goon = False 


helpstring = """
----------------------------------------------------------------------
autostream options:

    --dev=0       webcam (/dev/video0)  (default)
    --dev=1       webcam (/dev/video1)
        ...

    --dev=path/to/video.mp4   [--loop]

    --dev=glob:/path/to/images*.*    (list of images, no loop)

    --dev=dir:/path/to/images*.*     (list of images, selected in gui)

    --dev=url/of/stream (rstp, mjpg, etc.)

    --dev=snapshot:15:url   (single image from IP cams, every 15 seconds)
 
    --dev=alias   (defined in alias.txt)

    --size=800x600     (requested to source)

    --resize=320x0     (forced, 0=keep ratio)

GUI options

    ESC          exit
    SPACE        pause (client still running with the same frame)
    .            toggle frame by frame, advance with 'n', (client waits)
    --step       start in frame by frame mode
    s            save input image
----------------------------------------------------------------------
"""

