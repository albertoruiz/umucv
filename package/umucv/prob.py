from collections import Counter
from functools import reduce
import types
import matplotlib.pyplot as plt
import numpy as np


def concat(a,b):
    if type(a) != tuple:
        a = (a,)
    if type(b) != tuple:
        b = (b,)
    return sum((a,b),())

def is_func(obj):
    return isinstance(obj, (types.FunctionType, types.LambdaType))


class P(Counter):
    def __init__(self, d, norm=True):
        super().__init__(d)
        if norm:
            self.normalize()
    
    def normalize(self):
        s = sum(self.values())
        for key in self:
            self[key] /= s
    
    def marginal(self, f):
        return P(sum([Counter({f(x): v}) for x,v in self.items()], Counter({})), norm=False)

    def conditional(self, p):
        return P({x:v for x,v in self.items() if p(x)})

    def __add__(self,p):
        return joint2(p,self).marginal(sum)
    
    def __and__(self,p):
        return joint2(self,p)
    
    def __rand__(self,p): 
        return joint2(p,self)
    
    def __or__(self,p):
        return self.conditional(p)
    
    def transform(self, p ):
         return joint2(p,self).marginal(lambda x: x[0])

    def prob(self, p):
        return self.marginal(p)[True]

    def __rlshift__(self, f):
        return self.marginal(f)
    
    def __rshift__(self, f):
        return self.marginal(f)

    def sample(self,n):
        v,p = zip(*self.items())
        try:
            return np.random.choice(v,p=p,size=n)
        except:
            j = np.arange(len(v))
            s = np.random.choice(j,p=p,size=n)
            return [v[k] for k in s]

    def __repr__(self):
        s = ''
        for k in self:
            s+='{:5.1f}%  {}\n'.format(100*self[k],k)
        return s

    def mean(self):
        return np.dot(*zip(*self.items()))

    def median(self):
        x,v = zip(*sorted(self.items()))
        k = sum(np.cumsum(v)<0.5)
        return x[k]
    
    def mode(self):
        return self.most_common(1)[0][0]
    
    
def is_constant(x):
    return isinstance(x, P)

def joint2(p,q):
        
    if is_constant(q):
        if is_constant(p): F = lambda x: p
        elif is_func(p):   F = p
        else:              F = lambda x: p[x]
        return P({ concat(x,y): u*v for y,v in q.items() for x,u in F(y).items() }, norm=False)
    
    if is_func(q): G = q
    else: G = lambda x: q[x]
    return P({ concat(x,y): u*v for x,u in p.items() for y,v in G(x).items() }, norm=False)



def joint(l):
    L = list(l)
    if len(L) == 1:
        return L[0].marginal(lambda x: (x,))
    else:
        return reduce(joint2,L)


def show(p,alpha=1, ticks=False, rotation=0, edgecolor='black', color=None):
    x,y = zip(*sorted(p.items()))
    x1,x2,*_ = x
    try:
        plt.bar(x,y,width=(x2-x1)*1, alpha=alpha, edgecolor=edgecolor, color=color)
        if ticks: plt.xticks(x)
    except TypeError:
        z = np.arange(len(x))
        plt.bar(z,y,width=1, alpha=alpha,edgecolor=edgecolor, color=color)
        plt.xticks(z,x,rotation=rotation)


def hdi(p, prob):
    r = list(sorted(p.items()))
    s = 0
    while True:
        j = min([(r[0][1],0), (r[-1][1],-1)])[1]
        if s + r[j][1] > 1-prob:
            break
        s += r[j][1]
        del r[j]
    x1 = r[0][0]
    x2 = r[-1][0]
    return x1, x2


def showhdi(p,percent):
    _,_,y1,y2 = plt.axis()
    delta = (y2-y1)/10
    x1,x2 = hdi(p,percent/100)
    if not isinstance(x1,int):
        sx1, sx2 = '{:.2f}'.format(x1) , '{:.2f}'.format(x2)
    else:
        sx1, sx2 = x1, x2
    y = y1+delta
    plt.plot([x1,x2],[y,y],color='black',lw=3);
    plt.text(x1,y+delta/2,sx1,horizontalalignment='center');
    plt.text(x2,y+delta/2,sx2,horizontalalignment='center');

## in db
def evidence(p):
    return 10 * np.log10(p/(1-p))


#########################################################################

from numpy.linalg import inv
from numpy.linalg import slogdet

def extractVars(js,x):
    if x.ndim == 1:
        return x[js]
    return x[js][:,js]

class G():
    def __init__(self,m,c=None,ic=None):
        self.m = np.array(m)
        self.dim = len(m)
        if ic is not None:
            self.ic = np.array(ic)
            self.c  = inv(ic)
            return
        if c is not None:
            self.c = np.array(c)
            self.ic = inv(c)
            return
        self.c  = np.eye(len(m))
        self.ic = np.eye(len(m))
    
    def marg(self,js):
        m = extractVars(js,self.m)
        c = extractVars(js,self.c)
        return G(m,c)
    
    def cond(self,y):
        ny = len(y)
        nx = len(self.m) - ny
        mx = self.m[:nx]
        my = self.m[nx:]
        cxx = self.c[:nx][:,:nx]
        cyy = self.c[nx:][:,nx:]
        cxy = self.c[:nx][:,nx:]
        cyx = cxy.T
        r = cxy @ inv(cyy)
        m = mx + r @ (y-my)
        c = cxx - r @ cyx
        return G(m,c)
    
    # a linear transformation of x is convolved with (suffers additive noise)
    # with offset o and covariance r 
    # x ~ N m c,   y|x ~ N (h x + o,  r)
    def jointLinear(self, h, n):
        h = np.array(h)
        o = n.m
        r = n.c
        m = np.append(self.m, h @ self.m + o)
        cxy = self.c @ h.T
        cyy = h @ cxy + r
        c = np.bmat([[self.c  , cxy],
                     [cxy.T   , cyy]])
        return G(m, np.array(c))
    
    
    # direct, alternative method
    def bayesGaussianLinear(self,h,n,y):
        h = np.array(h)
        c = inv(self.ic + h.T @ n.ic @ h)
        m = c @ (h.T @ n.ic @ (y-n.m) + self.ic @ self.m)
        return G(m,c)
    
    
    # kalman-style method
    def bayesGaussianLinearK(self,h,n,y):
        h = np.array(h)
        k = self.c @ h.T @ inv (h @ self.c @ h.T + n.c)
        c = (np.eye(len(self.m)) - k @ h) @ self.c
        m = self.m + k @ (y - h @ self.m - n.m)
        return G(m,c)

    
    def logprob(self):
        l1 = -self.dim/2 * np.log(2*np.pi)
        l2 = -1/2 * slogdet(self.c)[1]
        l3 = lambda x: -0.5* (x-self.m) @ self.ic @ (x-self.m)
        return lambda x: l1 + l2 + l3(x)
    
    def ellipse(self):
        assert self.dim == 2
        
        l,v = np.linalg.eigh(self.c)

        sl1 = np.sqrt(l[0])
        sl2 = np.sqrt(l[1])
        v1 = v[:,0]
        v2 = v[:,1]

        CIR = np.array([ [np.cos(t), np.sin(t)] for t in np.linspace(0,2*np.pi, 50) ])

        def rot(a):
            c = np.cos(a)
            s = np.sin(a)
            return np.array([[c,-s],
                             [s, c]])

        return self.m + CIR @ np.diag([2*sl2,2*sl1]) @ rot(-np.arctan2(v2[1],v2[0]))

