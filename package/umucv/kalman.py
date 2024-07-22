import numpy as np
import numpy.linalg as la

def kalman(mu,P,F,Q,B,u,z,H,R):
    # mu, P : estado actual y su incertidumbre
    # F, Q  : sistema dinámico y su ruido
    # B, u  : control model y la entrada
    # z     : observación
    # H, R  : modelo de observación y su ruido
    
    mup = F @ mu + B @ u;
    pp  = F @ P @ F.T + Q;

    zp = H @ mup

    # si no hay observación solo hacemos predicción 
    if z is None:
        return mup, pp, zp

    epsilon = z - zp

    k = pp @ H.T @ la.inv(H @ pp @ H.T +R)

    new_mu = mup + k @ epsilon;
    new_P  = (np.eye(len(P))-k @ H) @ pp;
    return new_mu, new_P, zp



def unscentedSamples(m,c):
    n = len(m)
    alpha = 0.5
    beta = 2
    kappa = 3 - n
    lamda = alpha**2 * (n + kappa) - n
    ds = la.cholesky((n+lamda)*c).T
    ws = np.ones(2*n)*(1/2/(n+lamda))
    wm = np.hstack([lamda/(n+lamda), ws])
    wc = np.hstack([lamda/(n+lamda) + 1 - alpha**2 + beta,ws])
    s = np.asarray(np.vstack([m,m+ds,m-ds]))
    return s,wm,wc

def unscentCov(wc,xcen,ycen):
    rx,cx = xcen.shape
    _,cy  = ycen.shape
    xy = np.zeros([cx,cy])
    for k in range(rx):
        xy += wc[k]* xcen[[k],:].T @ ycen[[k],:]
    return xy

def unscentEstimate(s,wm,wc):
    m = np.sum(np.diag(wm)@s,axis=0)
    sc = s-m
    c = unscentCov(wc,sc,sc)
    return m,c


def ukf0(mu,P,F,Q,B,u):
    # predicción sin observación
    # (no predice la observación, mejor usar ukf con None)
    ns = len(mu);
    Z = np.zeros([ns,ns])
    
    mua = np.concatenate([mu,np.zeros(ns)])
    Pa  = np.bmat(
        [[P , Z],
         [Z , Q]])
    
    s,wm,wc = unscentedSamples(mua,Pa)

    b = B(u)
    st = np.array([F(x) + b for x in s[:,:ns]])
    
    return unscentEstimate(st,wm,wc);


def ukf(mu,P,F,Q,B,u,z,H,R):
    # mu, P : estado actual y su incertidumbre
    # F, Q  : sistema dinámico (función general) y su ruido
    # B, u  : control model (también función general) y la entrada
    # z     : observación (puede ser None)
    # H, R  : modelo de observación (función general) y su ruido

    ns = len(mu)
    nz = len(R)
    
    def Z(n,m): return np.zeros([n,m])
    
    mua = np.concatenate([mu, np.zeros(ns+nz)])
    Pa  = np.bmat(
        [[ P        , Z(ns,ns) , Z(ns,nz) ],
         [ Z(ns,ns) , Q        , Z(ns,nz) ],
         [ Z(nz,ns) , Z(nz,ns) , R        ]])
    
    s,wm,wc = unscentedSamples(mua,Pa)
    
    s_x = s[:,:ns]
    s_f = s[:,ns:ns+ns]
    s_r = s[:,ns+ns:]
    
    b = B(u)
    st = np.array([F(x) + b +r for x,r in zip(s_x, s_f)])
    mup,pp = unscentEstimate(st,wm,wc)
    
    sz = np.array([H(x) + r for x,r in zip(st, s_r)])
    muz = np.sum(np.diag(wm) @ sz, axis=0)
    
    szc = sz-muz
    zc = unscentCov(wc,szc,szc)
    
    stc = st-mup
    xz = unscentCov(wc,stc,szc)

    K = xz @ la.inv(zc)

    if z is None:
        return mup, pp, muz

    new_mu = mup + K @ (z-muz)
    new_P  = pp - K @ zc @ K.T

    return new_mu, new_P, muz

