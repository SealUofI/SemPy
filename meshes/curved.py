import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from quadrature import gauss_lobatto
from interpolation import lagrange

def trapezoid():
    N=1
    M=10
    
    zn,wn=gauss_lobatto(N)
    zm,wm=gauss_lobatto(M)
    J=lagrange(zm,zn)
    
    x=np.array([-0.5,0.5,-1.0/sqrt(2.0),1.0/sqrt(2.0),-0.5,0.5,-1.0/sqrt(2.0),1.0/sqrt(2.0)])
    y=np.array([ 0.5,0.5, 1.0/sqrt(2.0),1.0/sqrt(2.0), 0.5,0.5, 1.0/sqrt(2.0),1.0/sqrt(2.0)])
    z=np.array([ 0.0,0.0, 0.0          ,0.0          , 0.1,0.1, 0.1          ,0.1          ])
    
    n=N+1
    nn=n*n
    m=M+1
    mm=m*m
    
    x=x.reshape((nn,n))
    y=y.reshape((nn,n))
    z=z.reshape((nn,n))
    Jx=np.dot(x,J.T)
    Jy=np.dot(y,J.T)
    Jz=np.dot(z,J.T)
    
    Jx=Jx.reshape((n,n,m))
    Jy=Jy.reshape((n,n,m))
    Jz=Jz.reshape((n,n,m))
    JJx=np.zeros((n,m,m))
    JJy=np.zeros((n,m,m))
    JJz=np.zeros((n,m,m))
    for i in range(n):
        JJx[i,:,:]=np.dot(J,Jx[i,:,:])
        JJy[i,:,:]=np.dot(J,Jy[i,:,:])
        JJz[i,:,:]=np.dot(J,Jz[i,:,:])
    
    X=JJx
    Y=JJy
    Z=JJz
    
    for j in range(n):
        for i in range(m):
            Y[j,i,:]=Y[j,i,:]+(Y[j,i,:]-Y[j,0,:])*(X[j,i,0]*X[j,i,0]-np.multiply(X[j,i,:],X[j,i,:]))
    
    Jx=JJx.reshape((n,m*m))
    Jy=JJy.reshape((n,m*m))
    Jz=JJz.reshape((n,m*m))
    X=np.dot(J,Jx)
    Y=np.dot(J,Jy)
    Z=np.dot(J,Jz)
    
    X=X.reshape((m,m,m))
    Y=Y.reshape((m,m,m))
    Z=Z.reshape((m,m,m))
    
    return X,Y,Z
