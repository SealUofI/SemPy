import numpy as np
from sempy.quadrature import gauss_lobatto

def lagrange(x):
    assert x.ndim==1

    n=x.size

    a=np.ones((n,),dtype=np.float64)

    for i in range(n):
        for j in range(i):
            a[i]=a[i]*(x[i]-x[j])
        for j in range(i+1,n):
            a[i]=a[i]*(x[i]-x[j])
    a=1.0/a
    D=np.zeros((n,n))

    for i in range(n):
        D[i,:]=a[i]*(x[i]-x)
        D[i,i]=1
    for j in range(n):
        D[:,j]=D[:,j]/a[j]
    D=1.0/D

    for i in range(n):
        D[i,i]=0
        D[i,i]=-sum(D[i,:])
    return D

def reference_gradient(U,n):
    z,w=gauss_lobatto(n-1)
    D=lagrange(z)

    nn=n*n
    nnn=nn*n

    V=U.reshape(nn,n)
    Ur=np.dot(V,D.T)

    V=U.reshape(n,n,n)
    Us=np.zeros((n,n,n))
    for i in range(n):
        Us[i,:,:]=np.dot(D,V[i,:,:])

    V=U.reshape(n,nn)
    Ut=np.dot(D,V)

    return Ur.reshape((nnn,)),Us.reshape((nnn,)),Ut.reshape((nnn,))

def reference_gradient_transpose(Wx,Wy,Wz,n):
    z,w=gauss_lobatto(n-1)
    D=lagrange(z)

    nn=n*n
    nnn=nn*n

    V=Wx.reshape(nn,n)
    Ur=np.dot(V,D)

    V=Wy.reshape(n,n,n)
    Us=np.zeros((n,n,n))
    for i in range(n):
        Us[i,:,:]=np.dot(D.T,V[i,:,:])

    V=Wz.reshape(n,nn)
    Ut=np.dot(D.T,V)

    return Ur.reshape((nnn,))+Us.reshape((nnn,))+Ut.reshape((nnn,))
