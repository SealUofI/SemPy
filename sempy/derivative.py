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
    dims=U.shape
    z,w=gauss_lobatto(n-1)
    D=lagrange(z)

    if U.ndim==1:
        return np.dot(D,U)
    if U.ndim==2:
        assert dims[0]==dims[1]
        nn=n*n
        V=U.reshape(n,n)
        Ur=np.dot(D,V)
        Us=np.dot(V,D.T)
        return Ur.reshape((nn,)),Us.reshape((nn,))
    if U.ndim==3:
        assert dims[0]==dims[1]==dims[2]
        nn=n*n
        nnn=nn*n

        V=U.reshape(n,nn)
        Ur=np.dot(D,V)

        V=U.reshape(n,n,n)
        for i in range(n):
            V[i,:,:]=np.dot(D,V[i,:,:])
        Us=V

        V=U.reshape(nn,n)
        Ut=np.dot(V,D.T)

        return Ur.reshape((nnn,)),Us.reshape((nnn,)),Ut.reshape((nnn,))

def reference_gradient_transpose(Wx,Wy,Wz,n):
    z,w=gauss_lobatto(n-1)
    D=lagrange(z)
    D=D.T

    nn=n*n
    nnn=nn*n

    V=Wx.reshape(n,nn)
    Ur=np.dot(D,V)

    V=Wy.reshape(n,n,n)
    for i in range(n):
        V[i,:,:]=np.dot(D,V[i,:,:])
    Us=V

    V=Wz.reshape(nn,n)
    Ut=np.dot(V,D.T)

    return Ur.reshape((nnn,))+Us.reshape((nnn,))+Ut.reshape((nnn,))
