import numpy as np
from sempy.mass import reference_mass_matrix_3d,\
    reference_mass_matrix_2d
from sempy.derivative import reference_derivative_matrix

def gradient(U,n):
    D=reference_derivative_matrix(n-1)

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

def gradient_2d(U,n):
    D=reference_derivative_matrix(n-1)
    nn=n*n

    V=U.reshape(n,n)
    Ur=np.dot(V,D.T)

    V=U.reshape(n,n)
    Us=np.dot(D,V)

    return Ur.reshape((nn,)),Us.reshape((nn,))

def gradient_transpose(Wx,Wy,Wz,n):
    D=reference_derivative_matrix(n-1)

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

def gradient_transpose_2d(Wx,Wy,n):
    D=reference_derivative_matrix(n-1)

    nn=n*n

    V=Wx.reshape(n,n)
    Ur=np.dot(V,D)

    V=Wy.reshape(n,n)
    Us=np.dot(D.T,V)

    return Ur.reshape((nn,))+Us.reshape((nn,))
