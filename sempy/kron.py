import numpy as np

def kron_2d(Sy,Sx,U):
    nx,mx=Sx.shape
    ny,my=Sy.shape

    U=U.reshape((my,mx))
    U=np.dot(U,Sx.T)

    V=U.reshape((my,nx))
    U=np.dot(Sy,V)

    return U.reshape((nx*ny,))

def kron(Sz,Sy,Sx,U):
    nx,mx=Sx.shape
    ny,my=Sy.shape
    nz,mz=Sz.shape

    U=U.reshape((my*mz,mx))
    U=np.dot(U,Sx.T)

    U=U.reshape((mz,my,nx))
    V=np.zeros ((mz,ny,nx))
    for i in range(mz):
        V[i,:,:]=np.dot(Sy,U[i,:,:])

    V=V.reshape((mz,nx*ny))
    U=np.dot(Sz,V)

    return U.reshape((nx*ny*nz,))
