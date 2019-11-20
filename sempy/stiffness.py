import numpy as np
from sempy.derivative import reference_derivative_matrix
from sempy.mass import reference_mass_matrix_3D

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

def geometric_factors(X,Y,Z,n):
    Xr,Xs,Xt=gradient(X,n)
    Yr,Ys,Yt=gradient(Y,n)
    Zr,Zs,Zt=gradient(Z,n)

    J=Xr*(Ys*Zt-Yt*Zs)-Yr*(Xs*Zt-Xt*Zs)+Zr*(Xs*Yt-Ys*Xt)

    B=reference_mass_matrix_3D(n-1)
    
    rx=(Ys*Zt-Yt*Zs)/J
    sx=(Yt*Zr-Yr*Zt)/J
    tx=(Yr*Zs-Ys*Zr)/J
    
    ry=-(Zt*Xs-Zs*Xt)/J
    sy=-(Zr*Xt-Zt*Xr)/J
    ty=-(Zs*Xr-Zr*Xs)/J
    
    rz=(Xs*Yt-Xt*Ys)/J
    sz=-(Xr*Yt-Xt*Yr)/J
    tz=(Xr*Ys-Xs*Yr)/J
    
    G11=rx*rx+ry*ry+rz*rz
    G12=rx*sx+ry*sy+rz*sz
    G13=rx*tx+ry*ty+rz*tz
    G22=sx*sx+sy*sy+sz*sz
    G23=sx*tx+sy*ty+sz*tz
    G33=tx*tx+ty*ty+tz*tz

    G=np.zeros((3,3,G11.size))
    G[0,0,:]=G11*B*J
    G[0,1,:]=G12*B*J
    G[0,2,:]=G13*B*J
    G[1,0,:]=G12*B*J
    G[1,1,:]=G22*B*J
    G[1,2,:]=G23*B*J
    G[2,0,:]=G13*B*J
    G[2,1,:]=G23*B*J
    G[2,2,:]=G33*B*J

    return G,J,B
