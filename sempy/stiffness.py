import numpy as np
from derivative import reference_gradient,reference_gradient_transpose

def geometric_factors(X,Y,Z):
    Xr,Xs,Xt=reference_gradient(X)
    Yr,Ys,Yt=reference_gradient(Y)
    Zr,Zs,Zt=reference_gradient(Z)
    
    J=Xr*(Ys*Zt-Yt*Zs)-Yr*(Xs*Zt-Xt*Zs)+Zr*(Xs*Yt-Ys*Xt)
    
    rx=(Ys*Zt-Yt*Zs)/J
    sx=(Yt*Zr-Yr*Zt)/J
    tx=(Yr*Zs-Ys*Zr)/J
    
    ry=(Zt*Xs-Zs*Xt)/J
    sy=(Zr*Xt-Zt*Xr)/J
    ty=(Zs*Xr-Zr*Xs)/J
    
    rz=(Xs*Yt-Xt*Ys)/J
    sz=(Xr*Yt-Xt*Yr)/J
    tz=(Xr*Ys-Xs*Yr)/J
    
    G11=rx*rx+ry*ry+rz*rz
    G12=rx*sx+ry*sy+rz*sz
    G13=rx*tx+ry*ty+rz*tz
    G22=sx*sx+sy*sy+sz*sz
    G23=sx*tx+sy*ty+sz*tz
    G33=tx*tx+ty*ty+tz*tz

    G=np.zeros((3,3,G11.size))
    G[0,0,:]=G11
    G[0,1,:]=G12
    G[0,2,:]=G13
    G[1,0,:]=G12
    G[1,1,:]=G22
    G[1,2,:]=G23
    G[2,0,:]=G13
    G[2,1,:]=G23
    G[2,2,:]=G33

    return G

def laplace(X,Y,Z,U):
    G=geometric_factors(X,Y,Z)
    Ux,Uy,Uz=reference_gradient(U)

    Wx=G[0,0,:]*Ux+G[0,1,:]*Uy+G[0,2,:]*Uz
    Wy=G[1,0,:]*Ux+G[1,1,:]*Uy+G[1,2,:]*Uz
    Wz=G[2,0,:]*Ux+G[2,1,:]*Uy+G[2,2,:]*Uz

    W=reference_gradient_transpose(Wx,Wy,Wz)
    return W
