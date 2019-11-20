import numpy as np
from sempy.derivative import reference_gradient
from sempy.quadrature import gauss_lobatto
from sempy.mass import reference_mass_matrix_3D

def geometric_factors(X,Y,Z,n):
    z,w=gauss_lobatto(n-1)

    Xr,Xs,Xt=reference_gradient(X,n)
    Yr,Ys,Yt=reference_gradient(Y,n)
    Zr,Zs,Zt=reference_gradient(Z,n)

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
