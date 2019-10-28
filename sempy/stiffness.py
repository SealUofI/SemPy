import numpy as np
from sempy.derivative import reference_gradient
from sempy.quadrature import gauss_lobatto

def geometric_factors(X,Y,Z,n):
    z,w=gauss_lobatto(n-1)
    Q=np.ones((n*n*n,),dtype=np.float64)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                Q[k*n*n+j*n+i]=w[i]*w[j]*w[k]

    Xr,Xs,Xt=reference_gradient(X,n)
    Yr,Ys,Yt=reference_gradient(Y,n)
    Zr,Zs,Zt=reference_gradient(Z,n)

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
    G[0,0,:]=G11*Q
    G[0,1,:]=G12*Q
    G[0,2,:]=G13*Q
    G[1,0,:]=G12*Q
    G[1,1,:]=G22*Q
    G[1,2,:]=G23*Q
    G[2,0,:]=G13*Q
    G[2,1,:]=G23*Q
    G[2,2,:]=G33*Q

    return G
