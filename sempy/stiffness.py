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

def calc_geometric_factors(mesh):
    n=mesh.Nq

    mesh.geom = []
    mesh.jaco = []

    if mesh.get_ndim()==3:
        mesh.B=reference_mass_matrix_3d(n-1)
        for e in range(mesh.get_num_elements()):
            Xr,Xs,Xt=gradient(mesh.xe[e,:],n)
            Yr,Ys,Yt=gradient(mesh.ye[e,:],n)
            Zr,Zs,Zt=gradient(mesh.ze[e,:],n)

            J=Xr*(Ys*Zt-Yt*Zs)-Yr*(Xs*Zt-Xt*Zs)+Zr*(Xs*Yt-Ys*Xt)
            mesh.jaco.append(J)

            rx=(Ys*Zt-Yt*Zs)/J
            sx=(Yt*Zr-Yr*Zt)/J
            tx=(Yr*Zs-Ys*Zr)/J

            ry=-(Zt*Xs-Zs*Xt)/J
            sy=-(Zr*Xt-Zt*Xr)/J
            ty=-(Zs*Xr-Zr*Xs)/J

            rz= (Xs*Yt-Xt*Ys)/J
            sz=-(Xr*Yt-Xt*Yr)/J
            tz= (Xr*Ys-Xs*Yr)/J

            g11=rx*rx+ry*ry+rz*rz
            g12=rx*sx+ry*sy+rz*sz
            g13=rx*tx+ry*ty+rz*tz
            g22=sx*sx+sy*sy+sz*sz
            g23=sx*tx+sy*ty+sz*tz
            g33=tx*tx+ty*ty+tz*tz

            g=np.zeros((3,3,g11.size))
            g[0,0,:]=g11*mesh.B*J
            g[0,1,:]=g12*mesh.B*J
            g[0,2,:]=g13*mesh.B*J
            g[1,0,:]=g12*mesh.B*J
            g[1,1,:]=g22*mesh.B*J
            g[1,2,:]=g23*mesh.B*J
            g[2,0,:]=g13*mesh.B*J
            g[2,1,:]=g23*mesh.B*J
            g[2,2,:]=g33*mesh.B*J

            mesh.geom.append(g)
    else:
        B=reference_mass_matrix_2d(n-1)
        for e in range(mesh.get_num_elements()):
            Xr,Xs=gradient_2d(mesh.xe[e,:],n)
            Yr,Ys=gradient_2d(mesh.ye[e,:],n)

            J=Xr*Ys-Yr*Xs
            mesh.jaco.append(J)

            rx= Ys/J
            sx=-Yr/J

            ry=-Xs/J
            sy= Xr/J

            g11=rx*rx+ry*ry
            g12=rx*sx+ry*sy
            g22=sx*sx+sy*sy

            g=np.zeros((2,2,g11.size))
            g[0,0,:]=g11*mesh.B*J
            g[0,1,:]=g12*mesh.B*J
            g[1,0,:]=g12*mesh.B*J
            g[1,1,:]=g22*mesh.B*J

            mesh.geom.append(g)
    mesh.geom=np.array(mesh.geom)
    mesh.jaco=np.array(mesh.jaco)
