import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import reference,reference_2d,box_ab

from sempy.stiffness import geometric_factors,geometric_factors_2d
from sempy.stiffness import gradient,gradient_2d,gradient_transpose,gradient_transpose_2d

from sempy.iterative import cg,pcg

from sempy.derivative import reference_derivative_matrix

from sempy.mass import reference_mass_matrix_1d

from mayavi import mlab
import matplotlib.pyplot as plt

if_2d=1
if_plot=0

N=10
n=N+1

if if_2d:
    X,Y=reference_2d(N)
    G,J,B=geometric_factors_2d(X,Y,n)
else:
    X,Y,Z=reference(N)
    G,J,B=geometric_factors(X,Y,Z,n)

def fdm_d_inv_2d(p):
    n=p+1

    Bh=reference_mass_matrix_1d(p);
    Dh=reference_derivative_matrix(p)
    Ah=Dh.T@Bh@Dh; Ah=0.5*(Ah+Ah.T)

    I=np.identity(n,dtype=np.float64)
    Rx=I
    Ry=I[1:,:]

    Ax=Rx@Ah@Rx.T; nx=Ax.shape[0]
    Ay=Ry@Ah@Ry.T; ny=Ay.shape[0]
    Bx=Rx@Bh@Rx.T
    By=Ry@Bh@Ry.T

    Lx,Sx=sla.eig(Ax,Bx); Lx=np.diag(Lx); Ix=np.identity(nx,dtype=np.float64)
    Ly,Sy=sla.eig(Ay,By); Ly=np.diag(Ly); Iy=np.identity(ny,dtype=np.float64)

    Lx=Lx.real
    Ly=Ly.real

    D=np.kron(Iy,Lx)+np.kron(Ly,Ix)
    dinv=1.0/np.diag(D)

    return Rx,Ry,Sx.real,Sy.real,dinv

def fdm_d_inv(p):
    n=p+1

    Bh=reference_mass_matrix_1D(p);
    Dh=reference_derivative_matrix(p)
    Ah=Dh.T@Bh@Dh; Ah=0.5*(Ah+Ah.T)

    I=np.identity(n,dtype=np.float64)
    Rx=Rz=I
    Ry=I[1:,:]

    Ax=Rx@Ah@Rx.T; nx=Ax.shape[0]
    Ay=Ry@Ah@Ry.T; ny=Ay.shape[0]
    Az=Rz@Ah@Rz.T; nz=Az.shape[0]
    Bx=Rx@Bh@Rx.T
    By=Ry@Bh@Ry.T
    Bz=Rz@Bh@Rz.T

    Lx,Sx=sla.eig(Ax,Bx); Lx=np.diag(Lx); Ix=np.identity(nx,dtype=np.float64)
    Ly,Sy=sla.eig(Ay,By); Ly=np.diag(Ly); Iy=np.identity(ny,dtype=np.float64)
    Lz,Sz=sla.eig(Az,Bz); Lz=np.diag(Lz); Iz=np.identity(nz,dtype=np.float64)

    Lx=Lx.real
    Ly=Ly.real
    Lz=Lz.real

    D=np.kron(Iz,np.kron(Iy,Lz))+np.kron(Iz,np.kron(Ly,Ix))+np.kron(Lx,np.kron(Iy,Ix))
    dinv=1.0/np.diag(D)

    return Rx,Ry,Rz,Sx.real,Sy.real,Sz.real,dinv

def mask(W):
    return np.dot(R.T,np.dot(R,W))

if if_2d:
    Rx,Ry,Sx,Sy,dinv=fdm_d_inv_2d(N)
    R=np.kron(Ry,Rx)
else:
    Rx,Ry,Rz,Sx,Sy,Sz,dinv=fdm_d_inv_2d(N)
    R=np.kron(Rz,np.kron(Ry,Rx))

def Ax_2d(x):
    Ux,Uy=gradient_2d(x,n)

    Wx=G[0,0,:]*Ux+G[0,1,:]*Uy
    Wy=G[1,0,:]*Ux+G[1,1,:]*Uy

    W=gradient_transpose_2d(Wx,Wy,n)
    return mask(W)

def Ax(x):
    Ux,Uy,Uz=gradient(x,n)

    Wx=G[0,0,:]*Ux+G[0,1,:]*Uy+G[0,2,:]*Uz
    Wy=G[1,0,:]*Ux+G[1,1,:]*Uy+G[1,2,:]*Uz
    Wz=G[2,0,:]*Ux+G[2,1,:]*Uy+G[2,2,:]*Uz

    W=gradient_transpose(Wx,Wy,Wz,n)
    return mask(W)

Minv=1.0/(B*J)
def precon_mass(r):
    return Minv*r

def fast_kron_2d(Sy,Sx,U):
    nx,mx=Sx.shape
    ny,my=Sy.shape

    U=U.reshape((my,mx))
    U=np.dot(U,Sx.T)

    V=U.reshape((my,nx))
    U=np.dot(Sy,V)

    return U.reshape((nx*ny,))

def fast_kron(Sz,Sy,Sx,U):
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

def precon_fdm_2d(r):
    r=np.dot(R,r)
    b=fast_kron_2d(Sy.T,Sx.T,r)
    b=dinv*b
    return np.dot(R.T,fast_kron_2d(Sy,Sx,b))

def precon_fdm(r):
    r=np.dot(R,r)
    b=fast_kron(Sz.T,Sy.T,Sx.T,r)
    b=dinv*b
    return np.dot(R.T,fast_kron(Sz,Sy,Sx,b))

if if_2d:
    b=np.exp(10*Y)*np.sin(10*X)
    b=mask(b.reshape((n*n,))*B*J)
else:
    b=np.exp(10*Y*Z)*np.sin(10*X)
    b=mask(b.reshape((n*n*n,))*B*J)

tol=1.e-10
maxit=10000
verbose=0

if if_2d:
    x_cg  ,niter_cg  =cg (Ax_2d              ,b,tol,maxit,verbose)
    x_mass,niter_mass=pcg(Ax_2d,precon_mass  ,b,tol,maxit,verbose)
    x_fdm ,niter_fdm =pcg(Ax_2d,precon_fdm_2d,b,tol,maxit,verbose)
else:
    x_cg  ,niter_cg  =cg (Ax            ,b,tol,maxit,verbose)
    x_mass,niter_mass=pcg(Ax,precon_mass,b,tol,maxit,verbose)
    x_fdm ,niter_fdm =pcg(Ax,precon_fdm ,b,tol,maxit,verbose)

print("# iterations: cg {} pcg (mass) {} pcg (fdm) {}".format(niter_cg,
  niter_mass,niter_fdm))
print("error: {}".format(np.max(np.abs(x_cg-x_fdm))))

if if_plot:
    if if_2d:
      print("N/A")
    else:
        mlab.figure()
        mlab.points3d(X,Y,Z,(x_cg-x_fdm).reshape((n,n,n)),scale_mode="none",scale_factor=0.1)
        mlab.axes()
        mlab.show()
