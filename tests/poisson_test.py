import pytest
import numpy as np

from sempy.gradient import gradient,gradient_2d,\
    gradient_transpose,gradient_transpose_2d

from sempy.mesh import load_mesh
from sempy.iterative import pcg

from sempy.meshes.box import box_ab_2d

def test_poisson_sin_3d():
    N=15
    n=N+1

    mesh=load_mesh("box001.msh")
    mesh.find_physical_nodes(N)
    mesh.calc_geometric_factors()

    G=mesh.geom[0,:]
    J=mesh.jaco[0,:]
    B=mesh.B

    def mask(W):
        W=W.reshape((n,n,n))
        W[0,:,:]=0
        W[n-1,:,:]=0
        W[:,0,:]=0
        W[:,n-1,:]=0
        W[:,:,0]=0
        W[:,:,n-1]=0
        W=W.reshape((n*n*n,))
        return W

    def Ax(x):
        Ux,Uy,Uz=gradient(x,n)

        Wx=G[0,0,:]*Ux+G[0,1,:]*Uy+G[0,2,:]*Uz
        Wy=G[1,0,:]*Ux+G[1,1,:]*Uy+G[1,2,:]*Uz
        Wz=G[2,0,:]*Ux+G[2,1,:]*Uy+G[2,2,:]*Uz

        W=gradient_transpose(Wx,Wy,Wz,n)
        W=mask(W)
        return W

    X=mesh.xe[0,:]
    Y=mesh.ye[0,:]
    Z=mesh.ze[0,:]

    x_analytic=np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
    x_analytic=mask(x_analytic.reshape((n*n*n,)))

    b_analytic=3*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
    b_analytic=mask(b_analytic.reshape(n*n*n,)*B*J)

    Minv_=1.0/(B*J)
    def Minv(r):
        return Minv_*r

    x,niter=pcg(Ax,Minv,b_analytic,tol=1e-8,maxit=1000,verbose=0)

    assert np.allclose(x,x_analytic,1e-8)

#def test_poisson_sin_2d():
#    N=15
#    n=N+1
#
#    X,Y=box_ab_2d(0.,1.,N)
#    G,J,B=geometric_factors_2d(X,Y,n)
#
#    def mask(W):
#        W=W.reshape((n,n))
#        W[0,:]=0
#        W[n-1,:]=0
#        W[:,0]=0
#        W[:,n-1]=0
#        W=W.reshape((n*n,))
#        return W
#
#    def Ax(x):
#        Ux,Uy=gradient_2d(x,n)
#
#        Wx=G[0,0,:]*Ux+G[0,1,:]*Uy
#        Wy=G[1,0,:]*Ux+G[1,1,:]*Uy
#
#        W=gradient_transpose_2d(Wx,Wy,n)
#        W=mask(W)
#        return W
#
#    x_analytic=np.sin(np.pi*X)*np.sin(np.pi*Y)
#    x_analytic=mask(x_analytic.reshape((n*n,)))
#
#    b_analytic=2*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)
#    b_analytic=mask(b_analytic.reshape(n*n,)*B*J)
#
#    Minv_=1.0/(B*J)
#    def Minv(r):
#        return Minv_*r
#
#    x,niter=pcg(Ax,Minv,b_analytic,tol=1e-8,maxit=1000,verbose=0)
#
#    assert np.allclose(x,x_analytic,1e-8)
