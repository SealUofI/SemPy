import pytest
import numpy as np

from sempy.stiffness import geometric_factors
from sempy.derivative import reference_gradient,reference_gradient_transpose
from sempy.quadrature import gauss_lobatto
from sempy.iterative import pcg

from sempy.meshes.box import reference,box_ab
def test_poisson_sin():
    N=20
    n=N+1
    X,Y,Z=box_ab(-2.,2.,N)
    G,J,B=geometric_factors(X,Y,Z,n)
    
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
        Ux,Uy,Uz=reference_gradient(x,n)
    
        Wx=G[0,0,:]*Ux+G[0,1,:]*Uy+G[0,2,:]*Uz
        Wy=G[1,0,:]*Ux+G[1,1,:]*Uy+G[1,2,:]*Uz
        Wz=G[2,0,:]*Ux+G[2,1,:]*Uy+G[2,2,:]*Uz
    
        W=reference_gradient_transpose(Wx,Wy,Wz,n)
        W=mask(W)
        return W
    
    x_analytic=np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
    x_analytic=mask(x_analytic.reshape((n*n*n,)))
    
    b_analytic=3*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
    b_analytic=mask(b_analytic.reshape(n*n*n,)*B*J)
    
    Minv=1.0/(B*J)
    x,niter=pcg(Ax,Minv,b_analytic,tol=1e-12,maxit=1000,verbose=0)

    assert np.allclose(x,x_analytic,1e-8)
