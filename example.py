import numpy as np

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import reference,box_ab

from sempy.stiffness import geometric_factors
from sempy.stiffness import gradient,gradient_transpose

from sempy.iterative import cg,pcg

from mayavi import mlab

N=10
n=N+1

X,Y,Z=trapezoid(N)
G,J,B=geometric_factors(X,Y,Z,n)

def mask(W):
    W=W.reshape((n,n,n))
    W[:,0,:]=0
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

Minv=1.0/(B*J)
def precon(r):
    return Minv*r

b=np.exp(10*Y*Z)*np.sin(10*X)
b=mask(b.reshape((n*n*n,))*B*J)

tol=1.e-8
maxit=1000
verbose=0

x,niter_cg =cg (Ax,       b,tol,maxit,verbose)
x,niter_pcg=pcg(Ax,precon,b,tol,maxit,verbose)
print("# iterations: cg {} pcg {}".format(niter_cg,niter_pcg))

plot=0
if plot:
    mlab.figure()
    mlab.points3d(X,Y,Z,(x-x_analytic).reshape((n,n,n)),scale_mode="none",scale_factor=0.1)
    mlab.axes()
    mlab.show()
