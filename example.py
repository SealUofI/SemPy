import numpy as np

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import box01,reference
from sempy.stiffness import geometric_factors
from sempy.derivative import reference_gradient,reference_gradient_transpose
from sempy.quadrature import gauss_lobatto
from sempy.iterative import cg,pcg

from mayavi import mlab

def read(fname):
    with open(fname) as f:
        l=[]
        lines=f.readlines()
        for line in lines:
            l.append(float(line))
        return np.array(l)

N=10
n=N+1

X,Y,Z=reference(N)
G=geometric_factors(X,Y,Z,n)
print(X.shape)

z,w=gauss_lobatto(n-1)
Q=np.ones((n*n*n,),dtype=np.float64)
for k in range(n):
    for j in range(n):
        for i in range(n):
            Q[k*n*n+j*n+i]=w[i]*w[j]*w[k]

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
x_analytic=x_analytic.reshape((n*n*n,))
x_analytic=mask(x_analytic)

b=3*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
b=b.reshape((n*n*n,))
b=b*Q
b=mask(b)

#x,niter=cg(Ax,b,tol=1e-8,maxit=1000,verbose=1)
Minv=1.0/Q
x,niter=pcg(Ax,Minv,b,tol=1e-12,maxit=1000,verbose=1)
print(np.max(np.abs(x-x_analytic)))

mlab.figure()
mlab.points3d(X,Y,Z,(x-x_analytic).reshape((n,n,n)),scale_mode="none",scale_factor=0.1)
mlab.axes()
mlab.show()
