import numpy as np

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import box01,reference
from sempy.stiffness import geometric_factors
from sempy.derivative import reference_gradient,reference_gradient_transpose
from sempy.iterative import cg

from mayavi import mlab

N=5
n=N+1

X,Y,Z=reference(N)
G=geometric_factors(X,Y,Z,n)

def Ax(x):
    Ux,Uy,Uz=reference_gradient(x,n)

    Wx=G[0,0,:]*Ux+G[0,1,:]*Uy+G[0,2,:]*Uz
    Wy=G[1,0,:]*Ux+G[1,1,:]*Uy+G[1,2,:]*Uz
    Wz=G[2,0,:]*Ux+G[2,1,:]*Uy+G[2,2,:]*Uz

    W=reference_gradient_transpose(Wx,Wy,Wz,n)
    return W

b=-1*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
b=b.reshape((n*n*n,))
x,niter=cg(Ax,b)
print(np.max(np.abs(b-x)))

mlab.figure()
mlab.points3d(X,Y,Z,b.reshape((n,n,n)),scale_mode="none",scale_factor=0.1)
mlab.axes()
mlab.show()
