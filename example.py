import numpy as np
from meshes.curved import trapezoid
from meshes.box import box01,reference
from mayavi import mlab
from iterative import cg

def Ax(x):
    assert x.ndim==1
    assert x.size==2

    A=np.array([4.0,1.0,1.0,3.0])
    A=A.reshape((2,2))

    return np.dot(A,x)

b=np.array([1.0,2.0])
x,niter=cg(Ax,b)
print(x)
print(niter)

N=20
X,Y,Z=reference(N)
mlab.figure()
mlab.points3d(X,Y,Z,X+0*Y+0*Z,scale_mode="none",scale_factor=0.01)
mlab.axes()
mlab.show()
