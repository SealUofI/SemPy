import numpy as np

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import box01,reference
from sempy.stiffness import laplace
from sempy.iterative import cg

from mayavi import mlab

N=20
n=N+1
X,Y,Z=reference(N)

def Ax(x):
    xx=x.reshape((n,n,n))
    Ax=laplace(X,Y,Z,xx)

    return Ax.reshape((n*n*n,))

b=-1*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
b=b.reshape((n*n*n,))
x,niter=cg(Ax,b)
print(np.max(np.abs(b-x)))

#mlab.figure()
#mlab.points3d(X,Y,Z,X+0*Y+0*Z,scale_mode="none",scale_factor=0.01)
#mlab.axes()
#mlab.show()
