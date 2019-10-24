from meshes.curved import trapezoid
from meshes.box import box01,reference
from mayavi import mlab

N=20

#X,Y,Z=trapezoid(N)
#X,Y,Z=box01(N)
X,Y,Z=reference(N)

mlab.figure()
mlab.points3d(X,Y,Z,X+0*Y+0*Z,scale_mode="none",scale_factor=0.01)
mlab.axes()
mlab.show()
