from meshes.curved import trapezoid
from mayavi import mlab

X,Y,Z=trapezoid(20)

mlab.figure()
mlab.points3d(X,Y,Z,0*X+0*Y+0*Z,scale_mode="none", scale_factor=0.01)
mlab.axes()
mlab.show()
