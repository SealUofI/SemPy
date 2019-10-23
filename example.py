from meshes.curved import trapezoid
from mayavi import mlab

N=20

X,Y,Z=trapezoid(N)

#mlab.figure()
#mlab.points3d(X,Y,Z,0*X+0*Y+0*Z,scale_mode="none",scale_factor=0.01)
#mlab.axes()
#mlab.show()

