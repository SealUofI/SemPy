import numpy as np
from sempy.meshes.curved import trapezoid
from sempy.meshes.box import box01,reference
from mayavi import mlab
from sempy.iterative import cg

N=20
X,Y,Z=reference(N)
mlab.figure()
mlab.points3d(X,Y,Z,X+0*Y+0*Z,scale_mode="none",scale_factor=0.01)
mlab.axes()
mlab.show()
