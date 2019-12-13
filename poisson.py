import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from sempy.meshes.curved import trapezoid
from sempy.meshes.box import reference,reference_2d,box_ab
from sempy.mesh import load_mesh

from sempy.mass import reference_mass_matrix_1d
from sempy.stiffness import geometric_factors,geometric_factors_2d
from sempy.stiffness import gradient,gradient_2d,gradient_transpose,\
    gradient_transpose_2d

from sempy.iterative import cg,pcg

from mayavi import mlab
import matplotlib.pyplot as plt

N=5
n=N+1

mesh=load_mesh("box001.msh")
mesh.find_physical_nodes(N)
X=mesh.xe[0].reshape((n,n,n))
Y=mesh.ye[0].reshape((n,n,n))
Z=mesh.ze[0].reshape((n,n,n))
print("{} {} {}".format(X.shape,Y.shape,Z.shape))

example_2d=0
plot_on   =1

if plot_on:
    if example_2d:
      print("N/A")
    else:
        mlab.figure()
        mlab.points3d(X,Y,Z,X,\
            scale_mode="none",scale_factor=0.1)
        mlab.axes()
        mlab.show()
