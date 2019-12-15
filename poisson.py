import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from sempy.mesh import load_mesh

from sempy.gradient import gradient,gradient_2d,\
    gradient_transpose,gradient_transpose_2d

from sempy.elliptic import elliptic_cg

from mayavi import mlab
import matplotlib.pyplot as plt

plot_on=0

N=10
n=N+1

mesh=load_mesh("box008.msh")
mesh.find_physical_coordinates(N)
mesh.establish_global_numbering()
mesh.calc_geometric_factors()
mesh.setup_mask()

nelem=mesh.get_num_elems()
Np=mesh.Np

X=mesh.get_x()
Y=mesh.get_y()
Z=mesh.get_z()

J=mesh.get_jaco()
B=mesh.get_mass()

x=np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
x=mesh.apply_mask(x)

b=3*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)*np.sin(np.pi*Z)
b=b*B*J
b=mesh.dssum(b)
b=mesh.apply_mask(b)

x_cg,niter=elliptic_cg(mesh,b,tol=1e-8,maxit=1000,verbose=0)

assert np.allclose(x,x_cg,1e-8)

if plot_on:
    if example_2d:
      print("N/A")
    else:
        mlab.figure()
        mlab.points3d(X,Y,Z,u,\
            scale_mode="scalar",scale_factor=0.05)
        mlab.axes()
        mlab.show()
