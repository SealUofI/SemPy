import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from sempy.mesh import load_mesh

from sempy.gradient import gradient,gradient_2d,\
    gradient_transpose,gradient_transpose_2d

from sempy.helmholtz import helmholtz
from mayavi import mlab

N=20
n=N+1

mesh=load_mesh("quad001.msh")
mesh.find_physical_coordinates(N)
mesh.establish_global_numbering()
mesh.calc_geometric_factors()
mesh.setup_mask()
masked_ids=mesh.get_mask_ids()
global_to_local,global_start=mesh.get_global_to_local_map()

nelem=mesh.get_num_elems()

X=mesh.get_x()
Y=mesh.get_y()
Z=mesh.get_z()
J=mesh.get_jaco()
B=mesh.get_mass()

x=np.sin(np.pi*X)*np.sin(np.pi*Y)
x=mesh.apply_mask(x)

b=2*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)
b=b*B*J
b=mesh.dssum(b)
b=mesh.apply_mask(b)

x_helm,niter=helmholtz(mesh,b,0.0,tol=1e-8,maxit=10000,verbose=0)
error=np.max(np.abs(x-x_helm))
print("CG iters: {} error: {}".format(niter,error))

assert np.allclose(x,x_helm,1e-8)

plot_on=1
if plot_on:
    mlab.figure()
    mlab.points3d(X,Y,Z,(x-x_helm),scale_mode="none",
        scale_factor=0.1)
    mlab.axes()
    mlab.show()
