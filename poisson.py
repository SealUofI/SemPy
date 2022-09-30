# from mayavi import mlab
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from sempy.elliptic import elliptic_cg, elliptic_cg_loopy
from sempy.gradient import (gradient, gradient_2d, gradient_transpose,
                            gradient_transpose_2d)
from sempy.mesh import load_mesh

N = 10
n = N + 1

mesh = load_mesh("box001.msh")
mesh.find_physical_coordinates(N)
mesh.establish_global_numbering()
mesh.calc_geometric_factors()
mesh.setup_mask()
masked_ids = mesh.get_mask_ids()
global_to_local, global_start = mesh.get_global_to_local_map()

nelem = mesh.get_num_elems()
Np = mesh.Np

X = mesh.get_x()
Y = mesh.get_y()
Z = mesh.get_z()

J = mesh.get_jaco()
B = mesh.get_mass()

x = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
x = mesh.apply_mask(x)

b = (
    3
    * np.pi
    * np.pi
    * np.sin(np.pi * X)
    * np.sin(np.pi * Y)
    * np.sin(np.pi * Z)
)
b = b * B * J
b = mesh.dssum(b)
b = mesh.apply_mask(b)

x_cg, niter = elliptic_cg(mesh, b, tol=1e-8, maxit=10000, verbose=0)
x_cg_loopy, niter_loopy = elliptic_cg_loopy(
    mesh, b, tol=1e-8, maxit=10000, verbose=0
)

print(
    "CG iters (host/device): {}/{} error: {}".format(niter, niter_loopy, error)
)
print(
    "is nan? (host/device): {}/{}".format(
        np.isnan(x_cg).any(), np.isnan(x_cg_loopy).any()
    )
)
assert np.allclose(x, x_cg, 1e-8)
assert np.allclose(x, x_cg_loopy, 1e-8)
assert np.allclose(x_cg, x_cg_loopy, 1e-8)
