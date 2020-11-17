from sempy.agg_amg.J0 import get_J0
from sempy.agg_amg.stiffness_mat import stiffness_mat
from sempy.agg_amg.project import project

import numpy as np
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

np.set_printoptions(threshold=np.inf)

pts = np.loadtxt("pts.dat")
tri = np.loadtxt("tri.dat", dtype=np.int32) - 1

AL, BL, Q, R, xb, yb, p, t = stiffness_mat(pts, tri)

Ab = Q.T.dot(AL.dot(Q))
A = R.dot(Ab.dot(R.T))
Bb = Q.T.dot(BL.dot(Q))
B = R.dot(Bb.dot(R.T))
x = R.dot(xb)
y = R.dot(yb)

J0 = get_J0(x, y)
n, ncuts = J0.shape

rr = x**2 + y**2
uex = 1.0 - rr
f = 4*B.dot(np.ones((n, 1)))

tol = 1e-8
prec = 0  # Jacobi
u, res, n_iter = project(f, A, J0, tol, prec, verbose=0)

prec = 1  # V-cycle
u_mg, res_mg, n_iter_mg = project(f, A, J0, tol, prec, verbose=0)

itr = np.arange(0, n_iter+1)
itr_mg = np.arange(0, n_iter_mg+1)

plt.figure()
plt.title("Jacobi vs MG V-cycle on a disk, n={}".format(n))
plt.plot(itr, res, '-x', label='Jacobi')
plt.plot(itr_mg, res_mg, '-x', label='MG')
plt.yscale('log')
plt.legend(loc=0)
plt.savefig('residual_vs_iter.pdf', bbox_inches='tight')
plt.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print("pts: {} tri: {}".format(pts.shape, tri.shape))
#triangles = mtri.Triangulation(pts[:, 0], pts[:, 1], tri)
#ax.plot_trisurf(triangles, u.ravel())
#plt.savefig('solution.pdf', bbox_inches='tight')
