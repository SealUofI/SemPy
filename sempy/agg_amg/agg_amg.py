from sempy.agg_amg.J0 import get_J0
from sempy.agg_amg.stiffness_mat import stiffness_mat
from sempy.agg_amg.project import project

import numpy as np
import scipy.sparse.linalg as sla

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
print("J0: {} x {}, {}".format(n, ncuts, sla.norm(J0)))

rr = x**2 + y**2  # np.multiply(x, x)+np.multiply(y, y)
uex = 1.0 - rr
f = 4*B.dot(np.ones((n, 1)))

JTrr = (J0.T).dot(rr)
Arr = A.dot(rr)
print("rr: {} f: {} A.dot(rr): {} (J^T).dot(rr): {}".format(np.linalg.norm(
    rr), np.linalg.norm(f), np.linalg.norm(Arr), np.linalg.norm(JTrr)))

tol = 1e-8
prec = 1  # V-cycle, prec = 0 for Jacobi
u, res, n_iter = project(f, A, J0, tol, prec, verbose=0)
print("n_iter: {} norm u: {}".format(n_iter, np.linalg.norm(u)))
