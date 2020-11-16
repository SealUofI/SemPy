from sempy.agg_amg.J0 import get_J0
from sempy.agg_amg.stiffness_mat import stiffness_mat
from sempy.agg_amg.project import project

import numpy as np

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
print("J0: {} x {}".format(n, ncuts))

rr = np.multiply(x, x)+np.multiply(y, y)
uex = 1.0 - rr
f = 4*B.dot(np.ones((n, 1)))
print("norm rr: {}, Ar: {} uex: {}, f:{} {}".format(
    np.linalg.norm(rr), np.linalg.norm(A.dot(rr)), np.linalg.norm(uex), np.linalg.norm(f), f.shape))

#tol = 1e-8
#prec = 0
#u, res, n_iter = project(f, A.todense(), J0, tol, prec, verbose=1)
#print("n_iter: {} norm x: {}".format(n_iter, np.linalg.norm(u)))
