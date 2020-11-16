from sempy.agg_amg.J0 import get_J0
from sempy.agg_amg.stiffness_mat import stiffness_mat
import numpy as np

pts = np.loadtxt("pts.dat")
tri = np.loadtxt("tri.dat", dtype=np.int32) - 1

AL, BL, Q, R, xb, yb, p, t = stiffness_mat(pts, tri)

Ab = np.dot(Q.T, np.dot(AL, Q))
A = np.dot(R.T, np.dot(Ab, R))
Bb = np.dot(Q.T, np.dot(BL, Q))
B = np.dot(R.T, np.dot(Bb, R))
x = np.dot(R, xb)
y = np.dot(R, yb)

print(x)

J0 = get_J0(x, y)
