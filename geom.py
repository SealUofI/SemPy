import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from quadrature import gauss_lobatto
from interpolation import lagrange

N=1
M=10

zn,wn=gauss_lobatto(N)
zm,wm=gauss_lobatto(M)
J=lagrange(zm,zn)

x=np.array([-0.5,0.5,-1.0/sqrt(2.0),1.0/sqrt(2.0)])
y=np.array([ 0.5,0.5, 1.0/sqrt(2.0),1.0/sqrt(2.0)])
x=x.reshape((N+1,N+1))
y=y.reshape((N+1,N+1))

Xhat=np.dot(np.dot(J,x),J.T)
Yhat=np.dot(np.dot(J,y),J.T)
plt.pcolormesh(Xhat,Yhat,0*Xhat)
plt.show()
plt.savefig("geom.png")
