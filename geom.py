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

x=np.array([-0.5,0.5,-1.0/sqrt(2.0),1.0/sqrt(2.0),-0.5,0.5,-1.0/sqrt(2.0),1.0/sqrt(2.0)])
y=np.array([ 0.5,0.5, 1.0/sqrt(2.0),1.0/sqrt(2.0), 0.5,0.5, 1.0/sqrt(2.0),1.0/sqrt(2.0)])
z=np.array([ 0.0,0.0, 0.0          ,0.0          , 0.1,0,1, 0.1          ,0.1          ])

n=n+1
nn=n*n
m=m+1
mm=m*m

x=x.reshape((n,nn))
y=y.reshape((n,nn))
z=z.reshape((n,nn))
Jx=np.dot(J,x)
Jy=np.dot(J,y)
Jz=np.dot(J,z)

Jx=Jx.reshape((m,nn))
Jy=Jy.reshape((m,nn))
Jz=Jz.reshape((m,nn))
for i in range(m):
  Jx[i,:]=np.dot(J,Jx[i,:])
  Jy[i,:]=np.dot(J,Jy[i,:])
  Jz[i,:]=np.dot(J,Jz[i,:])

Jx=Jx.reshape((mm,n))
Jy=Jy.reshape((mm,n))
Jz=Jz.reshape((mm,n))
for i in range(M+1):
  Yhat[i,:]=Yhat[i,:]+(Yhat[i,:]-Yhat[0,:])*(Xhat[i,0]*Xhat[i,0]-np.multiply(Xhat[i,:],Xhat[i,:]))

plt.pcolormesh(Xhat,Yhat,0*Xhat)
plt.show()
plt.savefig("geom.png")
