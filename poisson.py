from meshes.curved import trapezoid
from mayavi import mlab
from derivative import reference_gradient

N=20

X,Y,Z=trapezoid(N)

##mlab.figure()
##mlab.points3d(X,Y,Z,0*X+0*Y+0*Z,scale_mode="none",scale_factor=0.01)
##mlab.axes()
##mlab.show()

Xr,Xs,Xt=reference_gradient(X)
Yr,Ys,Yt=reference_gradient(Y)
Zr,Zs,Zt=reference_gradient(Z)

J=Xr*(Ys*Zt-Yt*Zs)-Yr*(Xs*Zt-Xt*Zs)+Zr*(Xs*Yt-Ys*Xt)

rx=(Ys*Zt-Yt*Zs)/J
sx=(Yt*Zr-Yr*Zt)/J
tx=(Yr*Zs-Ys*Zr)/J

ry=(Zt*Xs-Zs*Xt)/J
sy=(Zr*Xt-Zt*Xr)/J
ty=(Zs*Xr-Zr*Xs)/J

rz=(Xs*Yt-Xt*Ys)/J
sz=(Xr*Yt-Xt*Yr)/J
tz=(Xr*Ys-Xs*Yr)/J

G11=rx*rx+ry*ry+rz*rz
G12=rx*sx+ry*sy+rz*sz
G13=rx*tx+ry*ty+rz*tz
G22=sx*sx+sy*sy+sz*sz
G23=sx*tx+sy*ty+sz*tz
G33=tx*tx+ty*ty+tz*tz
