import numpy as np

from sempy.derivative import reference_derivative_matrix

from sempy.gradient import gradient,gradient_2d,\
    gradient_transpose,gradient_transpose_2d

from sempy.elliptic import elliptic_ax_2d,elliptic_ax_3d

def helmholtz_ax_2d(mesh,p,lmbda):
    Ap=elliptic_ax_2d(mesh,p)
    Ap=Ap+lmbda*np.multiply(mesh.get_mass(),p)
    return Ap

def helmholtz(mesh,b,lmbda,tol=1e-12,maxit=100,verbose=0):
    ndim=mesh.get_ndim()
    if ndim==3:
        helmholtz_ax=helmholtz_ax_3d
    else:
        helmholtz_ax=helmholtz_ax_2d

    rmult=mesh.get_rmult()
    norm_b=np.dot(np.multiply(rmult,b),b)
    TOL=max(tol*tol*norm_b,tol*tol)

    r=b
    rdotr=norm_b
    if verbose:
        print('Initial rnorm={}'.format(rdotr))

    x=0*b
    niter=0
    if rdotr<1.e-20:
        return x,niter

    p=r
    while niter<maxit and rdotr>TOL:
        Ap=helmholtz_ax(mesh,p,lmbda)
        mesh.apply_mask(Ap)

        pAp=np.dot(Ap,p)
        alpha=rdotr/pAp

        Ap=mesh.dssum(Ap)

        x=x+alpha*p
        r=r-alpha*Ap

        rdotr0=rdotr
        rdotr=np.dot(np.multiply(rmult,r),r)
        beta=rdotr/rdotr0

        if verbose:
            print("niter={} r0={} r1={} alpha={} beta={} pap={}"\
                .format(niter,rdotr0,rdotr,alpha,beta,pAp))

        p=r+beta*p
        niter=niter+1

    return x,niter
