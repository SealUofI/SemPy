import numpy as np

from sempy.gradient import gradient,gradient_2d,\
    gradient_transpose,gradient_transpose_2d

def elliptic_ax(mesh,p):
    g=mesh.get_geom()
    nelem=mesh.get_num_elems()
    Np=mesh.Np
    Nq=mesh.Nq

    p_=p.reshape((nelem,Np))
    ap=np.zeros_like(p_)
    for e in range(nelem):
        px,py,pz=gradient(p_[e,:],Nq)

        apx=g[e,0,0,:]*px+g[e,0,1,:]*py+g[e,0,2,:]*pz
        apy=g[e,1,0,:]*px+g[e,1,1,:]*py+g[e,1,2,:]*pz
        apz=g[e,2,0,:]*px+g[e,2,1,:]*py+g[e,2,2,:]*pz

        ap[e,:]=gradient_transpose(apx,apy,apz,Nq)

    return ap.reshape((nelem*Np,))

def elliptic_cg(mesh,b,tol=1e-12,maxit=100,verbose=0):
    rmult=mesh.get_rmult()

    norm_b=np.dot(np.multiply(rmult,b),b)
    TOL=max(tol*tol*norm_b,tol*tol)

    r=b
    rdotr=np.dot(np.multiply(rmult,r),r)
    if verbose:
        print('Initial rnorm={}'.format(rdotr))

    x=0*b
    niter=0
    if rdotr<1.e-20:
        return x,niter

    p=r
    while niter<maxit and rdotr>TOL:
        Ap=elliptic_ax(mesh,p)
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

def elliptic_cg_loopy(mesh,b,tol=1e-12,maxit=100,verbose=0):
    rmult=mesh.get_rmult()

    norm_b=np.dot(np.multiply(rmult,b),b)
    TOL=max(tol*tol*norm_b,tol*tol)

    r=b
    rdotr=np.dot(np.multiply(rmult,r),r)
    if verbose:
        print('Initial rnorm={}'.format(rdotr))

    x=0*b
    niter=0
    if rdotr<1.e-20:
        return x,niter

    p=r
    while niter<maxit and rdotr>TOL:
        Ap=elliptic_ax(mesh,p)
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
