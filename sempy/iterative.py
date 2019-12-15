import numpy as np

def cg(A,b,tol=1e-12,maxit=100,verbose=0):
    norm_b=np.dot(b,b)
    TOL=max(tol*tol*norm_b,tol*tol)

    r=b
    rdotr=np.dot(r,r)
    if verbose:
        print('Initial rnorm={}'.format(rdotr))

    x=0*b
    niter=0
    if rdotr<1.e-20:
        return x,niter

    p=r
    while niter<maxit and rdotr>TOL:
        Ap=A(p)
        pAp=np.dot(p,Ap)

        alpha=rdotr/pAp

        x=x+alpha*p
        r=r-alpha*Ap

        rdotr0=rdotr
        rdotr=np.dot(r,r)
        beta=rdotr/rdotr0

        if verbose:
            print("niter={} r0={} r1={} alpha={} beta={} pap={}"\
                .format(niter,rdotr0,rdotr,alpha,beta,pAp))

        p=r+beta*p
        niter=niter+1

    return x,niter

def pcg(A,Minv,b,tol=1e-8,maxit=100,verbose=0):
    assert b.ndim==1

    n=b.size
    x=np.zeros((n,),dtype=np.float64)

    norm_b=np.dot(b,b)
    TOL=max(tol*tol*norm_b,tol*tol)

    r=b
    niter=0

    z=Minv(r)
    rdotz=np.dot(r,z)

    p=z
    while niter<maxit and rdotz>TOL:
        Ap=A(p)
        pAp=np.dot(p,Ap)

        alpha=rdotz/pAp

        x=x+alpha*p
        r=r-alpha*Ap

        z=Minv(r)

        rdotz0=rdotz
        rdotz=np.dot(r,z)
        beta=rdotz/rdotz0
        if verbose:
            print("niter={} r0={} r1={} alpha={} beta={} pap={}".format( \
                niter,rdotz0,rdotz,alpha,beta,pAp))

        p=z+beta*p
        niter=niter+1

    return x,niter
