import numpy as np

def cg(A,b,tol=1e-8):
    assert b.ndim==1

    n=b.size
    x=np.zeros((n,),dtype=np.float64)

    norm_b=np.dot(b,b)
    TOL=max(tol*tol*norm_b,tol*tol)

    r=b
    rdotr=np.dot(r,r)
    niter=0

    if rdotr<1.e-20:
        return x,niter

    p=r
    while niter<100 and rdotr>TOL:
        Ap=A(p)
        pAp=np.dot(p,Ap)

        alpha=rdotr/pAp

        x=x+alpha*p
        r=r-alpha*Ap

        rdotr0=rdotr
        rdotr=np.dot(r,r)
        beta=rdotr/rdotr0
        print("niter={} r0={} r1={} alpha={} beta={} pap={}".format(niter,rdotr0,rdotr,alpha,beta,pAp))

        p=r+beta*p
        niter=niter+1

    return x,niter
