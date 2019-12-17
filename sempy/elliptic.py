import numpy as np

import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.kernel.data import AddressSpace

import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom

from sempy.types import SEMPY_SCALAR

# setup
# -----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

import sempy.loopy.loopy_kernels as lpk

from sempy.derivative import reference_derivative_matrix

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
    ## Get mesh data
    masked_ids=mesh.get_mask_ids()
    global_to_local,global_start=mesh.get_global_to_local_map()
    max_iter=np.max(global_start[1:]-global_start[:-1])
    #print(global_to_local)
    #print(global_start)
    #print(max_iter)

    ## Setup loopy
    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=\
        cl.device_type.GPU)
    #ctx = cl.Context(devices=my_gpu_devices)
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    wnorm = lpk.gen_weighted_norm_knl()
    inner = lpk.gen_inner_product_knl()
    xpay  = lpk.gen_inplace_xpay_knl()
    axpy  = lpk.gen_inplace_axpy_knl()
    mask  = lpk.gen_zero_boundary_knl()
    dssum = lpk.gen_gather_scatter_knl()

    nelem   = mesh.get_num_elems()
    ndofs_1d= mesh.Nq
    ax_lp   = lpk.gen_elliptic_Ax_knl(nelem,ndofs_1d)
    D=reference_derivative_matrix(ndofs_1d-1)
    G=mesh.get_geom()

    rmult=mesh.get_rmult()

    event,(norm_b,)=wnorm(queue, w=rmult, x=b)

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
        event,(Ap,)=ax_lp(queue,D=D,U=p,g=G)

        event,(Ap,)=mask(queue,boundary_indices=masked_ids,dofs=Ap)

        event,(pAp,)=inner(queue,x=Ap,y=p)
        alpha=rdotr/pAp

        event,(Ap,)=dssum(queue,max_iter=max_iter,\
            gather_ids=global_to_local,gather_start=global_start,
            q=Ap)

        event,(x,)=xpay(queue,x=x,a=alpha,y=p)

        event,(r,)=xpay(queue,x=r,a=-alpha,y=Ap)

        rdotr0=rdotr
        event,(rdotr,)=wnorm(queue, w=rmult, x=r)

        beta=rdotr/rdotr0

        if verbose:
            print("niter={} r0={} r1={} alpha={} beta={} pap={}"\
                .format(niter,rdotr0,rdotr,alpha,beta,pAp))

        event,(p,)=axpy(queue,x=p,a=beta,y=r)

        niter=niter+1

    return x,niter
