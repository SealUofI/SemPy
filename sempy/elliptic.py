import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.kernel.data import AddressSpace

from sempy.types import SEMPY_SCALAR

# setup
# -----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

import sempy.loopy.loopy_kernels as lpk


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

    rmult=mesh.get_rmult()

    event,(norm_b,)=wnorm(queue, w=rmult, x=b)
    #norm_b=np.dot(np.multiply(rmult,b),b)

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

        event,(Ap,)=mask(queue,boundary_indices=masked_ids,dofs=Ap)
        #mesh.apply_mask(Ap)

        event,(pAp,)=inner(queue,x=Ap,y=p)
        #pAp=np.dot(Ap,p)
        alpha=rdotr/pAp

        event,(Ap,)=dssum(queue,max_iter=max_iter,\
            gather_ids=global_to_local,gather_start=global_start,
            q=Ap)
        #Ap=mesh.dssum(Ap)

        event,(x,)=xpay(queue,x=x,a=alpha,y=p)
        #x=x+alpha*p

        event,(r,)=xpay(queue,x=r,a=-alpha,y=Ap)
        #r=r-alpha*Ap

        rdotr0=rdotr
        event,(rdotr,)=wnorm(queue, w=rmult, x=r)
        #rdotr=np.dot(np.multiply(rmult,r),r)

        beta=rdotr/rdotr0

        if verbose:
            print("niter={} r0={} r1={} alpha={} beta={} pap={}"\
                .format(niter,rdotr0,rdotr,alpha,beta,pAp))

        event,(p,)=axpy(queue,x=p,a=beta,y=r)
        #p=r+beta*p

        niter=niter+1

    return x,niter

def cg(A,b,tol=1e-12,maxit=100,verbose=0):

    assert b.ndim==1

    m,n = A.shape

    Ax = lpk.gen_Ax_knl()
    norm = lpk.gen_norm_knl()
    ip = lpk.gen_inner_product_knl()
    cgi = lpk.gen_CG_iteration()
    vupdt = lpk.gen_inplace_xpay_knl()
    axpy = lpk.gen_inplace_axpy_knl()
    #Ax = lp.set_options(Ax, "write_code")
    #print(lp.generate_code_v2(Ax).device_code())

    x=np.zeros((n,),dtype=SEMPY_SCALAR)
    
    evt, (norm_b_lp,) = norm(queue,x=b)
    print(norm_b_lp)

    norm_b=np.dot(b,b)
    print(norm_b)

    TOL=max(tol*tol*norm_b,tol*tol)

    r=b

    rdotr=np.dot(r,r)
    print(rdotr)
    niter=0

    if verbose:
        print('Initial rnorm={}'.format(rdotr))
    if rdotr<1.e-20:
        return x,niter

    p=r

    x_lp=cl.array.to_device(queue, x)#x.copy()
    r_lp=cl.array.to_device(queue, r)#r.copy()
    p_lp = r_lp.copy()
    A_lp = cl.array.to_device(queue, A)
    #A_lp = A.copy()
    evt, (rdotr_lp,) = norm(queue, x=r_lp)
    rdotr_lp = rdotr_lp.get()

    #rdotr_lp = rdotr
    #p_lp=p.copy()

    while niter<maxit and rdotr>TOL and rdotr_lp > TOL:
        niter+=1
        """
        <> a = rdotr_prev / sum(j, p[j]*Ap[j]) {id=a}
        x[l] = x[l] + a*p[l] {id=x, dep=a}
        r[l] = r[l] - a*Ap[l] {id=r, dep=a}
        rdotr = sum(k, r[k]*r[k]) {id=rdotr, dep=r}
        p_out[i] = r[i] + (rdotr/rdotr_prev) * p[i] {id=p, dep=rdotr}
        """
       

        # We need to define this for multiple elements
        #p_lp=p; rdotr_lp = rdotr #Delete this line when finished
        
        evt, (Ap_lp,) = Ax(queue, A=A_lp, x=p_lp)
        #evt, (p_lp,r_lp,rdotr_lp,x_lp) = cgi(queue, Ap=Ap_lp, p=p_lp, r=r_lp, rdotr_prev=rdotr_lp, x=x_lp)

        #"""
        evt, (pAp_lp,) = ip(queue, x=p_lp, y=Ap_lp)
        a_lp = rdotr_lp / pAp_lp.get() # Host operation because rdotr is on the host anyway with gslib
        evt, (x_lp,) = vupdt(queue, a=a_lp, x=x_lp, y=p_lp)
        evt, (r_lp,) = vupdt(queue, a=-a_lp, x=r_lp, y=Ap_lp) 
        rdotr_prev_lp = rdotr_lp # Host operation
        evt, (rdotr_lp,) = norm(queue, x=r_lp)
        rdotr_lp = rdotr_lp.get()
        evt, (p_lp,) = axpy(queue, a=(rdotr_lp/rdotr_prev_lp), x=p_lp, y=r_lp)
        #"""

        print("CL: {}".format(rdotr_lp))

        # Numpy version for comparison
        """
        Ap=A@p
        pAp=np.dot(p,Ap)

        alpha=rdotr/pAp

        x=x+alpha*p
        r=r-alpha*Ap

        rdotr0=rdotr
        rdotr=np.dot(r,r)
        beta=rdotr/rdotr0
        if verbose:
            print("niter={} r0={} r1={} alpha={} beta={} pap={}".format( \
                niter,rdotr0,rdotr,alpha,beta,pAp))

        p=r+beta*p
        print("Numpy: {}".format(rdotr))
        """

 
    #print(np.linalg.norm(x - x_lp))
    return x,niter
