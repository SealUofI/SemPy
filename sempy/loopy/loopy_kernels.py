import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.kernel.data import AddressSpace

def gen_zero_boundary_knl():
    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        dofs[boundaryIndices[i]] = 0
        """,
        assumptions="n > 0",
        default_offset=None,
        name="zero_boundary"
    )
    
    return knl

def python_gather_scatter(gatherIds, gatherStarts, maxIter, q_in):
    q_out = np.zeros_like(q_in)
    n = gatherStarts.shape[0]
    for k in range(1,n):
        start = gatherStarts[k-1]
        diff = gatherStarts[k] - start
        gq = 0
        for i in range(maxIter):
            if i < diff:
                gq += q_in[gatherIds[start + i]]
        for j in range(maxIter):
            if j < diff:
                q_out[gatherIds[start + j]] = gq
    return q_out

def gen_gather_scatter_knl():
    knl = lp.make_kernel(
        """
        {[k,i,j]: 0<=i,j<maxIter and 1<=k<n}
        """,
        """
        # þis could go off of þe end of þe array.
        #gq := sum(i, (i < diff)*q_in[gatherIds[start + i]])
        for k
            <> start = gatherStarts[k-1]
            <> diff = gatherStarts[k] - start
            for i
                if i < diff
                    gq = gq + q_in[gatherIds[start + i]] {id=gq}
                end
            end
            for j
                if j < diff
                    q_out[gatherIds[start + j]] = gq
                end
            end
        end
        """,
        assumptions="maxIter > 0 and n > 1",
        default_offset=None,
        name="gather_scatter"
    )
    knl = lp.precompute(knl, ["gq"])
    
    return knl

def gen_CG_iteration():
    knl = lp.make_kernel(
        """
        {[i,j,k,l]: 0<=i,j,k,l<n }
        """,
        """ 
        # Calculated with some other function
        #<> Ap[i] = sum(j, A[i,j]*p[j]) {id=Ap} 

        <> a = rdotr_prev / sum(j, p[j]*Ap[j]) {id=a}
        x[l] = x[l] + a*p[l] {id=x, dep=a}
        r[l] = r[l] - a*Ap[l] {id=r, dep=a}
        rdotr = sum(k, r[k]*r[k]) {id=rdotr, dep=r}
        p_out[i] = r[i] + (rdotr/rdotr_prev) * p[i] {id=p, dep=rdotr}
        """,
        #kernel_data = [
        #    lp.GlobalArg("result", SEMPY_SCALAR, shape=(m), order="C"),
        #    lp.GlobalArg("A", SEMPY_SCALAR, shape=(m,n), order="C"),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0",
        default_offset=None,
        name="cg"
    )

    knl = lp.make_reduction_inames_unique(knl)
    #knl = lp.duplicate_inames(knl, "i", within="id:b0*")
    #knl = lp.duplicate_inames(knl, "i", within="id:r")
    #knl = lp.duplicate_inames(knl, "i", within="id:r_old")

    return knl

def gen_apply_geometric_factors_knl():
    knl = lp.make_kernel(
        """
        {[i,j,k]: 0<=i,j<3 and 0<=k<n }
        """,
        """
        apx[i,k] = sum(j, g[e,i,j,k]*px[j,k])
        """,
        assumptions="n > 0",
        default_offset=None,
        name="geo_factors_apply"
    )

    return knl



def gen_Ax_knl():
    knl = lp.make_kernel(
        """
        {[i,j]: 0<=i<m and 0<=j<n }
        """,
        """
        result[i] = sum(j,A[i,j]*x[j])
        """,
        #kernel_data = [
        #    lp.GlobalArg("result", SEMPY_SCALAR, shape=(m), order="C"),
        #    lp.GlobalArg("A", SEMPY_SCALAR, shape=(m,n), order="C"),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0 and m > 0",
        default_offset=None,
        name="Ax"
    )

    return knl


def gen_gradient_knl():
    knl = lp.make_kernel(
        #["{[i,i0,ii,j,k,k0,kk,l]: 0<=i,i0,j,k,k0,l<n and 0<=ii, kk < nn}"],
        ["{[i,i0,ii,j,k,k0,kk,l]: 0<=i,i0,j,k,k0,l<n and 0<=ii, kk < nn}",
         "{[kkk, d0,d1,d2,kkk0]: 0<=kkk,kkk0<nnn and 0<=d0,d1,d2<3}",
         "{[it,i0t,iit,kt,k0t,kkt,lt]: 0<=it,i0t,kt,k0t,lt<n and 0<=iit, kkt < nn}",
         "{[e]: 0<=e<nElem}"],
        """
        <> nn = n*n
        <> nnn = nn*n
        for e
        with {id_prefix=grad}
            # Better to just pass in D.T?
            pr[0, ii*n + k0] = sum(j,U[ii*n + j]*D[k0,j])
            pr[1, l*nn + n*i + k] = sum(j,D[i,j]*U[l*nn + j*n + k])
            pr[2, i0*nn + kk] = sum(j,D[i0,j]*U[j*nn + kk])
        end
        W[d1,kkk] = sum(d0, g[e,d1,d0,kkk]*pr[d0,kkk]) {id=W, dep=*grad*}
        with {id_prefix=Ur,dep=W}
            Ur[0, iit*n + k0t] = sum(j,W[0,iit*n + j]*D[j,k0t])
            Ur[1, lt*nn + it*n + kt] = sum(j,D[j,it]*W[1, lt*nn + j*n + kt])
            Ur[2, i0t*nn + kkt] = sum(j,D[j,i0t]*W[2, j*nn + kkt])           
        end
        result[e,kkk0] = sum(d2, Ur[d2,kkk0]) {dep=Ur*}
        end
        """,
        kernel_data = [
            lp.GlobalArg("U", SEMPY_SCALAR, shape=(n*n*n,), order="C"),
            lp.GlobalArg("D", SEMPY_SCALAR, shape=(n,n), order="C"),
            lp.GlobalArg("result", SEMPY_SCALAR, shape=(nElem, n*n*n), order="C"),
            lp.GlobalArg("g", SEMPY_SCALAR, shape=(nElem,3,3,n*n*n), order="C"), 
            # If fix params
            lp.GlobalArg("Ur", SEMPY_SCALAR, shape=(3,n*n*n), order="C"),
            lp.GlobalArg("W", SEMPY_SCALAR, shape=(3,n*n*n,), order="C"),
            lp.GlobalArg("pr", SEMPY_SCALAR, shape=(3,n*n*n), order="C"), 
            lp.ValueArg("n", np.int32),
            lp.ValueArg("nElem", np.int32),
        ],
        assumptions="n > 0 and nn > 0",
        default_offset=None,
        name="grad"
    )
    knl = lp.make_reduction_inames_unique(knl)

    return knl


def gen_mxm_knl():
    knl = lp.make_kernel(
        """
        {[i,j,k]: 0<=i<n, 0<=j<m, 0<=k<o }
        """,
        """
        result[i,k] = sum(j,A[i,j]*X[j,k])
        """,
        #kernel_data = [
        #    lp.GlobalArg("result", SEMPY_SCALAR, shape=(m), order="C"),
        #    lp.GlobalArg("A", SEMPY_SCALAR, shape=(m,n), order="C"),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0 and m > 0 and o > 0",
        default_offset=None,
        name="mxm"
    )

    return knl
    

def gen_tensor_product_2dx3d_knl():
    knl = lp.make_kernel(
        """
        {[i,j,k,l]: 0<=i,j,k,l<n }
        """,
        """
        result[l,i,k] = sum(j,A2d[i,j]*X3d[l,j,k])
        """,
        #kernel_data = [
        #    lp.GlobalArg("result", SEMPY_SCALAR, shape=(m), order="C"),
        #    lp.GlobalArg("A", SEMPY_SCALAR, shape=(m,n), order="C"),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0",
        default_offset=None,
        name="tensor_product_2dx3d"
    )

    return knl
   
def gen_triple_vector_sum_knl():
    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n }
        """,
        """
        result[i] = v0[i] + v1[i] + v2[i]
        """,
        assumptions="n > 0",
        default_offset=None,
        name="triple_vector_sum"
    )

    return knl

def gen_norm_knl():
    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        result = sum(i,x[i]*x[i])
        """,
        #kernel_data = [
        #    lp.ValueArg("result", SEMPY_SCALAR),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C"),
        #    lp.GlobalArg("y", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0",
        default_offset=None,
        name="norm"
    )

    return knl


def gen_inner_product_knl():
    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        result = sum(i,x[i]*y[i])
        """,
        #kernel_data = [
        #    lp.ValueArg("result", SEMPY_SCALAR),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C"),
        #    lp.GlobalArg("y", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0",
        default_offset=None,
        name="inner_product"
    )

    return knl

def gen_weighted_inner_product_knl():
    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        result = sum(i,w[i]*x[i]*y[i])
        """,
        #kernel_data = [
        #    lp.ValueArg("result", SEMPY_SCALAR),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C"),
        #    lp.GlobalArg("y", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0",
        default_offset=None,
        name="weighted_inner_product"
    )

    return knl

def gen_weighted_norm_knl():
    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        result = sum(i,w[i]*x[i]*x[i])
        """,
        #kernel_data = [
        #    lp.ValueArg("result", SEMPY_SCALAR),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C"),
        #    lp.GlobalArg("y", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0",
        default_offset=None,
        name="weighted_norm"
    )

    return knl

def gen_inplace_xpay_knl():

    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        x[i] = x[i] + a*y[i]
        """,
        #kernel_data = [
        #    lp.GlobalArg("result", SEMPY_SCALAR, shape=(n,), order="C"),
        #    lp.ValueArg("a", SEMPY_SCALAR),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C"),
        #    lp.GlobalArg("y", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0",
        default_offset=None,
        name="inplace_xpay"
    )

    #knl = lp.tag_inames(knl, [("i", "g.0")])

    return knl

def gen_inplace_axpy_knl():

    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        x[i] = a*x[i] + y[i]
        """,
        #kernel_data = [
        #    lp.GlobalArg("result", SEMPY_SCALAR, shape=(n,), order="C"),
        #    lp.ValueArg("a", SEMPY_SCALAR),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C"),
        #    lp.GlobalArg("y", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0",
        default_offset=None,
        name="inplace_axpy"
    )

    #knl = lp.tag_inames(knl, [("i", "g.0")])

    return knl

def gen_axpy_knl():

    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        result[i] = a*x[i] + y[i]
        """,
        #kernel_data = [
        #    lp.GlobalArg("result", SEMPY_SCALAR, shape=(n,), order="C"),
        #    lp.ValueArg("a", SEMPY_SCALAR),
        #    lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C"),
        #    lp.GlobalArg("y", SEMPY_SCALAR, shape=(n,), order="C")
        #],
        assumptions="n > 0",
        default_offset=None,
        name="axpy"
    )

    #knl = lp.tag_inames(knl, [("i", "g.0")])

    return knl

if __name__ == "__main__":
    # setup
    # -----
    lp.set_caching_enabled(False)
    from warnings import filterwarnings, catch_warnings
    filterwarnings('error', category=lp.LoopyWarning)
    import loopy.options
    loopy.options.ALLOW_TERMINAL_COLORS = False

    # Add to path so can import from above directory
    import sys
    sys.path.append('../')
    from sempy_types import SEMPY_SCALAR

    platform = cl.get_platforms()
    my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
    #ctx = cl.Context(devices=my_gpu_devices)
    ctx = cl.create_some_context(interactive=True)
    queue = cl.CommandQueue(ctx)

    n = np.int32(10)
    nElem=np.int32(10)
    grad = gen_gradient_knl()
    print(grad)
    grad = lp.set_options(grad, "write_code")
    U = np.random.rand(n*n*n)
    D = np.random.rand(n,n)
    g = np.random.rand(nElem,3,3,n*n*n)
    evt, (pr,W,Ur,result) = grad(queue, D=D, U=U, g=g,n=n,nElem=nElem)
    print(pr)
    print(W)
    """
    g_app = gen_apply_geometric_factors_knl()
    print(g_app)
    n = 10
    G = np.random.rand(2,3,3,n)
    P = np.random.rand(3,n)
    #result = np.empty_like(V)
    g_app = lp.set_options(g_app, "write_code")
    evt, (result,) = g_app(queue,e=np.int32(1),g=G,px=P)
    print(result)
    print()
    R = np.empty_like(P)
    for i in range(3):
        R[i,:] = np.sum(G[1,i,:,:]*P[:,:], axis=0)
    print(R)
    #print(result)
    #R = np.empty_like(V)
    #for i in range(n):
    #    R[i,:,:] = D@V[i,:,:]
    #print(R)
    """
    """
    mxm = gen_mxm_knl()
    print(mxm)
    n = 5
    V = np.random.rand(n,n)
    D = np.random.rand(n,n)
    #result = np.empty_like(V)
    mxm = lp.set_options(mxm, "write_code")
    evt, (result,) = mxm(queue, A=D, X=V)
    print(result)
    print(D@V)
    #R = np.empty_like(V)
    #for i in range(n):
    #    R[i,:,:] = D@V[i,:,:]
    #print(R)
    """
    """
    tp = gen_tensor_product_2dx3d_knl()
    print(tp)
    n = 3
    V = np.random.rand(n,n,n)
    D = np.random.rand(n,n)
    result = np.empty_like(V)
    tp = lp.set_options(tp, "write_code")
    evt, (result,) = tp(queue, A2d=D, X3d=V)
    print(result)
    R = np.empty_like(V)
    for i in range(n):
        R[i,:,:] = D@V[i,:,:]
    print(R)
    """
    """
    zeroBoundary = gen_zero_boundary_knl()
    dofs = np.random.rand(10)
    boundaryIndices = np.array([0,5], dtype=np.int32)
    result = zeroBoundary(queue, dofs=dofs, boundaryIndices=boundaryIndices)
    print(result)
    """
    """
    wip = gen_weighted_inner_product_knl()dient_knl()
    print(grad)
    print(wip)
    w_nrm = gen_weighted_norm_knl()
    print(w_nrm)
    """
    """
    gs = gen_gather_scatter_knl()
    print(gs)
    gs = lp.set_options(gs, "write_code")

    maxIter = 1
    # Note that the last entry mote be the length of the list
    gatherStarts=np.array([0,1,2,3,4], dtype=np.int32)
    gatherIds = np.array([0,1,2,3], dtype=np.int32)
    q_in = np.array([0.5,0.5,1.0,1.0])

    evt, output = gs(queue, maxIter=maxIter, gatherStarts=gatherStarts, \
            gatherIds=gatherIds, q_in=q_in,q_out=np.empty_like(q_in))
    print(output)

    result =  python_gather_scatter(gatherIds, gatherStarts, maxIter, q_in)
    print(result)
    """
    """
    #norm = gen_norm_knl(100)
    #print(norm)
    #Ax = gen_Ax_knl(100,100)
    #print(Ax)
    #inner_product = gen_inner_product_knl(100)
    #print(inner_product)
    axpy = gen_axpy_knl(100)
    print(axpy)
    a = SEMPY_SCALAR(1.0)
    x = np.ones(100, SEMPY_SCALAR)
    y = np.ones(100, SEMPY_SCALAR)
    result = np.empty(100, SEMPY_SCALAR)
    #(evt, result) = axpy(queue, a=a, x=x, y=y)
    print(result)
    #lp.set_options(axpy, "no_numpy")
    #lp.set_options(axpy, edit_code=True)
    #axpy = axpy.copy(target=lp.CudaTarget())
    #axpy_code = lp.generate_code_v2(axpy).device_code()
    #print(axpy_code)
    """
    """
    cg = gen_CG_iteration()
    lp.set_options(cg, "write_code")
    print(cg)
    Ap = np.ones(100,dtype=SEMPY_SCALAR)
    p = np.ones(100, dtype=SEMPY_SCALAR)
    r = np.ones(100, dtype=SEMPY_SCALAR)
    x = np.ones(100, dtype=SEMPY_SCALAR)
    result = cg(queue, Ap=Ap, p=p, r=r, rdotr_prev=SEMPY_SCALAR(1.0), x=x)
    print(result)
    #cg_code = lp.generate_code_v2(cg).device_code()
    #print(cg_code)
    """
