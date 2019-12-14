import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.kernel.data import AddressSpace

def gen_gather_scatter_knl():
    knl = lp.make_kernel(
        """
        {[k,i,j]: 0<=i,j<maxIter and 1<=k<n}
        """,
        """
        for k
            <> start = gatherStarts[k-1]
            <> diff = gatherStarts[k] - start
            <> gq = 0 {id=gq1}
            for i
                if i < diff
                    gq = gq + q_in[gatherIds[start + i]] {id=gq2}
                end
            end
            for j
                if j < diff
                    q_out[gatherIds[start + j]] = gq {dep=gq*}
                end
            end
        end
        """,
        assumptions="maxIter > 0 and n > 1",
        default_offset=None,
        name="gather_scatter"
    )
    #knl = lp.precompute(knl, ["rhs"])
    
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
        name="cg",
        target=lp.PyOpenCLTarget()
    )

    knl = lp.make_reduction_inames_unique(knl)
    #knl = lp.duplicate_inames(knl, "i", within="id:b0*")
    #knl = lp.duplicate_inames(knl, "i", within="id:r")
    #knl = lp.duplicate_inames(knl, "i", within="id:r_old")

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
        name="Ax",
        target=lp.PyOpenCLTarget()
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

    """
    wip = gen_weighted_inner_product_knl()
    print(wip)
    w_nrm = gen_weighted_norm_knl()
    print(w_nrm)
    """
    
    gs = gen_gather_scatter_knl()
    print(gs)
    gs = lp.set_options(gs, "write_code")

    evt, output = gs(queue, maxIter=2, gatherStarts=np.array([0,2,4], dtype=np.int32), \
            gatherIds=np.array([0,1,2,3],dtype=np.int32), q_in=np.array([0.5,0.5,1.0,1.0]))

    #evt, output = gs(queue, start=np.array([0,2]), end=np.array([2,4]), \
    #        gatherIds=np.array([0,1,2,3]), q_in=np.array([0.5,0.5,1.0,1.0]))
    print(output)
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
