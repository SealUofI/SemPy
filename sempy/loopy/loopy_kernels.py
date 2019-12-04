import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
from loopy.kernel.data import AddressSpace

# Add to path so can import from above directory
import sys
sys.path.append('../')
from sempy_types import SEMPY_SCALAR


def gen_Ax_knl(m,n):
    knl = lp.make_kernel(
        """
        {[i,j]: 0<=i<m and 0<=j<n }
        """,
        """
        result[i] = sum(j,A[i,j]*y[j])
        """,
        kernel_data = [
            lp.GlobalArg("result", SEMPY_SCALAR, shape=(m), order="C"),
            lp.GlobalArg("A", SEMPY_SCALAR, shape=(m,n), order="C"),
            lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C")
        ],
        assumptions="n > 0 and m > 0",
        default_offset=None,
        name="inner_product"
    )

    return knl


def gen_norm_knl(n):
    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        result = sqrt(sum(i,x[i]*y[i]))
        """,
        kernel_data = [
            lp.ValueArg("result", SEMPY_SCALAR),
            lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C"),
            lp.GlobalArg("y", SEMPY_SCALAR, shape=(n,), order="C")
        ],
        assumptions="n > 0",
        default_offset=None,
        name="inner_product"
    )

    return knl


def gen_inner_prod_knl(n):
    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        result = sum(i,x[i]*y[i])
        """,
        kernel_data = [
            lp.ValueArg("result", SEMPY_SCALAR),
            lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C"),
            lp.GlobalArg("y", SEMPY_SCALAR, shape=(n,), order="C")
        ],
        assumptions="n > 0",
        default_offset=None,
        name="inner_product"
    )

    return knl

def gen_axpy_knl(n):

    knl = lp.make_kernel(
        """
        {[i]: 0<=i<n}
        """,
        """
        result[i] = a*x[i] + y[i]
        """,
        kernel_data = [
            lp.GlobalArg("result", SEMPY_SCALAR, shape=(n,), order="C"),
            lp.ValueArg("a", SEMPY_SCALAR),
            lp.GlobalArg("x", SEMPY_SCALAR, shape=(n,), order="C"),
            lp.GlobalArg("y", SEMPY_SCALAR, shape=(n,), order="C")
        ],
        assumptions="n > 0",
        default_offset=None,
        name="axpy"
    )

    return knl

'''
norm = gen_norm_knl(100)
print(norm)
Ax = gen_Ax_knl(100,100)
print(Ax)
inner_product = gen_inner_prod_knl(100)
print(inner_product)
axpy = gen_axpy_knl(100)
print(axpy)
'''
