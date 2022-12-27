from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._interface import _get_dtype, MatrixLinearOperator
from typing import Sequence, Iterable, Union
#from dataclasses import dataclass
#from numpy.typing import ArrayLike
from scipy.sparse import diags

ScalarType = Union[np.number, float, int]

#@dataclass
class KroneckerProductOperator(LinearOperator):
    # An operator scalar*(A x B x C ...)

    def __init__(self, *args, order:str='F'):#, scalar:ScalarType=1):
        self.args = tuple([aslinearoperator(arg) for arg in args])
        self.order=order # Order for the fast kronecker matvecs. Is this actually needed?

    @property
    def shape(self):
        return tuple(np.product([term.shape for term in self.args], axis=0))
    
    @property
    def dtype(self):
        return _get_dtype(self.args)

    def _matvec(self, v):
        if len(self.args) == 1:
            return self.args[0]@v
        elif len(self.args) == 2:
            return kron_2d(self.args[1], self.args[0], v, order=self.order)
            #return kron_2d(*self.args, v, order=self.order)
            #return kron_2d(*[arg.A if isinstance(arg, MatrixLinearOperator) else arg for arg in self.args], v, order=self.order)

        elif len(self.args) == 3:
            return kron(self.args[2], self.args[1], self.args[0], v, order=self.order)
            #return kron(*[arg.A if isinstance(arg, MatrixLinearOperator) else arg for arg in self.args], v, order=self.order)
        else:
            raise NotImplementedError("Kronecker matvecs with more than three args not implemented")

    def _adjoint(self):
        return KroneckerProductOperator(*[term.H for term in self.args], order=self.order)

    def _transpose(self):
        # Is this true in general?
        return self._adjoint()   

    def dot(self, x):
        if isinstance(x, KroneckerProductOperator):
            return self._matmat(x)
        else:
            return super().dot(x)
    
    def _matmat(self, X):
        #print("HERE I AM")
        if isinstance(X, KroneckerProductOperator) and all([A.shape[1] == B.shape[0] for (A,B) in zip(self.args, X.args)]):
            if all([isinstance(A, MatrixLinearOperator) and isinstance(B, MatrixLinearOperator) for (A,B) in zip(self.args, X.args)]):
                return KroneckerProductOperator(*[A.A@B.A for (A, B) in zip(self.args, X.args)], order=self.order)
            else:
                return KroneckerProductOperator(*[A@B for (A, B) in zip(self.args, X.args)], order=self.order)
        #elif isinstance(X, KroneckerProductSumOperator):
        #    return KroneckerProductSumOperator(*[self@term for term in X.args], order=self.order)
        else:
            # Should probably handle the dense mxm case
            return super()._matmat(X)

    """
    def __mul__(self, x):
        if np.isscalar(x):
            return KroneckerProductOperator(*(self.args), scalar=x*self.scalar, order=self.order)
        else:
            return super().__mul__(x)

    def __add__(self, X):
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductSumOperator(*[self, X], order=self.order)
        elif isinstance(X, KroneckerProductSumOperator):
            return KroneckerProductSumOperator(*([self] + X.args), order=self.order)
        elif 
        else:
            raise NotImplementedError("Addition only supported with other KroneckerProduct objects")

    def __sub__(self, X):
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductSumOperator(*[self, X*-1], order=self.order)
        elif isinstance(X, KroneckerProductSumOperator):
            return KroneckerProductSumOperator(*([self] + [term*-1 for term in X.args]), order=self.order)
        else:
            raise NotImplementedError("Subtraction only supported with other KroneckerProduct objects")

    # Would require forming any KroneckerProductSum operators
    #def inv(self):
    #    return KroneckerProductOperator([term.inv for term in self.args], scalar=1/self.scalar, order=order)

    # These may not be defined for arbitary LinearOperator objects

    def diagonal(self, as_operator=False):
        if as_operator:
            return KroneckerProductOperator(*[diags(term.diagonal(), format="csr") for term in self.args], scalar=self.scalar, order=self.order)
        else:
            return KroneckerProductOperator(*[term.diagonal() for term in self.args], scalar=self.scalar, order=self.order).to_sparse().A

    def block_diagonal(self, keep_intact=None):
        if intact is None:
            keep_intact = {len(self.args)-1}
        return KroneckerProductOperator(*[(diags(term.diagonal(), format="csr") if i not in keep_intact else term) for i, term in enumerate(self.args)], scalar=self.scalar, order=self.order)

    # Add some functions to obtain the block lower and upper triangular portions?
    #def block_triu(self, ks:Sequence=None, keep_intact:Sequence=None):
    #    if ks is None:
    #        ks = np.zeros((len(self.args),))
    #    components = 
  
    # TODO: Make this work recursively
    """
    def to_sparse(self, format=None):
        from scipy.sparse import kron as spkron
        result = self.args[0].A
        for term in self.args[1:]:
            result = spkron(result, term.A, format=format)
        print(result.A)
        return result
    """ 
    def to_dense(self):
        from numpy import kron as npkron
        result = self.scalar*self.args[0]
        for term in self.args[1:]:
            result = npkron(result, term)
        return result
    """
       
"""
class KroneckerProductSumOperator(LinearOperator):
    # An operator A + B + C ... where A,B,C are KroneckerProductOperators of the same shape
    #args: Iterable[KroneckerProductOperator, KroneckerProductSumOperator]
    #order: Optional[str] = None

    # Perhaps __init__ or post_init should attempt to combine args
    # Nothing here currently asserts an ordering on the args
    def __init__(self, *args: Iterable[Union[KroneckerProductOperator, KroneckerProductSumOperator]], order:str='F', scalar:ScalarType=1):
        self.args = args
        self.scalar=scalar
        self.order=order
 
    @property
    def shape(self):
        return args[0].shape       
 
    @property
    def dtype(self):
        return np.result_type(*(self.args))

    def _matvec(self, v):
        v = args[0]@v
        for term in self.args[1:]:
            v += term@v
        return v

    def _adjoint(self):
        return KroneckerProductSumOperator(*[term.H for term in self.args], order=order)

    def _matmat(self, X):
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductSumOperator(*[term@X for term in self.args], order=self.order)
        #elif isinstance(X, KroneckerProductSumOperator): # Not a great implementation, probably ill-advised to use
            #from itertools import combinations
            #return KroneckerProductSumOperator([A@B for A, B in combinations(self.args, X)], order=self.order)
        else:
            return super()._matmat(X)

    def __mul__(self, x):
        if np.isscalar(x):
            return KroneckerProductSumOperator(*[term*x for term in self.args] , order=order)
        else:
            raise NotImplementedError("Only scalar multiplication supported")

    def __add__(self, X):
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductSumOperator(*(self.args + [X]), order=self.order)
        elif isinstance(X, KroneckerProductSumOperator):
            return KroneckerProductSumOperator(*(self.args + X.args), order=self.order)
        else:
            raise NotImplementedError("Addition only supported with other KroneckerProduct objects")

    def __sub__(self, X):
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductSumOperator(*(self.args + [X*-1]), order=self.order)
        elif isinstance(X, KroneckerProductSumOperator):
            return KroneckerProductSumOperator(*(self.args + [term*-1 for term in X.args]), order=self.order)
        else:
            raise NotImplementedError("Subtraction only supported with other KroneckerProduct objects")

    def diagonal(self):
        return KroneckerProductSumOperator(*[term.diagonal() for term in self.args], order=self.order)

    # Don't necessarily have to use the final term.
    def block_diagonal(self, keep_intact=None):
        return KroneckerProductSumOperator(*[term.block_diagonal(keep_intact=keep_intact) for term in self.args], order=self.order)

    def to_sparse(self, format=None):
        result = self.args[0].to_sparse(format=format)
        for term in self.args[1:]:
            result += term.to_sparse(format=format)
        return result

"""
def kron_2d(Sy, Sx, U, order='F'):
    nx, mx = Sx.shape
    ny, my = Sy.shape

    U = U.reshape((my, mx), order=order)
    if all([isinstance(X, np.ndarray) for X in [Sy, Sx]]):
        U = np.einsum('ai,ij,bj->ab', Sy, U, Sx, order=order, optimize=True)
    else:
        U = Sy @ U @ Sx.T

    return U.reshape((nx * ny,), order=order)


def kron(Sz, Sy, Sx, U, order='F'):
    nx, mx = Sx.shape
    ny, my = Sy.shape
    nz, mz = Sz.shape

    #print(type(Sz))
    #print(type(Sy))
    #print(type(Sx))
    #print(type(U))
    #print("HERE")

    if isinstance(Sx, MatrixLinearOperator):
        Sx = Sx.A
    if isinstance(Sx, MatrixLinearOperator):
        Sy = Sy.A
    if isinstance(Sx, MatrixLinearOperator):
        Sz = Sz.A

    if all([isinstance(X, np.ndarray) for X in [Sz, Sy, Sx]]):
        U = U.reshape((mz, my, mx), order=order)
        U = np.einsum('ai,bj,ijk,ck->abc', Sz, Sy, U,
                      Sx, order=order, optimize=True)
    #elif all([isinstance(X, MatrixLinearOperator) and isinstance(X.A, np.ndarray) for X in [Sz, Sy, Sx]]):
    #    U = U.reshape((mz, my, mx), order=order)
    #    U = np.einsum('ai,bj,ijk,ck->abc', Sz.A, Sy.A, U,
    #                  Sx.A, order=order, optimize=True)
    else:
        U = U.reshape((my * mz, mx), order=order)

        #print(type(U))
        #print(type(Sx))
        #print(Sx.T.shape, U.shape)

        U = U @ Sx.T
        U = U.reshape((mz, my, nx), order=order)

        if isinstance(Sy, np.ndarray):
            V = np.einsum('mj,ijk->imk', Sy, U, order=order, optimize=True)
        else:
            V = np.empty((mz, ny, nx), order=order)
            for i in range(mz):
                V[i, :, :] = Sy @ U[i, :, :]

        V = V.reshape((mz, nx * ny), order=order)
        U = Sz @ V

    return U.reshape((nx * ny * nz,), order=order)
