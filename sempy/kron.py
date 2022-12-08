import numpy as np
from scipy.sparse.linalg import LinearOperator
from typing import Sequence, Iterable, Union
from dataclasses import dataclass
from numpy.typing import ArrayLike
from scipy.sparse import diags

@dataclass
class KroneckerProductOperator(LinearOperator):
    # An operator scalar*(A x B x C ...)
    # Should this be immutable?
    terms: Sequence[Union[ArrayLike, KroneckerProductOperator, KroneckerProductSumOperator]]
    order: Optional[str] = None # order for matvecs, should this be here?
    scalar: Union[np.number, float, int] = 1

    @property
    def shape(self):
        return tuple(np.product([term.shape for term in self.terms], axis=0))
        
    @property
    def dtype(self):
        return np.result_type(*([self.scalar] + self.terms))

    def _matvec(self, v):
        if self.scalar != 1
            v = self.scalar*v
        if len(self.terms) == 1:
            return self.terms[0]@v
        elif len(self.terms) == 2:
            return kron_2d(self.terms[1], self.terms[0], v, order=self.order)
        elif len(self.terms) == 3:
            return kron_2d(self.terms[2], self.terms[1], self.terms[0], v, order=self.order)
        else:
            raise NotImplementedError("Kronecker matvecs with more than three terms not implemented")

    def _adjoint(self,v):
        return KroneckerProductOperator([term.H for term in self.terms], scalar=self.scalar, order=self.order)
    
    def _matmat(self, X):
        # Not certain what order the output should have
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductOperator([A@B for (A, B) in zip(self.terms, X.terms)], scalar=self.scalar*X.scalar, order=self.order)
        elif isinstance(X, KroneckerProductSumOperator)
            return KroneckerProductSumOperator([self@term for term in X.terms], order=self.order)
        else:
            # Should probably handle the dense mxm case
            return super()._matmat(X)

    def __mul__(self, x):
        if np.isscalar(x):
            return KroneckerProductOperator(self.terms, scalar=x*self.scalar, order=order)
        else:
            raise NotImplementedError("Only scalar multiplication supported")

    def __add__(self, X):
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductSumOperator([self, X], order=self.order)
        elif isinstance(X, KroneckerProductSumOperator):
            return KroneckerProductSumOperator([self] + X.terms, order=self.order)
        else:
            raise NotImplementedError("Addition only supported with other KroneckerProduct objects")

    def __sub__(self, X):
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductSumOperator([self, X*-1], order=self.order)
        elif isinstance(X, KroneckerProductiSumOperator):
            return KroneckerProductSumOperator([self] + [term*-1 for term in X.terms], order=self.order)
        else:
            raise NotImplementedError("Subtraction only supported with other KroneckerProduct objects")

    def inv(self):
        return KroneckerProductOperator([term.inv for term in self.terms], scalar=1/self.scalar, order=order)

    def diagonal(self):
        return KroneckerProductOperator([diags(term.diagonal(), format="csr") for term in self.terms], scalar=self.scalar, order=self.order)

    def block_diagonal(self, intact=None):
        if intact is None:
            intact = len(self.terms) - 1
        return KroneckerProductOperator([(diags(term.diagonal(), format="csr") if i != intact else term) for i, term in enumerate(self.terms)], scalar=self.scalar, order=self.order)

    # Add some functions to obtain the block lower and upper triangular portions?
  
    def to_sparse(self, format=None):
        from scipy.sparse import kron as spkron
        result = spkron(self.scalar*self.terms[0], self.terms[1], format=format)
        for term in self.terms[2:]:
            result = spkron(result, term, format=format)
        return result


class KroneckerProductSumOperator(LinearOperator):
    # An operator A + B + C ... where A,B,C are KroneckerProductOperators of the same shape
    terms: Iterable[KroneckerProductOperator, KroneckerProductSumOperator]
    order: Optional[str] = None

    # Perhaps __init__ or post_init should attempt to combine terms
    # Nothing here currently asserts an ordering on the terms
 
    @property
    def shape(self):
        return terms[0].shape       
 
    @property
    def dtype(self):
        return np.result_type(*(self.terms))

    def _matvec(self, v):
        v = terms[0]@v
        for term in self.terms[1:]:
            v += term@v
        return v

    def _adjoint(self,v):
        return KroneckerProductSumOperator([term.H for term in self.terms], order=order)
    

    def _matmat(self, X):
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductSumOperator([term@X for term in self.terms], order=self.order)
        #elif isinstance(X, KroneckerProductSumOperator): # Not a great implementation, probably ill-advised to use
            #from itertools import combinations
            #return KroneckerProductSumOperator([A@B for A, B in combinations(self.terms, X)], order=self.order)
        else:
            return super()._matmat(X)

    def __mul__(self, x):
        if np.isscalar(x):
            return KroneckerProductSumOperator([term*x for term in self.terms] , order=order)
        else:
            raise NotImplementedError("Only scalar multiplication supported")

    def __add__(self, X):
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductSumOperator(self.terms + [X], order=self.order)
        elif isinstance(X, KroneckerProductSumOperator):
            return KroneckerProductSumOperator(self.terms + X.terms, order=self.order)
        else:
            raise NotImplementedError("Addition only supported with other KroneckerProduct objects")

    def __sub__(self, X):
        if isinstance(X, KroneckerProductOperator):
            return KroneckerProductSumOperator(self.terms + [X*-1], order=self.order)
        elif isinstance(X, KroneckerProductSumOperator):
            return KroneckerProductSumOperator(self.terms + [term*-1 for term in X.terms], order=self.order)
        else:
            raise NotImplementedError("Subtraction only supported with other KroneckerProduct objects")

    def diagonal(self):
        return KroneckerProductSumOperator([term.diagonal() for term in self.terms], order=self.order)

    # Don't necessarily have to use the final term.
    def block_diagonal(self, intact=None):
        return KroneckerProductSumOperator([term.block_diagonal(intact=intact) for term in self.terms], order=self.order)

    def to_sparse(self, format=None):
        result = self.terms[0].to_sparse(format=format)
        for term in self.terms[1:]:
            result += term.to_sparse(format=format)
        return result


def kron_2d(Sy, Sx, U, order=None):
    nx, mx = Sx.shape
    ny, my = Sy.shape

    U = U.reshape((my, mx), order=order)
    if all([isinstance(X, np.ndarray) for X in [Sy, Sx]]):
        U = np.einsum('ai,ij,bj->ab', Sy, U, Sx, order=order, optimize=True)
    else:
        U = Sy @ U @ Sx.T

    return U.reshape((nx * ny,), order=order)


def kron(Sz, Sy, Sx, U, order=None):
    nx, mx = Sx.shape
    ny, my = Sy.shape
    nz, mz = Sz.shape

    if all([isinstance(X, np.ndarray) for X in [Sz, Sy, Sx]]):
        U = U.reshape((mz, my, mx), order=order)
        U = np.einsum('ai,bj,ijk,ck->abc', Sz, Sy, U,
                      Sx, order=order, optimize=True)
    else:
        U = U.reshape((my * mz, mx), order=order)
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
