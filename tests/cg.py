import pytest
import SemPy.iterative.cg as cg

def test_cg_2x2():
    def Ax(x):
        assert x.ndim==1
        assert x.size==2
    
        A=np.array([4.0,1.0,1.0,3.0])
        A=A.reshape((2,2))
    
        return np.dot(A,x)

    b=np.array([1.0,2.0])
    x,niter=cg(Ax,b)
    assert niter==1
    assert x==np.array([0.,0.])
