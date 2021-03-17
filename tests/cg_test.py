import pytest
import numpy as np
from sempy.iterative import cg


def test_cg_2x2():
    A = np.array([4.0, 1.0, 1.0, 3.0])
    A = A.reshape((2, 2))

    def Ax(x):
        return np.dot(A, x)

    b = np.array([1.0, 2.0])
    x, niter = cg(Ax, b)

    assert niter <= 2
    assert (np.round(x, decimals=4) == np.array([0.0909, 0.6364])).all()


def test_cg_3x3():
    A = np.array([1.0, 2.0, -1.0, 2.0, 2.0, 2.0, 1.0, -1.0, 2.0])
    A = A.reshape((3, 3))
    A = (A + A.T) / 2.0

    def Ax(x):
        return np.dot(A, x)

    b = np.array([2.0, 12.0, 5.0])
    x, niter = cg(Ax, b)

    np_x = np.linalg.solve(A, b)

    assert niter <= 3
    assert np.allclose(x, np_x, atol=1e-8)


def test_cg_10x10():
    n = 10
    A = np.random.rand(n, n)
    A = (A + A.T) / 2.0

    def Ax(x):
        return np.dot(A, x)

    b = np.random.rand(n)
    x, niter = cg(Ax, b)

    np_x = np.linalg.solve(A, b)
    assert np.allclose(x, np_x, atol=1e-10)


def test_cg_20x20():
    n = 20
    A = np.random.rand(n, n)
    A = (A + A.T) / 2.0

    def Ax(x):
        return np.dot(A, x)

    b = np.random.rand(n)
    x, niter = cg(Ax, b)

    np_x = np.linalg.solve(A, b)
    assert np.allclose(x, np_x, atol=1e-8)
