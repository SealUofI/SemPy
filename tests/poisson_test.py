import numpy as np

from sempy.elliptic import elliptic_cg
from sempy.gradient import gradient, gradient_transpose
from sempy.iterative import cg, pcg
from sempy.mesh import load_mesh


def test_single_element_pcg_sin_3d():
    N = 10
    n = N + 1

    mesh = load_mesh("box001.msh")
    mesh.find_physical_coordinates(N)
    mesh.calc_geometric_factors()
    mesh.establish_global_numbering()
    mesh.setup_mask()

    G = mesh.get_geom()
    J = mesh.get_jaco()
    B = mesh.get_mass()

    def mask(W):
        W = W.reshape((n, n, n))
        W[0, :, :] = 0
        W[n - 1, :, :] = 0
        W[:, 0, :] = 0
        W[:, n - 1, :] = 0
        W[:, :, 0] = 0
        W[:, :, n - 1] = 0
        W = W.reshape((n * n * n,))
        return W

    def Ax(x):
        Ux, Uy, Uz = gradient(x, n)

        Wx = G[0, 0, 0, :] * Ux + G[0, 0, 1, :] * Uy + G[0, 0, 2, :] * Uz
        Wy = G[0, 1, 0, :] * Ux + G[0, 1, 1, :] * Uy + G[0, 1, 2, :] * Uz
        Wz = G[0, 2, 0, :] * Ux + G[0, 2, 1, :] * Uy + G[0, 2, 2, :] * Uz

        W = gradient_transpose(Wx, Wy, Wz, n)
        return mask(W)

    X = mesh.get_x()
    Y = mesh.get_y()
    Z = mesh.get_z()

    x = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    x = mask(x)

    b = (
        3
        * np.pi
        * np.pi
        * np.sin(np.pi * X)
        * np.sin(np.pi * Y)
        * np.sin(np.pi * Z)
    )
    b = b * B * J
    b = mask(b)

    Minv_ = 1.0 / (B * J)

    def Minv(r):
        return Minv_ * r

    x_pcg, niter = pcg(Ax, Minv, b, tol=1e-8, maxit=1000, verbose=0)

    assert np.allclose(x, x_pcg, 1e-8)


def test_single_element_cg_sin_3d():
    N = 10
    n = N + 1

    mesh = load_mesh("box001.msh")
    mesh.find_physical_coordinates(N)
    mesh.calc_geometric_factors()
    mesh.establish_global_numbering()
    mesh.setup_mask()

    G = mesh.get_geom()
    J = mesh.get_jaco()
    B = mesh.get_mass()

    def mask(W):
        W = W.reshape((n, n, n))
        W[0, :, :] = 0
        W[n - 1, :, :] = 0
        W[:, 0, :] = 0
        W[:, n - 1, :] = 0
        W[:, :, 0] = 0
        W[:, :, n - 1] = 0
        W = W.reshape((n * n * n,))
        return W

    def Ax(x):
        Ux, Uy, Uz = gradient(x, n)

        Wx = G[0, 0, 0, :] * Ux + G[0, 0, 1, :] * Uy + G[0, 0, 2, :] * Uz
        Wy = G[0, 1, 0, :] * Ux + G[0, 1, 1, :] * Uy + G[0, 1, 2, :] * Uz
        Wz = G[0, 2, 0, :] * Ux + G[0, 2, 1, :] * Uy + G[0, 2, 2, :] * Uz

        W = gradient_transpose(Wx, Wy, Wz, n)

        return mask(W)

    X = mesh.get_x()
    Y = mesh.get_y()
    Z = mesh.get_z()

    x = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    x = mask(x)

    b = (
        3
        * np.pi
        * np.pi
        * np.sin(np.pi * X)
        * np.sin(np.pi * Y)
        * np.sin(np.pi * Z)
    )
    b = b * B * J
    b = mask(b)

    x_cg, niter = cg(Ax, b, tol=1e-8, maxit=100, verbose=0)

    assert np.allclose(x, x_cg, 1e-8)


def test_multi_element_cg_sin_3d():
    mesh = load_mesh("box004.msh")
    N = 10
    mesh.find_physical_coordinates(N)
    mesh.establish_global_numbering()
    mesh.calc_geometric_factors()
    mesh.setup_mask()

    X = mesh.get_x()
    Y = mesh.get_y()
    Z = mesh.get_z()

    J = mesh.get_jaco()
    B = mesh.get_mass()

    x = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    x = mesh.apply_mask(x)

    b = (
        3
        * np.pi
        * np.pi
        * np.sin(np.pi * X)
        * np.sin(np.pi * Y)
        * np.sin(np.pi * Z)
    )
    b = b * B * J
    b = mesh.dssum(b)
    b = mesh.apply_mask(b)

    x_cg, niter = elliptic_cg(mesh, b, tol=1e-8, maxit=1000, verbose=0)

    assert np.allclose(x, x_cg, 1e-8)
