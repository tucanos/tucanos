import numpy as np
from . import Mesh2d


def plot_metric(ax, msh, m, loc="vertex"):
    assert isinstance(msh, Mesh2d)

    xy = msh.get_verts()
    tris = msh.get_elems()

    m = sym2mat(m)
    t = np.linspace(0, 2 * np.pi, 30)

    res = np.zeros((30, 2))
    if loc != "vertex":
        xy = xy[tris, :].mean(axis=1)

    for i, (x, y) in enumerate(xy):
        eigvals, eigvecs = np.linalg.eigh(m[i, :, :])
        sizes = 0.25 * 1.0 / eigvals**0.5
        for i in range(2):
            res[:, i] = (
                sizes[0] * np.cos(t) * eigvecs[i, 0]
                + sizes[1] * np.sin(t) * eigvecs[i, 1]
            )
        ax.plot(x + res[:, 0], y + res[:, 1], "k")


def sym2mat(m):
    if m.shape[1] == 3:
        dim = 2
    elif m.shape[1] == 6:
        dim = 3
    else:
        raise ValueError("Invalid shape")

    mm = np.zeros((m.shape[0], dim, dim))
    if dim == 2:
        mm[:, 0, 0] = m[:, 0]
        mm[:, 1, 1] = m[:, 1]
        mm[:, 0, 1] = m[:, 2]
        mm[:, 1, 0] = m[:, 2]
    else:
        mm[:, 0, 0] = m[:, 0]
        mm[:, 1, 1] = m[:, 1]
        mm[:, 2, 2] = m[:, 2]
        mm[:, 0, 1] = m[:, 3]
        mm[:, 1, 0] = m[:, 3]
        mm[:, 1, 2] = m[:, 4]
        mm[:, 2, 1] = m[:, 4]
        mm[:, 0, 2] = m[:, 5]
        mm[:, 2, 0] = m[:, 5]

    return mm


def mat2sym(m):
    if m.shape[1] == 2:
        dim = 3
    elif m.shape[1] == 3:
        dim = 6
    else:
        raise ValueError("Invalid shape")

    mm = np.zeros((m.shape[0], dim))
    if dim == 3:
        mm[:, 0] = m[:, 0, 0]
        mm[:, 1] = m[:, 1, 1]
        mm[:, 2] = m[:, 0, 1]
        mm[:, 2] = m[:, 1, 0]
    else:
        mm[:, 0] = m[:, 0, 0]
        mm[:, 1] = m[:, 1, 1]
        mm[:, 2] = m[:, 2, 2]
        mm[:, 3] = m[:, 0, 1]
        mm[:, 3] = m[:, 1, 0]
        mm[:, 4] = m[:, 1, 2]
        mm[:, 4] = m[:, 2, 1]
        mm[:, 5] = m[:, 0, 2]
        mm[:, 5] = m[:, 2, 0]

    return mm


def anisotropy(m):
    m = sym2mat(m)
    eigvals, _ = np.linalg.eigh(m)
    sizes = 1.0 / np.sqrt(eigvals)
    return sizes.max(axis=1) / sizes.min(axis=1)


def bound_anisotropy(m, aniso_max):
    m = sym2mat(m)

    eigvals, eigvecs = np.linalg.eigh(m)
    sizes = 1.0 / np.sqrt(eigvals)
    min_sizes = sizes.min(axis=1)
    for i in range(eigvals.shape[1]):
        sizes[:, i] = np.minimum(sizes[:, i], aniso_max * min_sizes)
    eigvals = 1.0 / sizes**2
    res = np.einsum("ijk,ik,ilk->ijl", eigvecs, eigvals, eigvecs)

    return mat2sym(res)


def metric2sizes(m):
    m = sym2mat(m)
    eigvals, eigvecs = np.linalg.eigh(m)
    eigvals = 1.0 / np.sqrt(eigvals)
    res = np.einsum("ijk,ik,ilk->ijl", eigvecs, eigvals, eigvecs)
    return mat2sym(res)
