import numpy as np


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


def metric2sizes(m):

    m = sym2mat(m)
    eigvals, eigvecs = np.linalg.eigh(m)
    eigvals = 1.0 / np.sqrt(eigvals)
    res = np.einsum("ijk,ik,ilk->ijl", eigvecs, eigvals, eigvecs)
    return mat2sym(res)
