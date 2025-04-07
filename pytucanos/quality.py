import numpy as np
from .edges import edges, TRI2EDG, TET2EDG
from .metric import sym2mat


def qualities_and_lengths(mesh, m):

    coords = mesh.get_verts()
    els = mesh.get_elems()

    return qualities(coords, els, m), lengths(coords, els, m)


def lengths(coords, els, m):

    edgs = edges(els)
    aniso = m.ndim == 2 and m.shape[1] > 1

    tmp = coords[edgs[:, 1], :] - coords[edgs[:, 0], :]
    if not aniso:
        m = m.squeeze()
        l = np.linalg.norm(tmp, axis=1)
        l0 = l / m[edgs[:, 0]]
        l1 = l / m[edgs[:, 1]]
    else:
        mm = sym2mat(m)
        l0 = np.einsum("ik,ikl,il->i", tmp, mm[edgs[:, 0]], tmp) ** 0.5
        l1 = np.einsum("ik,ikl,il->i", tmp, mm[edgs[:, 1]], tmp) ** 0.5

    r = l0 / l1
    fac = np.ones(r.size)
    (idx,) = np.nonzero(np.abs(r - 1) > 0.01)
    fac[idx] = (r[idx] - 1) / r[idx] / np.log(r[idx])

    return fac * l0


def qualities(coords, els, m):

    aniso = m.ndim == 2 and m.shape[1] > 1
    dim = coords.shape[1]

    e1 = coords[els[:, 1], :] - coords[els[:, 0], :]
    e2 = coords[els[:, 2], :] - coords[els[:, 0], :]

    if dim == 2:
        vol = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    else:
        n = np.cross(e1, e2)
        e3 = coords[els[:, 3], :] - coords[els[:, 0], :]
        vol = np.einsum("ij,ij->i", e3, n) / 6.0

    if aniso:
        mm = sym2mat(m)
        v = 1.0 / np.sqrt(np.linalg.det(mm))

        idx = v[els].argmin(axis=1)
        tmp = els.shape[1] * np.arange(idx.size) + idx
        vidx = els.ravel()[tmp]
        m = mm[vidx, :, :]
        vol /= v[vidx]

    l = np.zeros(els.shape[0])
    el2edg = TRI2EDG if dim == 2 else TET2EDG
    for e in el2edg:
        tmp = coords[els[:, e[1]], :] - coords[els[:, e[0]], :]
        if not aniso:
            l += (np.linalg.norm(tmp, axis=1)) ** 2
        else:
            l += np.einsum("ik,ikl,il->i", tmp, m, tmp)

    q = vol ** (2 / dim) / l

    if dim == 2:
        ideal_vol = 3**0.5 / 4
        n_edgs = 3
    else:
        ideal_vol = 1.0 / (6 * 2**0.5)
        n_edgs = 6

    q /= ideal_vol ** (2 / dim) / n_edgs
    return q
