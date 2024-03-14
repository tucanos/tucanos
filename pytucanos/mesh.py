import numpy as np
import matplotlib.pyplot as plt
from ._pytucanos import (
    Mesh21,
    Mesh22,
    Mesh32,
    Mesh33,
)
from .metric import sym2mat


def create_mesh(coords, elems, etags, faces, ftags):

    if coords.shape[1] == 2:
        return Mesh22(coords, elems, etags, faces, ftags)
    else:
        if elems.shape[1] == 3:
            return Mesh32(coords, elems, etags, faces, ftags)
        else:
            return Mesh33(coords, elems, etags, faces, ftags)


def __plot_boundary(ax, bdy, normals):

    xy = bdy.get_coords()
    edgs = bdy.get_elems()
    etags = bdy.get_etags()

    ax.scatter(xy[:, 0], xy[:, 1], c="k", marker=".")

    tags = np.unique(etags)
    labels = {}
    for e, t in zip(edgs, etags):
        (i,) = np.nonzero(tags == t)
        X, Y = xy[e, 0], xy[e, 1]
        if not t in labels:
            ax.plot(X, Y, c="C%d" % i, label=repr(t))
            labels[t] = True
        else:
            ax.plot(X, Y, c="C%d" % i)

        if normals:
            ax.arrow(
                X.mean(),
                Y.mean(),
                np.diff(Y)[0],
                -np.diff(X)[0],
                color="gray",
                linewidth=0.5,
            )

    ax.legend()


def plot_mesh(ax, msh, etag=True, boundary=True, normals=False):

    if isinstance(msh, Mesh22):

        xy = msh.get_coords()

        tris = msh.get_elems()

        if etag:
            ax.tripcolor(xy[:, 0], xy[:, 1], tris, msh.get_etags(), alpha=0.5)

        ax.triplot(xy[:, 0], xy[:, 1], tris, color="m", linewidth=0.5)

        if boundary:
            bdy, _ = msh.boundary()
            __plot_boundary(ax, bdy, normals)
    elif isinstance(msh, Mesh21):
        __plot_boundary(ax, msh, normals)

    ax.axis("scaled")


def plot_field(ax, msh, arr, loc="vertex"):

    assert isinstance(msh, Mesh22)

    xy = msh.get_coords()
    tris = msh.get_elems()

    if loc == "vertex":
        cax = ax.tricontourf(xy[:, 0], xy[:, 1], tris, arr)
    else:
        cax = ax.tripcolor(xy[:, 0], xy[:, 1], tris, arr)

    ax.triplot(xy[:, 0], xy[:, 1], tris, color="m", linewidth=0.5)

    ax.axis("scaled")

    return cax


def plot_metric(ax, msh, m, loc="vertex"):

    assert isinstance(msh, Mesh22)

    xy = msh.get_coords()
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


def get_cube():

    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    elems = np.array(
        [
            [0, 1, 2, 5],
            [0, 2, 7, 5],
            [0, 2, 3, 7],
            [0, 5, 7, 4],
            [2, 7, 5, 6],
        ],
        dtype=np.uint32,
    )
    etags = np.array([1, 1, 1, 1, 1], dtype=np.int16)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [5, 6, 7],
            [5, 7, 4],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [1, 2, 5],
            [2, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ],
        dtype=np.uint32,
    )
    ftags = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype=np.int16)

    return coords, elems, etags, faces, ftags


def get_square(two_tags=True):

    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    elems = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=np.uint32,
    )
    if two_tags:
        etags = np.array([1, 2], dtype=np.int16)
        faces = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [0, 2],
            ],
            dtype=np.uint32,
        )
        ftags = np.array([1, 2, 3, 4, 5], dtype=np.int16)
    else:
        etags = np.array([1, 1], dtype=np.int16)
        faces = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
            ],
            dtype=np.uint32,
        )
        ftags = np.array([1, 2, 3, 4], dtype=np.int16)

    return coords, elems, etags, faces, ftags
