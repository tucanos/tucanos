import numpy as np
from . import Mesh2d, Mesh3d, BoundaryMesh2d, BoundaryMesh3d, Idx

TRI2EDG = [
    [0, 1],
    [1, 2],
    [2, 0],
]

TET2EDG = [
    [0, 1],
    [1, 2],
    [2, 0],
    [0, 3],
    [1, 3],
    [2, 3],
]


def edges(els):
    assert els.ndim == 2
    if els.shape[1] == 2:
        edgs = els.copy()
    elif els.shape[1] == 3:
        edgs = np.vstack([els[:, e] for e in TRI2EDG])
    elif els.shape[1] == 4:
        edgs = np.vstack([els[:, e] for e in TET2EDG])

    edgs.sort(axis=1)
    return np.unique(edgs, axis=0)


def create_mesh(coords, elems, etags, faces, ftags):
    if coords.shape[1] == 2:
        return Mesh2d(coords, elems, etags, faces, ftags)
    else:
        if elems.shape[1] == 3:
            return BoundaryMesh3d(coords, elems, etags, faces, ftags)
        else:
            return Mesh3d(coords, elems, etags, faces, ftags)


def __plot_boundary(ax, bdy, normals):
    xy = bdy.get_verts()
    edgs = bdy.get_elems()
    etags = bdy.get_etags()

    ax.scatter(xy[:, 0], xy[:, 1], c="k", marker=".")

    tags = np.unique(etags)
    labels = {}
    for e, t in zip(edgs, etags):
        (i,) = np.nonzero(tags == t)
        X, Y = xy[e, 0], xy[e, 1]
        if t not in labels:
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
    if isinstance(msh, Mesh2d):
        xy = msh.get_verts()

        tris = msh.get_elems()

        if etag:
            ax.tripcolor(xy[:, 0], xy[:, 1], tris, msh.get_etags(), alpha=0.5)

        ax.triplot(xy[:, 0], xy[:, 1], tris, color="m", linewidth=0.5)

        if boundary:
            bdy, _ = msh.boundary()
            __plot_boundary(ax, bdy, normals)
    elif isinstance(msh, BoundaryMesh2d):
        __plot_boundary(ax, msh, normals)

    ax.axis("scaled")


def plot_field(ax, msh, arr, loc="vertex"):
    assert isinstance(msh, Mesh2d)

    xy = msh.get_verts()
    tris = msh.get_elems()

    if loc == "vertex":
        cax = ax.tricontourf(xy[:, 0], xy[:, 1], tris, arr)
    else:
        cax = ax.tripcolor(xy[:, 0], xy[:, 1], tris, arr)

    ax.triplot(xy[:, 0], xy[:, 1], tris, color="m", linewidth=0.5)

    ax.axis("scaled")

    return cax


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
        dtype=Idx,
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
        dtype=Idx,
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
        dtype=Idx,
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
            dtype=Idx,
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
            dtype=Idx,
        )
        ftags = np.array([1, 2, 3, 4], dtype=np.int16)

    return coords, elems, etags, faces, ftags
