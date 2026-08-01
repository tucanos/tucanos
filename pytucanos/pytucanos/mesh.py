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


def calibrate_mid_edge_columns(qm, tol=1e-9):
    """
    Determine which connectivity column of a `QuadraticMesh2d` (Tri6) holds
    the mid-edge node for each corner pair.

    `to_quadratic()` does not document which of its extra columns holds
    which edge's mid-node, so this discovers it empirically: on a
    straight-sided quadratic mesh (fresh out of `to_quadratic()`, before any
    curving/projection), every mid-edge node is still the exact arithmetic
    midpoint of its corner pair.

    Args:
        qm: `QuadraticMesh2d` produced by `Mesh2d.to_quadratic()` on a
            straight-sided mesh, so every mid-edge node is still the exact
            arithmetic midpoint of its corner pair.
        tol: max allowed distance between a candidate column and the
            arithmetic midpoint for a match.

    Returns:
        Dict mapping each corner pair `(0, 1)`, `(1, 2)`, `(2, 0)` to the
        connectivity column index (3, 4, or 5) holding that edge's mid-node.

    Raises:
        RuntimeError: if some corner pair can't be matched to any column
            within `tol` (the mesh isn't actually straight-sided, or
            `to_quadratic()`'s column convention changed).
    """
    elems = qm.get_elems()
    verts = qm.get_verts()
    col_for_pair = {}
    for pair in TRI2EDG:
        pair = tuple(pair)
        a, b = pair
        target = 0.5 * (verts[elems[:, a]] + verts[elems[:, b]])
        for col in (3, 4, 5):
            if np.abs(verts[elems[:, col]] - target).max() < tol:
                col_for_pair[pair] = col
                break
        else:
            raise RuntimeError(
                f"could not match corner pair {pair} to a mid-edge column"
            )
    assert set(col_for_pair.values()) == {3, 4, 5}, col_for_pair
    return col_for_pair


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
        i = np.nonzero(tags == t)[0][0]
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
