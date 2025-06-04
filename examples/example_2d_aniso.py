import numpy as np
import matplotlib.pyplot as plt
from pytucanos.mesh import get_square, Mesh22, plot_mesh
from pytucanos.geometry import LinearGeometry2d
from pytucanos.remesh import Remesher2dAniso, PyRemesherParams


def get_m(msh):
    x, y = msh.get_verts().T

    hx = 0.1
    h0 = 0.001
    hy = h0 + 2 * (0.1 - h0) * abs(y - 0.5)

    m = np.zeros((x.size, 3))
    m[:, 0] = 1.0 / hx**2
    m[:, 1] = 1.0 / hy**2

    return m


if __name__ == "__main__":
    coords, elems, etags, faces, ftags = get_square()
    msh = Mesh22(coords, elems, etags, faces, ftags)

    # add the missing boundaries, & orient them outwards
    msh.add_boundary_faces()

    # Hilbert renumbering
    msh.reorder_hilbert()

    msh.compute_topology()
    geom = LinearGeometry2d(msh)
    for _ in range(5):
        m = get_m(msh)
        remesher = Remesher2dAniso(msh, geom, m)
        remesher.remesh(geom, params=PyRemesherParams.default())

        msh = remesher.to_mesh()
        msh.compute_topology()
    fig, ax = plt.subplots()
    plot_mesh(ax, msh)
    ax.set_title("Adapted")

    qualities = remesher.qualities()
    lengths = remesher.lengths()

    fig, ax = plt.subplots(2, 1, tight_layout=True)
    ax[0].hist(
        qualities,
        bins=50,
        alpha=0.25,
        density=True,
        label="parmesan (q_min = %.2f)" % qualities.min(),
    )
    ax[0].set_xlabel("quality")
    ax[0].legend()
    ax[1].hist(
        lengths,
        bins=50,
        alpha=0.25,
        density=True,
        label="parmesan (l_min = %.2f, l_max = %.2f)" % (lengths.min(), lengths.max()),
    )
    ax[1].axvline(x=0.5**0.5, c="r")
    ax[1].axvline(x=2**0.5, c="r")
    ax[1].set_xlabel("edge lengths")
    ax[1].legend()

    plt.show()
