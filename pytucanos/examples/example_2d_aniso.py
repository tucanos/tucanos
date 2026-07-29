import matplotlib.pyplot as plt
import numpy as np
from pytucanos.mesh import (
    get_square,
    plot_mesh,
)

from pytucanos import (
    LinearGeometry2d,
    Mesh2d,
    Remesher2dAniso,
    RemesherParams,
)


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
    msh = Mesh2d(coords, elems, etags, faces, ftags)

    # add the missing boundaries, & orient them outwards
    msh.fix()

    # Hilbert renumbering
    msh.reorder_hilbert()
    bdy = msh.boundary()[0]
    geom = LinearGeometry2d(bdy)
    for _ in range(5):
        m = get_m(msh)
        remesher = Remesher2dAniso(msh, geom, m)
        remesher.remesh(geom, params=RemesherParams.default())

        msh = remesher.to_mesh()
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
        label=f"parmesan (q_min = {qualities.min():.2f})",
    )
    ax[0].set_xlabel("quality")
    ax[0].legend()
    ax[1].hist(
        lengths,
        bins=50,
        alpha=0.25,
        density=True,
        label=f"parmesan (l_min = {lengths.min():.2f}, l_max = {lengths.max():.2f})",
    )
    ax[1].axvline(x=0.5**0.5, c="r")
    ax[1].axvline(x=2**0.5, c="r")
    ax[1].set_xlabel("edge lengths")
    ax[1].legend()

    plt.show()
