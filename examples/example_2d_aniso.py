import numpy as np
import matplotlib.pyplot as plt
from pytucanos.mesh import get_square, Mesh22, plot_mesh
from pytucanos.geometry import LinearGeometry2d
from pytucanos.remesh import Remesher2dAniso


def get_m(msh):

    x, y = msh.get_coords().T

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
    msh = msh.split().split().split().split()

    # add the missing boundaries, & orient them outwards
    msh.add_boundary_faces()

    # Hilbert renumbering
    msh.reorder_hilbert()

    for _ in range(3):
        m = get_m(msh)
        geom = LinearGeometry2d(msh)
        remesher = Remesher2dAniso(msh, geom, m)
        remesher.remesh(
            collapse_constrain_q=0.5,
            split_constrain_q=0.5,
            smooth_type="laplacian",
            smooth_iter=2,
        )

        msh = remesher.to_mesh()

    fig, ax = plt.subplots()
    plot_mesh(ax, msh)
    ax.set_title("Adapted")

    plt.show()
