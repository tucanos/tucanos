import numpy as np
import matplotlib.pyplot as plt
from pytucanos import Mesh2d, implied_metric
from pytucanos.mesh import get_square, plot_mesh
from pytucanos.metric import plot_metric


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
    msh = msh.split().split()

    # add the missing boundaries, & orient them outwards
    msh.fix()

    # Hilbert renumbering
    msh.reorder_hilbert()

    m_i = implied_metric(msh)
    fig, ax = plt.subplots()
    plot_mesh(ax, msh)
    plot_metric(ax, msh, m_i, loc="vertex")
    ax.set_title("Implied metric - vertex")

    plt.show()
