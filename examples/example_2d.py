import numpy as np
import matplotlib.pyplot as plt
from pytucanos.mesh import get_square, Mesh22, plot_mesh, plot_field
from pytucanos.geometry import LinearGeometry2d
from pytucanos.remesh import Remesher2dIso


def get_h(msh):
    x, y = msh.get_coords().T
    hmin = 0.01
    hmax = 0.3
    return hmin + (hmax - hmin) * (
        1 - np.exp(-((x - 0.5) ** 2 + (y - 0.25) ** 2) / 0.25**2)
    )


if __name__ == "__main__":
    coords, elems, etags, faces, ftags = get_square()
    # Invert the first face
    faces[0, :] = faces[0, ::-1]
    msh = Mesh22(coords, elems, etags, faces, ftags)
    msh = msh.split().split().split().split()

    # add some data
    f = get_h(msh)

    # initial mesh
    fig, ax = plt.subplots()
    plot_mesh(ax, msh)
    ax.set_title("Initial")

    # add the missing boundaries, & orient them outwards
    msh.add_boundary_faces()
    fig, ax = plt.subplots()
    plot_mesh(ax, msh)
    ax.set_title("Fix boundaries")

    # Hilbert renumbering
    new_vert_indices, new_elem_indices, new_face_indices = msh.reorder_hilbert()
    f2 = np.empty(f.shape)
    f2[new_vert_indices] = f

    fig, ax = plt.subplots()
    plot_mesh(ax, msh, etag=True)
    xy = msh.get_coords()
    ax.plot(np.stack([xy[:-1, 0], xy[1:, 0]]), np.stack([xy[:-1, 1], xy[1:, 1]]), "k")
    ax.set_title("Reorder the nodes")

    # check that data is still correct
    fig, ax = plt.subplots()
    cax = plot_field(ax, msh, f2, "vertex")
    fig.colorbar(cax, ax=ax)
    ax.set_title("f")

    for _ in range(3):
        h = get_h(msh)
        msh.compute_topology()
        geom = LinearGeometry2d(msh)
        remesher = Remesher2dIso(msh, geom, h.reshape((-1, 1)))
        remesher.remesh(geom)

        msh = remesher.to_mesh()

    fig, ax = plt.subplots()
    plot_mesh(ax, msh)
    ax.set_title("Adapted")

    # interpolation
    other = Mesh22(coords, elems, etags, faces, ftags).split().split().split().split()
    other.compute_octree()
    f_other = get_h(other).reshape((-1, 1))

    f = other.interpolate(msh, f_other)
    fig, ax = plt.subplots()
    cax = plot_field(ax, msh, f[:, 0], "vertex")
    fig.colorbar(cax, ax=ax)
    ax.set_title("f")

    plt.show()
