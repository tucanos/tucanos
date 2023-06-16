import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import gmsh
from pytucanos.mesh import Mesh22, Mesh21, plot_mesh, plot_metric, plot_field
from pytucanos.geometry import LinearGeometry2d
from pytucanos.remesh import Remesher2dAniso


def naca_profile(
    id="0012",
    chord=1.0,
    n=512,
    finite_te=False,
):

    m = int(id[0]) / 100
    p = int(id[1]) / 10
    t = int(id[2:4]) / 100

    beta = np.linspace(0, np.pi, n)
    x_c = 0.5 * (1 - np.cos(beta))
    x = x_c * chord

    # thickness
    if finite_te:
        thickness = (
            0.2969 * x_c**0.5
            - 0.1260 * x_c
            - 0.3516 * x_c**2
            + 0.2843 * x_c**3
            - 0.1015 * x_c**4
        )
    else:
        thickness = (
            0.2969 * x_c**0.5
            - 0.1260 * x_c
            - 0.3516 * x_c**2
            + 0.2843 * x_c**3
            - 0.1036 * x_c**4
        )
    thickness *= t * chord / 0.2

    # camber
    if p > 0:
        camber = (m * x / p**2) * (2 * p - x_c) * (x < p * chord)
        camber += (
            (m * (chord - x) / (1 - p) ** 2) * (1 + x_c - 2 * p) * (x >= p * chord)
        )

        theta = np.arctan((m / p**2) * (2 * p - 2 * x_c)) * (x < p * chord)
        theta += np.arctan((m / (1 - p) ** 2) * (-2 * x_c + 2 * p)) * (x >= p * chord)
    else:
        camber = np.zeros(n)
        theta = np.zeros(n)

    # upper surface
    upper = np.stack(
        [x - thickness * np.sin(theta), camber + thickness * np.cos(theta)], axis=1
    )

    # lower surface
    lower = np.stack(
        [x + thickness * np.sin(theta), camber - thickness * np.cos(theta)], axis=1
    )

    return upper, lower


def get_meshes():

    upper, lower = naca_profile()

    coords = np.vstack(
        [
            upper,
            lower,
            np.array(
                [
                    [-10.0, -10.0],
                    [10.0, -10.0],
                    [10.0, 10.0],
                    [-10.0, 10.0],
                ]
            ),
        ]
    )
    n = upper.shape[0]
    elems = np.vstack(
        [
            np.stack([np.arange(n - 1), np.arange(n - 1) + 1], axis=1),
            n + np.stack([np.arange(n - 1), np.arange(n - 1) + 1], axis=1),
            2 * n
            + np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [3, 0],
                ]
            ),
        ]
    )
    etags = np.concatenate([1 + np.zeros(n - 1), 2 + np.zeros(n - 1), [3, 4, 5, 6]])

    bmsh = Mesh21(
        coords,
        elems.astype(np.uint32),
        etags.astype(np.int16),
        np.empty((0, 1), dtype=np.uint32),
        np.empty(0, dtype=np.int16),
    )

    gmsh.initialize(sys.argv)
    gmsh.model.add("NACA 0012")

    curvs = []
    upper = gmsh.model.occ.addSpline(
        [gmsh.model.occ.addPoint(x, y, 0, 0.1) for x, y in upper]
    )
    lower = gmsh.model.occ.addSpline(
        [gmsh.model.occ.addPoint(x, y, 0, 0.1) for x, y in lower]
    )
    cl1 = gmsh.model.occ.addCurveLoop([upper, lower])
    # print(cl1, upper, lower)

    p1 = gmsh.model.occ.addPoint(-10, -10, 0, 1.0)
    p2 = gmsh.model.occ.addPoint(10, -10, 0, 1.0)
    p3 = gmsh.model.occ.addPoint(10, 10, 0, 1.0)
    p4 = gmsh.model.occ.addPoint(-10, 10, 0, 1.0)
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)
    cl2 = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

    s = gmsh.model.occ.addPlaneSurface([cl2, cl1])

    gmsh.model.occ.synchronize()

    tags = [upper, lower, l1, l2, l3, l4]
    phy_tags = []
    for tag in tags:
        dim = 1
        mass_ref = gmsh.model.occ.getMass(dim, abs(tag))
        center_ref = gmsh.model.occ.getCenterOfMass(dim, abs(tag))

        found = False
        for dim1, tag1 in gmsh.model.getBoundary([(2, s)]):
            mass = gmsh.model.occ.getMass(dim1, abs(tag1))
            center = gmsh.model.occ.getCenterOfMass(dim1, abs(tag1))
            if np.allclose(mass_ref, mass) and np.allclose(center_ref, center):
                phy_tags.append(gmsh.model.addPhysicalGroup(1, [tag1]))
                found = True
                break
        assert found

    phy_tags.append(gmsh.model.addPhysicalGroup(2, [s]))

    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 5)
    gmsh.model.mesh.generate(2)

    elems = []
    etags = []
    faces = []
    ftags = []
    for dim, tag in gmsh.model.getPhysicalGroups():
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            types, tags, conn = gmsh.model.mesh.getElements(dim, tag=ent)
            assert len(types) == 1
            if types[0] == 1:
                faces.append(conn[0].reshape((-1, 2)).astype(np.uint32))
                ftags.append(tag + np.zeros(tags[0].size, dtype=np.int16))
            elif types[0] == 2:
                elems.append(conn[0].reshape((-1, 3)).astype(np.uint32))
                etags.append(tag + np.zeros(tags[0].size, dtype=np.int16))
            else:
                raise NotImplementedError()

    elems = np.vstack(elems) - 1
    etags = np.concatenate(etags)
    faces = np.vstack(faces) - 1
    ftags = np.concatenate(ftags)

    indices, points, _ = gmsh.model.mesh.getNodes()
    indices -= 1
    perm_sort = np.argsort(indices)
    assert np.all(indices[perm_sort] == np.arange(len(indices)))
    coords = points.reshape((-1, 3))[perm_sort, :2].copy()

    new_ids = np.zeros(coords.shape[0], dtype=np.int32) - 1
    old_ids = np.unique(elems.ravel())
    new_ids[old_ids] = np.arange(old_ids.size)

    elems = new_ids[elems]
    faces = new_ids[faces]

    assert elems.min() == 0
    assert faces.min() >= 0

    mesh = Mesh22(
        coords[old_ids, :],
        elems.astype(np.uint32),
        etags,
        faces.astype(np.uint32),
        ftags,
    )
    # gmsh.fltk.run()
    gmsh.finalize()

    return mesh, bmsh


if __name__ == "__main__":

    FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    msh, bmsh = get_meshes()

    fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True)
    plot_mesh(ax0, msh, normals=True)
    ax0.set_title("Initial mesh")
    plot_mesh(ax1, bmsh)
    ax1.set_title("Geometry")

    ax1.set_xlim([-0.5, 1.5])
    ax1.set_ylim([-0.5, 0.5])

    msh.write_vtk(f"naca_0.vtu")

    for it in range(6):
        msh.compute_topology()
        geom = LinearGeometry2d(msh, bmsh)
        geom.compute_curvature()

        bdy, bdy_ids = msh.boundary()
        h_n = 1e-3 + np.zeros(bdy_ids.size)

        msh.compute_vertex_to_vertices()
        m_curv = msh.curvature_metric(
            geom, 4.0, 1.5, 1e-5, h_n, np.array([1, 2], dtype=np.int16)
        )

        msh.compute_vertex_to_elems()
        msh.compute_volumes()
        m_implied = msh.implied_metric()

        # fig, (ax0, ax1) = plt.subplots(
        #     1, 2, sharex=True, sharey=True, tight_layout=True
        # )
        # plot_mesh(ax0, msh, False, False, False)
        # plot_metric(ax0, msh, m_curv)
        # ax0.set_title("Curvature metric")
        # plot_mesh(ax1, msh, False, False, False)
        # plot_metric(ax1, msh, m_implied)
        # ax1.set_title("Implied metric")

        # ax1.set_xlim([-0.5, 1.5])
        # ax1.set_ylim([-0.5, 0.5])

        h = 1.0
        m = np.zeros((msh.n_verts(), 3))
        m[:, :2] = h

        m = Remesher2dAniso.scale_metric(
            msh,
            m,
            1e-5,
            1.0,
            1,
            fixed_m=m_curv,
            implied_m=m_implied,
            step=2.0,
            max_iter=0,
        )

        m = Remesher2dAniso.apply_metric_gradation(msh, m, 1.5, n_iter=10)

        # fig, (ax0, ax1) = plt.subplots(
        #     1, 2, sharex=True, sharey=True, tight_layout=True
        # )
        # plot_mesh(ax0, msh, False, False, False)
        # plot_metric(ax0, msh, m)
        # ax0.set_title("Metric")

        remesher = Remesher2dAniso(msh, geom, m)
        remesher.remesh(two_steps=True, num_iter=3)
        q = remesher.qualities()
        l = remesher.lengths()

        msh = remesher.to_mesh()

        # cax = plot_field(ax1, msh, q, "element")
        # fig.colorbar(cax, ax=ax1)
        # ax1.set_title("Adapted mesh")

        # ax1.set_xlim([-0.5, 1.5])
        # ax1.set_ylim([-0.5, 0.5])

        fig, ax = plt.subplots(2, 1, tight_layout=True)
        ax[0].hist(
            q,
            bins=50,
            alpha=0.25,
            density=True,
            label="parmesan (q_min = %.2f)" % q.min(),
        )
        ax[0].set_xlabel("quality")
        ax[0].legend()
        ax[1].hist(
            l,
            bins=50,
            alpha=0.25,
            density=True,
            label="parmesan (l_min = %.2f, l_max = %.2f)" % (l.min(), l.max()),
        )
        ax[1].axvline(x=0.5**0.5, c="r")
        ax[1].axvline(x=2**0.5, c="r")
        ax[1].set_xlabel("edge lengths")
        ax[1].legend()

        msh.write_vtk(f"naca_{it+1}.vtu", {}, {"q": q.reshape((-1, 1))})

    plt.show()
