import logging
import numpy as np
import matplotlib.pyplot as plt
from pytucanos import (
    Mesh2d,
    LinearGeometry2d,
    Remesher2dAniso,
    RemesherParams,
)
from pytucanos.mesh import (
    get_square,
    plot_mesh,
)


def get_f(msh):
    x, y = msh.get_verts().T
    f = np.tanh((x + y - 1.0) / 0.01)
    return f.reshape((-1, 1))


if __name__ == "__main__":
    FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    coords, elems, etags, faces, ftags = get_square()
    msh = Mesh2d(coords, elems, etags, faces, ftags)
    msh = msh.split().split().split().split()

    msh.fix()

    geom = LinearGeometry2d(msh)
    for _ in range(6):
        f = get_f(msh)
        hessian = msh.hessian(f)

        m = Remesher2dAniso.hessian_to_metric(msh, hessian)
        m = Remesher2dAniso.smooth_metric(msh, m, n_iter=1)
        for _ in range(2):
            m = Remesher2dAniso.scale_metric(
                msh, m, h_min=0.0001, h_max=0.3, n_elems=1000
            )
            m = Remesher2dAniso.apply_metric_gradation(
                msh, m, beta=1.5, t=1.0, n_iter=3
            )

        assert np.isfinite(m).all()

        remesher = Remesher2dAniso(msh, geom, m)
        remesher.remesh(geom, params=RemesherParams.default())

        msh = remesher.to_mesh()

    qualities = remesher.qualities()
    lengths = remesher.lengths()
    fig, axs = plt.subplots(2, 1, tight_layout=True)
    axs[0].hist(qualities, bins=50, alpha=0.25, density=True)
    axs[0].set_xlabel("quality")
    axs[1].hist(lengths, bins=50, alpha=0.25, density=True)
    axs[1].axvline(x=0.5**0.5, c="r")
    axs[1].axvline(x=2**0.5, c="r")
    axs[1].set_xlabel("edge lengths")

    fig, ax = plt.subplots(tight_layout=True)
    plot_mesh(ax, msh)
    ax.set_title("Adapted (%d elements)" % msh.n_elems())

    plt.show()
