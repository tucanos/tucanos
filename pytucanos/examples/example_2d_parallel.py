import numpy as np
import matplotlib.pyplot as plt
from pytucanos import (
    Mesh2d,
    LinearGeometry2d,
    ParallelRemesher2dIso,
    RemesherParams,
    ParallelRemesherParams,
)
from pytucanos.mesh import get_square


def get_h(msh):
    x, y = msh.get_verts().T
    hmin = 0.001
    hmax = 0.05
    return hmin + (hmax - hmin) * (
        1 - np.exp(-((x - 0.5) ** 2 + (y - 0.25) ** 2) / 0.25**2)
    )


if __name__ == "__main__":
    import logging

    FORMAT = "%(levelname)s %(name)s %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    coords, elems, etags, faces, ftags = get_square(two_tags=False)

    msh = Mesh2d(coords, elems, etags, faces, ftags).split().split().split().split()
    msh.fix()

    geom = LinearGeometry2d(msh)

    m = get_h(msh).reshape((-1, 1))

    remesher = ParallelRemesher2dIso(msh, 3)
    remesher.partitionned_mesh().write_vtk("initial.vtu")

    remesher.set_debug(True)
    (msh, m, info) = remesher.remesh(
        geom,
        m,
        params=RemesherParams.default(),
        parallel_params=ParallelRemesherParams.default(),
    )

    print(info)

    msh.write_vtk("final.vtu")

    qualities, lengths = ParallelRemesher2dIso.qualities_and_lengths(msh, m)

    fig, (ax0, ax1) = plt.subplots(2, 1)

    fig, axs = plt.subplots(2, 1, tight_layout=True)
    axs[0].hist(qualities, bins=50, alpha=0.25, density=True)
    axs[0].set_xlabel("quality")
    axs[1].hist(lengths, bins=50, alpha=0.25, density=True)
    axs[1].axvline(x=0.5**0.5, c="r")
    axs[1].axvline(x=2**0.5, c="r")
    axs[1].set_xlabel("edge lengths")
    fig.savefig("parallel_remeshing.png", dpi=600)
