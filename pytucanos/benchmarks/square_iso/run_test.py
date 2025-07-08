import os
import subprocess
from time import time
import numpy as np
import matplotlib.pyplot as plt
from pytucanos.mesh import Mesh22, get_square
from pytucanos.remesh import remesh, remesh_mmg
from pytucanos.quality import qualities_and_lengths


def get_mesh():

    return Mesh22(*get_square())


def get_metric(msh):

    x, y = msh.get_verts().T

    hmin = 0.001
    hmax = 0.3
    h = hmin + (hmax - hmin) * (
        1 - np.exp(-((x - 0.5) ** 2 + (y - 0.25) ** 2) / 0.25**2)
    )

    return h.reshape((-1, 1))


def run_loop(remesh):

    msh = get_mesh().split().split()

    for _ in range(5):
        h = get_metric(msh)
        t0 = time()
        msh = remesh(msh, h)
        t1 = time()

    return msh, t1 - t0


def run():

    print("Square - iso")

    pth = os.path.dirname(__file__)

    fig_q, axs_q = plt.subplots(2, 1, tight_layout=True)
    fig_p, axs_p = plt.subplots(2, 1, tight_layout=True, sharex=True)

    names = ["tucanos", "MMG"]
    fns = [remesh, remesh_mmg]

    perf = []
    for name, fn in zip(names, fns):
        print("Running %s" % name)
        try:
            msh, t = run_loop(fn)
            perf.append((name, t, msh.n_elems()))
            print("%s: %d elems, %f s" % (name, msh.n_elems(), t))

            qualities, lengths = qualities_and_lengths(msh, get_metric(msh))

            msh.write_vtk(os.path.join(pth, "square-iso-%s.vtu" % name))

            axs_q[0].hist(
                qualities,
                bins=50,
                alpha=0.25,
                density=True,
                label="%s (min = %.2f)" % (name, qualities.min()),
            )
            axs_q[1].hist(lengths, bins=50, alpha=0.25, density=True)
        except subprocess.CalledProcessError as e:
            print("%s failed: %s" % (name, e.output))

    axs_q[0].set_xlabel("quality")
    axs_q[0].legend()
    axs_q[1].axvline(x=0.5**0.5, c="r")
    axs_q[1].axvline(x=2**0.5, c="r")
    axs_q[1].set_xlabel("edge lengths")

    axs_p[0].bar([x[0] for x in perf], [x[1] for x in perf])
    axs_p[0].set_ylabel("remeshing time (s)")

    axs_p[1].bar([x[0] for x in perf], [x[2] for x in perf])
    axs_p[1].set_ylabel("# of elements")

    fig_q.savefig(os.path.join(pth, "quality.png"), dpi=300, transparent=True)
    fig_p.savefig(os.path.join(pth, "perfo.png"), dpi=300, transparent=True)


if __name__ == "__main__":

    import logging

    FORMAT = "%(levelname)s %(name)s %(message)s"
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    run()

    plt.show()
