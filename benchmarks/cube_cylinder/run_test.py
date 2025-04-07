import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from pytucanos.mesh import Mesh33, Mesh32
from pytucanos.remesh import (
    remesh,
    remesh_mmg,
    # remesh_omega_h,
    remesh_refine,
    remesh_avro,
)
from pytucanos.quality import qualities_and_lengths

from time import time


def get_metric(msh):

    x, y, z = msh.get_verts().T

    r = (x**2 + y**2) ** 0.5
    t = np.arctan2(y, x)
    h0 = 0.001
    h_z = 0.1
    h_t = 0.1
    h_r = h0 + 2 * (0.1 - h0) * abs(r - 0.5)

    m = np.array(
        [
            [np.cos(t), -np.sin(t), np.zeros(t.size)],
            [np.sin(t), np.cos(t), np.zeros(t.size)],
            [np.zeros(t.size), np.zeros(t.size), np.ones(t.size)],
        ]
    )
    n = np.array(
        [
            np.ones(t.size) / h_r**2,
            np.ones(t.size) / h_t**2,
            np.ones(t.size) / h_z**2,
        ]
    )
    M = np.einsum("ikl,kl,jkl->ijl", m, n, m)

    n = x.size
    m = np.zeros((n, 6))
    m[:, 0] = M[0, 0, :]
    m[:, 1] = M[1, 1, :]
    m[:, 2] = M[2, 2, :]
    m[:, 3] = M[0, 1, :]
    m[:, 4] = M[1, 2, :]
    m[:, 5] = M[0, 2, :]

    return m


def run_loop(remesh, name):

    pth = os.path.dirname(__file__)
    if name == "tucanos":
        msh = Mesh33.from_meshb(os.path.join(pth, "cube-cylinder.mesh"))
    else:
        msh = Mesh33.from_meshb(os.path.join(pth, "cube-cylinder.meshb"))

    t0 = time()
    for _ in range(5):
        h = get_metric(msh)
        msh = remesh(msh, h)
    t1 = time()

    return msh, t1 - t0


def run(cases):

    print("Cube - cylinder")

    pth = os.path.dirname(__file__)

    fig_q, axs_q = plt.subplots(2, 1, tight_layout=True)
    fig_p, axs_p = plt.subplots(2, 1, tight_layout=True, sharex=True)

    perf = []
    for name, fn in cases.items():
        print("Running %s" % name)
        try:
            msh, t = run_loop(fn, name)
            perf.append((name, t, msh.n_elems()))
            print("%s: %d elems, %f s" % (name, msh.n_elems(), t))

            q, l = qualities_and_lengths(msh, get_metric(msh))

            msh.write_vtk(os.path.join(pth, "cube-cylinder-%s.vtu" % name))

            axs_q[0].hist(
                q,
                bins=50,
                alpha=0.25,
                density=True,
                label="%s (min = %.2f)" % (name, q.min()),
            )
            axs_q[0].set_xlim([0.0, 1.0])
            axs_q[1].hist(
                l[l < 2.0],
                bins=50,
                alpha=0.25,
                density=True,
                label="%s (min = %.2f, max = %.2f)" % (name, l.min(), l.max()),
            )
            axs_q[1].set_xlim([0.0, 2.0])
        except subprocess.CalledProcessError as e:
            print("%s failed: %s" % (name, e.output.decode()))

    axs_q[0].set_xlabel("quality")
    axs_q[0].legend()
    axs_q[1].axvline(x=0.5**0.5, c="r")
    axs_q[1].axvline(x=2**0.5, c="r")
    axs_q[1].set_xlabel("edge lengths")
    axs_q[1].legend()

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

    pth = os.path.dirname(__file__)
    bdy = Mesh32.from_meshb(os.path.join(pth, "cube-cylinder-boundary.mesh"))

    shutil.copy("cube-cylinder.meshb", "refine.meshb")

    def remesh_refine_egads(mesh, h):
        mesh, fname = remesh_refine(
            mesh, h, "cube-cylinder.egads", fname="refine.meshb"
        )
        shutil.move(fname, "refine.meshb")
        return mesh

    cases_benchmark = {
        "tucanos": lambda mesh, h: remesh(mesh, h, bdy, step=4.0),
        "MMG": remesh_mmg,
        "refine": remesh_refine_egads,
        # "avro": lambda mesh, h: remesh_avro(mesh, h, "cube-cylinder.egads", limit=True),
        # "Omega_h": remesh_omega_h,
    }

    run(cases_benchmark)

    plt.show()
