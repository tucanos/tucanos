import os
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from ._pytucanos import Remesher2dIso, Remesher2dAniso, Remesher3dIso, Remesher3dAniso
from .mesh import Mesh22, Mesh33
from .geometry import LinearGeometry2d, LinearGeometry3d


def plot_stats(remesher):

    fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)

    stats = json.loads(remesher.stats_json())
    colors = {
        "Collapse": "C1",
        "Split": "C2",
        "Swap": "C3",
        "Smooth": "C4",
    }
    for idx, step in enumerate(stats):
        for name, data in step.items():
            data = data["r_stats"]
            axs[0].scatter(idx, data["n_elems"], color="r")

            stats_l = data["stats_l"]
            y = np.array(stats_l["bins"])
            x = np.array(stats_l["vals"])
            axs[1].barh(
                0.5 * (y[1:] + y[:-1]),
                x,
                (y[1:] - y[:-1]),
                left=idx,
                color="k",
            )
            axs[1].scatter(idx, stats_l["mean"], color="r")

            stats_q = data["stats_q"]
            y = np.array(stats_q["bins"])
            x = np.array(stats_q["vals"])
            axs[2].barh(
                0.5 * (y[1:] + y[:-1]),
                x,
                (y[1:] - y[:-1]),
                left=idx,
                color="k",
            )
            axs[2].scatter(idx, stats_q["mean"], color="r")

            if name != "Init":
                for i in range(3):
                    axs[i].axvspan(idx - 1, idx, alpha=0.25, color=colors[name])

    axs[0].set_ylabel("# of elements")
    axs[1].set_ylabel("lengths")
    axs[2].set_ylabel("qualities")

    return fig, axs


def __write_tmp_meshb(msh, h):

    if isinstance(msh, Mesh22):
        msh.write_meshb("tmp.meshb")
        msh.write_solb("tmp.solb", h)
    elif isinstance(msh, Mesh33):
        msh.write_meshb("tmp.meshb")
        msh.write_solb("tmp.solb", h)
    else:
        raise NotImplementedError()


def __read_tmp_meshb(dim):

    if dim == 2:
        msh = Mesh22.from_meshb("tmp.meshb")
    elif dim == 3:
        msh = Mesh33.from_meshb("tmp.meshb")

    os.remove("tmp.meshb")

    return msh


def __iso_to_aniso_3d(h):

    if h.shape[1] == 1:
        m = np.zeros((h.shape[0], 6), dtype=np.float64)
        for i in range(3):
            m[:, i] = 1.0 / h[:, 0] ** 2
        return m
    return h


def remesh(msh, h, bdy=None, step=None, **remesh_params):

    if isinstance(msh, Mesh33):
        LinearGeometry = LinearGeometry3d
        Remesher = Remesher3dIso if h.shape[1] == 1 else Remesher3dAniso
    elif isinstance(msh, Mesh22):
        LinearGeometry = LinearGeometry2d
        Remesher = Remesher2dIso if h.shape[1] == 1 else Remesher2dAniso
    else:
        raise NotImplementedError

    msh.compute_topology()
    geom = LinearGeometry(msh, bdy)

    if step is not None:
        # limit the metric sizes to 1/step -> 4 times the those given by the element implied
        # metric
        msh.compute_vertex_to_elems()
        msh.compute_volumes()
        m_implied = msh.implied_metric()
        h = Remesher.limit_metric(msh, h, m_implied, step)

    remesher = Remesher(msh, geom, h)
    remesher.remesh(
        two_steps=True,
        num_iter=2,
        split_min_q_rel=0.5,
        split_min_q_abs=0.1,
        collapse_min_q_rel=0.5,
        collapse_min_q_abs=0.1,
        swap_min_l_abs=0.25,
        swap_max_l_abs=4.0,
        smooth_iter=4,
        smooth_type="laplacian",
    )

    return remesher.to_mesh()


def remesh_mmg(msh, h, hgrad=10.0, hausd=10.0):

    __write_tmp_meshb(msh, h)

    if isinstance(msh, Mesh22):
        dim = 2
        subprocess.check_output(
            [
                "mmg2d_O3",
                "-in",
                "tmp.meshb",
                "-sol",
                "tmp.solb",
                "-out",
                "tmp.meshb",
                "-hgrad",
                repr(hgrad),
                "-hausd",
                repr(hausd),
            ],
            stderr=subprocess.STDOUT,
        )
    else:
        dim = 3
        subprocess.check_output(
            [
                "mmg3d_O3",
                "-in",
                "tmp.meshb",
                "-sol",
                "tmp.solb",
                "-out",
                "tmp.meshb",
                "-hgrad",
                repr(hgrad),
                "-hausd",
                repr(hausd),
            ],
            stderr=subprocess.STDOUT,
        )

    return __read_tmp_meshb(dim)


def remesh_omega_h(msh, h):

    h = __iso_to_aniso_3d(h)

    __write_tmp_meshb(msh, h)

    subprocess.check_output(
        [
            "osh_adapt",
            "--mesh-in",
            "tmp.meshb",
            "--metric-in",
            "tmp.solb",
            "--mesh-out",
            "tmp.meshb",
            "--metric-out",
            "tmp.solb",
        ],
        stderr=subprocess.STDOUT,
    )

    os.remove("tmp.solb")

    return __read_tmp_meshb(3)


def remesh_refine(msh, h):

    h = __iso_to_aniso_3d(h)

    __write_tmp_meshb(msh, h)

    subprocess.check_output(
        [
            "ref",
            "adapt",
            "tmp.meshb",
            "--metric",
            "tmp.solb",
            "-x",
            "tmp.meshb",
        ],
        stderr=subprocess.STDOUT,
    )

    return __read_tmp_meshb(3)
