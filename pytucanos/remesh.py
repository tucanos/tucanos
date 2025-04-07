import os
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import logging
from ._pytucanos import (
    PySplitParams,
    PyCollapseParams,
    PySwapParams,
    PySmoothParams,
    PySmoothingMethod,
    PyRemeshingStep,
    PyRemesherParams,
)
from ._pytucanos import Remesher2dIso, Remesher2dAniso, Remesher3dIso, Remesher3dAniso
from ._pytucanos import (
    ParallelRemesher2dIso,
    ParallelRemesher2dAniso,
    ParallelRemesher3dIso,
    ParallelRemesher3dAniso,
    PyParallelRemesherParams,
)

from .mesh import Mesh22, Mesh33
from .geometry import LinearGeometry2d, LinearGeometry3d


def print_params(params):

    for i, step in enumerate(params.steps):
        step = step._0
        if isinstance(step, PyCollapseParams):
            print(f"{i} - Collapse")
            attrs = [
                "l",
                "max_iter",
                "max_l_rel",
                "max_l_abs",
                "min_q_rel",
                "min_q_abs",
                "max_angle",
            ]
            for attr in attrs:
                print(f"  {attr} = {getattr(step, attr)}")
        elif isinstance(step, PySplitParams):
            print(f"{i} - Split")
            attrs = [
                "l",
                "max_iter",
                "min_l_rel",
                "min_l_abs",
                "min_q_rel",
                "min_q_rel_bdy",
                "min_q_abs",
            ]
            for attr in attrs:
                print(f"  {attr} = {getattr(step, attr)}")
        elif isinstance(step, PySwapParams):
            print(f"{i} - Swap")
            attrs = [
                "q",
                "max_iter",
                "max_l_rel",
                "max_l_abs",
                "min_l_rel",
                "min_l_abs",
                "max_angle",
            ]
            for attr in attrs:
                print(f"  {attr} = {getattr(step, attr)}")
        elif isinstance(step, PySmoothParams):
            print(f"{i} - Smooth")
            attrs = [
                "n_iter",
                "method",
                "relax",
                "keep_local_minima",
                "max_angle",
            ]
            for attr in attrs:
                print(f"  {attr} = {getattr(step, attr)}")
        else:
            raise RuntimeError()


def update_params(params, step, key, value):

    res = []
    for s in params.steps:
        if isinstance(s, step):
            tmp = s._0
            setattr(tmp, key, value)
            res.append(step(tmp))
        else:
            res.append(s)

    return PyRemesherParams(res, params.debug)


def plot_stats(remesher):
    """
    Plot the remesher stats
    """

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


def __read_tmp_meshb(dim, fname="tmp.meshb"):

    if dim == 2:
        msh = Mesh22.from_meshb(fname)
    elif dim == 3:
        msh = Mesh33.from_meshb(fname)

    os.remove(fname)

    return msh


def __iso_to_aniso_3d(h):

    if h.shape[1] == 1:
        m = np.zeros((h.shape[0], 6), dtype=np.float64)
        for i in range(3):
            m[:, i] = 1.0 / h[:, 0] ** 2
        return m
    return h


def remesh(msh, h, bdy=None, step=None, params=None):
    """
    Remesh using tucanos
    """

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
        h = Remesher.control_step_metric(msh, h, m_implied, step)

    remesher = Remesher(msh, geom, h)
    if params is None:
        params = PyRemesherParams.default()
    remesher.remesh(geom, params)

    return remesher.to_mesh()


def use_podman():

    if "USE_PODMAN" in os.environ:
        return [
            "podman",
            "run",
            "-v",
            "%s:/data:z" % os.getcwd(),
            "-w",
            "/data",
            "-it",
            "remeshers",
        ]
    else:
        return []


def remesh_mmg(msh, h, hgrad=10.0, hausd=10.0):
    """
    Remesh using MMG.
    The path to the mmg executable is given by environment variable MMG2D_EXE/MMG3D_EXE
    """
    __write_tmp_meshb(msh, h)

    if isinstance(msh, Mesh22):
        dim = 2
        exe = os.getenv("MMG2D_EXE", "mmg2d_O3")
    else:
        dim = 3
        exe = os.getenv("MMG3D_EXE", "mmg3d_O3")

    cmd = use_podman() + [
        exe,
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
    ]
    logging.info(f"Running {' '.join(cmd)}")
    subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
    )

    os.remove("tmp.solb")

    return __read_tmp_meshb(dim)


def remesh_omega_h(msh, h):
    """
    Remesh using Omega_h.
    The path to the osh_adapt executable is given by environment variable OSH_EXE
    """

    h = __iso_to_aniso_3d(h)

    __write_tmp_meshb(msh, h)

    exe = os.getenv("OSH_EXE", "osh_adapt")
    cmd = use_podman() + [
        exe,
        "--mesh-in",
        "tmp.meshb",
        "--metric-in",
        "tmp.solb",
        "--mesh-out",
        "tmp.meshb",
        "--metric-out",
        "tmp.solb",
    ]
    logging.info(f"Running {' '.join(cmd)}")
    subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
    )

    os.remove("tmp.solb")

    return __read_tmp_meshb(3)


def remesh_refine(msh, h, geom=None):
    """
    Remesh using refine.
    The path to the ref executable is given by environment variable REF_EXE
    """

    h = __iso_to_aniso_3d(h)

    __write_tmp_meshb(msh, h)

    exe = os.getenv("REF_EXE", "ref")

    cmd = use_podman() + [
        exe,
        "adapt",
        fname,
        "--metric",
        "tmp.solb",
        "-x",
        "tmp.meshb",
    ]
    if geom is not None:
        cmd += ["--egads", geom]

    logging.info(f"Running {' '.join(cmd)}")
    subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
    )

    os.remove("tmp.solb")

    return __read_tmp_meshb(3)


def remesh_avro(msh, h, geom, limit=False):
    """
    Remesh using avro.
    The path to the avro executable is given by environment variable AVRO_EXE
    """

    h = __iso_to_aniso_3d(h)

    __write_tmp_meshb(msh, h)

    exe = os.getenv("AVRO_EXE", "avro")
    cmd = use_podman() + [
        exe,
        "-adapt",
        fname,
        geom,
        "tmp.solb",
        "tmp.mesh",
        "limit=%s" % ("true" if limit else "false"),
    ]
    logging.info(f"Running {' '.join(cmd)}")
    subprocess.check_output(
        cmd,
        stderr=subprocess.STDOUT,
    )
    os.remove("tmp.solb")
    os.remove("tmp_0.sol")

    return __read_tmp_meshb(3, "tmp_0.mesh")
