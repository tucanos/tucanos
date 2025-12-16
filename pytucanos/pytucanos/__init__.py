from .pytucanos import PyMesh2d as Mesh2d  # noqa: F401
from .pytucanos import PyBoundaryMesh2d as BoundaryMesh2d  # noqa: F401
from .pytucanos import PyQuadraticBoundaryMesh2d as QuadraticBoundaryMesh2d  # noqa: F401
from .pytucanos import PyMesh3d as Mesh3d  # noqa: F401
from .pytucanos import PyBoundaryMesh3d as BoundaryMesh3d  # noqa: F401
from .pytucanos import PyQuadraticBoundaryMesh3d as QuadraticBoundaryMesh3d  # noqa: F401
from .pytucanos import PyDualType as DualType  # noqa: F401
from .pytucanos import PyDualMesh2d as DualMesh2d  # noqa: F401
from .pytucanos import PyDualMesh3d as DualMesh3d  # noqa: F401
from .pytucanos import PyPolyMeshType as PolyMeshType  # noqa: F401
from .pytucanos import PyPolyMesh2d as PolyMesh2d  # noqa: F401
from .pytucanos import PyPolyMesh3d as PolyMesh3d  # noqa: F401
from .pytucanos import PyExtrudedMesh2d as ExtrudedMesh2d  # noqa: F401
from .pytucanos import PyPartitionerType as PartitionerType  # noqa: F401
from .pytucanos import HAVE_METIS, USE_32BIT_INTS  # noqa: F401
import numpy as np

if USE_32BIT_INTS:
    Idx = np.uint32
else:
    Idx = np.uint64

from .cgns_io import load_cgns, write_cgns  # noqa: F401
from .pytucanos import get_thread_affinity, set_thread_affinity  # noqa: F401
from .pytucanos import (
    LinearGeometry2d,  # noqa: F401
    LinearGeometry3d,  # noqa: F401
    QuadraticGeometry2d,  # noqa: F401
    QuadraticGeometry3d,  # noqa: F401
)
from .pytucanos import (
    PySplitParams as SplitParams,  # noqa: F401
    PyCollapseParams as CollapseParams,  # noqa: F401
    PySwapParams as SwapParams,  # noqa: F401
    PySmoothParams as SmoothParams,  # noqa: F401
    PySmoothingMethod as SmoothingMethod,  # noqa: F401
    PyRemeshingStep as RemeshingStep,  # noqa: F401
    PyRemesherParams as RemesherParams,  # noqa: F401
    Remesher2dIso,  # noqa: F401
    Remesher2dAniso,  # noqa: F401
    Remesher3dIso,  # noqa: F401
    Remesher3dAniso,  # noqa: F401
    ParallelRemesher2dIso,  # noqa: F401
    ParallelRemesher2dAniso,  # noqa: F401
    ParallelRemesher3dIso,  # noqa: F401
    ParallelRemesher3dAniso,  # noqa: F401
    Remesher2dIsoQuadratic,  # noqa: F401
    Remesher2dAnisoQuadratic,  # noqa: F401
    Remesher3dIsoQuadratic,  # noqa: F401
    Remesher3dAnisoQuadratic,  # noqa: F401
    ParallelRemesher2dIsoQuadratic,  # noqa: F401
    ParallelRemesher2dAnisoQuadratic,  # noqa: F401
    ParallelRemesher3dIsoQuadratic,  # noqa: F401
    ParallelRemesher3dAnisoQuadratic,  # noqa: F401
    PyParallelRemesherParams as ParallelRemesherParams,  # noqa: F401
    implied_metric_3d,  # noqa: F401
    autotag_3d,  # noqa: F401
    curvature_metric_3d,  # noqa: F401
    curvature_metric_3d_quadratic,  # noqa: F401
    transfer_tags_face_3d,  # noqa: F401
    transfer_tags_elem_3d,  # noqa: F401
    autotag_2d,  # noqa: F401
    implied_metric_2d,  # noqa: F401
    curvature_metric_2d,  # noqa: F401
    curvature_metric_2d_quadratic,  # noqa: F401
    transfer_tags_face_2d,  # noqa: F401
    transfer_tags_elem_2d,  # noqa: F401
)


def autotag(msh, angle_deg):
    if isinstance(msh, BoundaryMesh2d):
        autotag_2d(msh, angle_deg)
    elif isinstance(msh, BoundaryMesh3d):
        autotag_3d(msh, angle_deg)
    else:
        raise ValueError()


def implied_metric(msh):
    if isinstance(msh, Mesh2d):
        return implied_metric_2d(msh)
    elif isinstance(msh, Mesh3d):
        return implied_metric_3d(msh)
    else:
        raise ValueError()


def curvature_metric(msh, geom, *args, **kwargs):
    if isinstance(msh, Mesh2d):
        if isinstance(geom, LinearGeometry2d):
            return curvature_metric_2d(msh, geom, *args, **kwargs)
        elif isinstance(geom, QuadraticGeometry2d):
            return curvature_metric_2d_quadratic(msh, geom, *args, **kwargs)
        else:
            raise ValueError()
    elif isinstance(msh, Mesh3d):
        if isinstance(geom, LinearGeometry3d):
            return curvature_metric_3d(msh, geom, *args, **kwargs)
        elif isinstance(geom, QuadraticGeometry3d):
            return curvature_metric_3d_quadratic(msh, geom, *args, **kwargs)
        else:
            raise ValueError()
    else:
        raise ValueError()


def transfer_tags(msh_from, msh_to):
    if isinstance(msh_from, BoundaryMesh2d):
        if isinstance(msh_to, BoundaryMesh2d):
            transfer_tags_elem_2d(msh_from, msh_to)
        elif isinstance(msh_to, Mesh2d):
            transfer_tags_face_2d(msh_from, msh_to)
        else:
            raise ValueError()
    elif isinstance(msh_from, BoundaryMesh3d):
        if isinstance(msh_to, BoundaryMesh3d):
            transfer_tags_elem_3d(msh_from, msh_to)
        elif isinstance(msh_to, Mesh3d):
            transfer_tags_face_3d(msh_from, msh_to)
        else:
            raise ValueError()
    else:
        raise ValueError()
