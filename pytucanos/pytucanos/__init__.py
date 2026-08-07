import numpy as np

from .pytucanos import (  # noqa: F401  # noqa: F401
    BUILD_RUST_CHANNEL,
    GIT_CLEAN,
    HAVE_METIS,
    SHORT_COMMIT,
    USE_32BIT_INTS,
)
from .pytucanos import PyBoundaryMesh2d as BoundaryMesh2d
from .pytucanos import PyBoundaryMesh3d as BoundaryMesh3d
from .pytucanos import PyDualMesh2d as DualMesh2d  # noqa: F401
from .pytucanos import PyDualMesh3d as DualMesh3d  # noqa: F401
from .pytucanos import PyDualType as DualType  # noqa: F401
from .pytucanos import PyExtrudedMesh2d as ExtrudedMesh2d  # noqa: F401
from .pytucanos import PyMesh2d as Mesh2d
from .pytucanos import PyMesh3d as Mesh3d
from .pytucanos import PyPartitionerType as PartitionerType  # noqa: F401
from .pytucanos import PyPolyMesh2d as PolyMesh2d  # noqa: F401
from .pytucanos import PyPolyMesh3d as PolyMesh3d  # noqa: F401
from .pytucanos import PyPolyMeshType as PolyMeshType  # noqa: F401
from .pytucanos import (
    PyQuadraticBoundaryMesh2d as QuadraticBoundaryMesh2d,  # noqa: F401
)
from .pytucanos import (
    PyQuadraticBoundaryMesh3d as QuadraticBoundaryMesh3d,  # noqa: F401
)
from .pytucanos import PyQuadraticMesh2d as QuadraticMesh2d  # noqa: F401
from .pytucanos import PyQuadraticMesh3d as QuadraticMesh3d  # noqa: F401

if USE_32BIT_INTS:
    Idx = np.uint32
else:
    Idx = np.uint64

from .cgns_io import load_cgns, write_cgns  # noqa: F401
from .pytucanos import (  # noqa: F401
    LinearGeometry2d,
    LinearGeometry3d,
    ParallelRemesher2dAniso,
    ParallelRemesher2dAnisoQuadratic,
    ParallelRemesher2dIso,
    ParallelRemesher2dIsoQuadratic,
    ParallelRemesher3dAniso,
    ParallelRemesher3dAnisoQuadratic,
    ParallelRemesher3dIso,
    ParallelRemesher3dIsoQuadratic,
    QuadraticGeometry2d,
    QuadraticGeometry2dQMesh,
    QuadraticGeometry3d,
    QuadraticGeometry3dQMesh,
    Remesher2dAniso,
    Remesher2dAnisoQuadratic,
    Remesher2dIso,
    Remesher2dIsoQuadratic,
    Remesher3dAniso,
    Remesher3dAnisoQuadratic,
    Remesher3dIso,
    Remesher3dIsoQuadratic,
    autotag_2d,
    autotag_3d,
    curvature_metric_2d,
    curvature_metric_2d_quadratic,
    curvature_metric_3d,
    curvature_metric_3d_quadratic,
    get_thread_affinity,
    implied_metric_2d,
    implied_metric_3d,
    intersect_aniso_metric_2d,
    intersect_aniso_metric_3d,
    set_thread_affinity,
    transfer_tags_elem_2d,
    transfer_tags_elem_3d,
    transfer_tags_face_2d,
    transfer_tags_face_3d,
)
from .pytucanos import (
    PyCollapseParams as CollapseParams,  # noqa: F401
)
from .pytucanos import (
    PyParallelRemesherParams as ParallelRemesherParams,  # noqa: F401
)
from .pytucanos import (
    PyRemesherParams as RemesherParams,  # noqa: F401
)
from .pytucanos import (
    PyRemeshingStep as RemeshingStep,  # noqa: F401
)
from .pytucanos import (
    PySmoothingMethod as SmoothingMethod,  # noqa: F401
)
from .pytucanos import (
    PySmoothParams as SmoothParams,  # noqa: F401
)
from .pytucanos import (
    PySplitParams as SplitParams,  # noqa: F401
)
from .pytucanos import (
    PySwapParams as SwapParams,  # noqa: F401
)


def autotag(msh, angle_deg):
    if isinstance(msh, BoundaryMesh2d):
        autotag_2d(msh, angle_deg)
    elif isinstance(msh, BoundaryMesh3d):
        autotag_3d(msh, angle_deg)
    else:
        raise TypeError(f"Unsupported mesh type: {type(msh)!r}")


def implied_metric(msh):
    if isinstance(msh, Mesh2d):
        return implied_metric_2d(msh)
    elif isinstance(msh, Mesh3d):
        return implied_metric_3d(msh)
    else:
        raise TypeError(f"Unsupported mesh type: {type(msh)!r}")


def curvature_metric(msh, geom, *args, **kwargs):
    if isinstance(msh, Mesh2d):
        if isinstance(geom, LinearGeometry2d):
            return curvature_metric_2d(msh, geom, *args, **kwargs)
        elif isinstance(geom, QuadraticGeometry2d):
            return curvature_metric_2d_quadratic(msh, geom, *args, **kwargs)
        else:
            raise TypeError(f"Unsupported geometry type for Mesh2d: {type(geom)!r}")
    elif isinstance(msh, Mesh3d):
        if isinstance(geom, LinearGeometry3d):
            return curvature_metric_3d(msh, geom, *args, **kwargs)
        elif isinstance(geom, QuadraticGeometry3d):
            return curvature_metric_3d_quadratic(msh, geom, *args, **kwargs)
        else:
            raise TypeError(f"Unsupported geometry type for Mesh3d: {type(geom)!r}")
    else:
        raise TypeError(f"Unsupported mesh type: {type(msh)!r}")


def transfer_tags(msh_from, msh_to):
    if isinstance(msh_from, BoundaryMesh2d):
        if isinstance(msh_to, BoundaryMesh2d):
            transfer_tags_elem_2d(msh_from, msh_to)
        elif isinstance(msh_to, Mesh2d):
            transfer_tags_face_2d(msh_from, msh_to)
        else:
            raise TypeError(
                f"Unsupported destination mesh type for BoundaryMesh2d: {type(msh_to)!r}"
            )
    elif isinstance(msh_from, BoundaryMesh3d):
        if isinstance(msh_to, BoundaryMesh3d):
            transfer_tags_elem_3d(msh_from, msh_to)
        elif isinstance(msh_to, Mesh3d):
            transfer_tags_face_3d(msh_from, msh_to)
        else:
            raise TypeError(
                f"Unsupported destination mesh type for BoundaryMesh3d: {type(msh_to)!r}"
            )
    else:
        raise TypeError(f"Unsupported source mesh type: {type(msh_from)!r}")
