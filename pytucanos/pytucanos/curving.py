"""P1 <-> P2 (isoparametric quadratic) mesh conversion.

Tucanos has no native P2 volume remesher: `Remesher*` only ever adapts
straight-sided (P1) meshes, even the `*Quadratic` variants (those take a
*quadratic geometry* as their projection target, not a quadratic mesh).
`p2_to_p1`/`p1_to_p2` bridge a curved mesh through a P1 remeshing step:

  1. `p2_to_p1`: strip a curved `QuadraticMesh2d` down to its corner-node
     (P1) skeleton before remeshing. Exact: corner coordinates are
     unchanged.
  2. `p1_to_p2`: re-curve the (adapted) P1 mesh by projecting its boundary
     mid-edge nodes onto the true geometry and propagating that correction
     to the rest of the mesh with an RBF (`ferreus_rbf`). This is the piece
     `to_quadratic()` + `geometry.project()` don't provide on their own:
     `to_quadratic()` only inserts arithmetic midpoints (no curvature), and
     `project()` only moves boundary vertices; moving the boundary alone
     would leave a curved skin over an otherwise straight-sided interior,
     folding elements at any real curvature.

Status: 2D only (`Mesh2d`/`QuadraticMesh2d`), validated on a real
production h-adaptation loop. 3D (`Mesh3d`/`QuadraticMesh3d`) is not
implemented here: `bcoords()` for `QuadraticTriangle` has an open
convergence bug (2-parameter Newton-CG, see `quadratic_triangle.rs`) that
needs a robust closed-form/guarded solver first, the same class of fix
already applied to `QuadraticEdge` (used by the 2D path here).

Requires the optional `ferreus_rbf` dependency: `pip install
pytucanos[curving]`.
"""

import numpy as np

from . import Idx
from .cgns_io import load_cgns
from .mesh import calibrate_mid_edge_columns
from .pytucanos import LinearGeometry2d, QuadraticGeometry2d
from .pytucanos import PyMesh2d as Mesh2d
from .pytucanos import PyQuadraticBoundaryMesh2d as QuadraticBoundaryMesh2d
from .pytucanos import PyQuadraticMesh2d as QuadraticMesh2d

try:
    from ferreus_rbf import RBFInterpolator
    from ferreus_rbf.interpolant_config import (
        FittingAccuracy,
        FittingAccuracyType,
        InterpolantSettings,
        RBFKernelType,
    )

    HAVE_FERREUS_RBF = True
except ImportError:
    HAVE_FERREUS_RBF = False


def p2_to_p1(qmesh: QuadraticMesh2d) -> Mesh2d:
    """
    Strip a curved `QuadraticMesh2d` (Tri6) to its P1 skeleton (Tri3).
    Corner coordinates are copied unchanged, so no geometric approximation
    is involved in this direction. The node array is compacted to only the
    referenced corner nodes (the now-unused mid-edge nodes are dropped and
    the rest renumbered contiguously).

    Args:
        qmesh: curved `QuadraticMesh2d` to strip.

    Returns:
        New `Mesh2d` with the same corner coordinates, element/boundary tags
        preserved.
    """
    verts = qmesh.get_verts()
    elems_p1 = np.ascontiguousarray(qmesh.get_elems()[:, :3])
    faces_p1 = np.ascontiguousarray(qmesh.get_faces()[:, :2])

    used = np.unique(np.concatenate([elems_p1.ravel(), faces_p1.ravel()]))
    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)

    return Mesh2d(
        verts[used],
        remap[elems_p1].astype(Idx),
        qmesh.get_etags(),
        remap[faces_p1].astype(Idx),
        qmesh.get_ftags(),
    )


def p1_to_p2(
    mesh: Mesh2d,
    geom: str | tuple,
    tag_names: dict,
    tag_name_map: dict | None = None,
    fitting_accuracy=None,
) -> QuadraticMesh2d:
    """
    Re-curve a straight-sided `Mesh2d` to a curved `QuadraticMesh2d`,
    projecting new boundary mid-edge nodes onto `geom` and RBF-propagating
    that correction to the whole mesh.

    Only the mesh boundary tags whose *name* also appears in the geometry
    get projected onto it (matched through `tag_name_map` if given); tags
    with no match (e.g. a symmetry-plane cap that carries no curvature) are
    simply left uncorrected. `fitting_accuracy` defaults to 1e-3 relative,
    much looser than `ferreus_rbf`'s own 1e-6 default: appropriate for a
    curving correction (not a value the solution otherwise depends on), and
    needed in practice for the RBF fit to converge in reasonable time at
    realistic control-point counts.

    Args:
        mesh: straight-sided `Mesh2d` to curve.
        geom: path to a CGNS geometry file to project onto, or an
            already-loaded `(boundary_mesh, tag_names)` pair as returned by
            `load_cgns` (avoids repeated file I/O, e.g. across several
            h-adaptation iterations against the same geometry).
        tag_names: `{tag_value: tag_name}` for `mesh`'s boundary tags.
        tag_name_map: optional `{mesh_tag_name: geometry_tag_name}` override
            for cases where the mesh's and the geometry's boundary tag names
            don't match exactly.
        fitting_accuracy: optional `ferreus_rbf.FittingAccuracy` override for
            the RBF fit tolerance; defaults to `FittingAccuracy(1e-3,
            FittingAccuracyType.Relative)` if None.

    Returns:
        Curved `QuadraticMesh2d`, all `distortion() > 0`. Its connectivity
        columns follow whatever order `Mesh2d.to_quadratic()` produced; call
        `calibrate_mid_edge_columns(mesh.to_quadratic())` if you need to know
        which column holds which corner pair's mid-edge node (e.g. to export
        to a mesh format with its own fixed node-slot convention).

    Raises:
        ImportError: if the optional `ferreus_rbf` dependency isn't
            installed.
        RuntimeError: if none of `tag_names` (after `tag_name_map`) match a
            geometry tag name, or if the RBF-morphed mesh has any
            folded/invalid element (`distortion() <= 0`).
        pyo3_runtime.PanicException: `geometry.project()` needs every
            boundary tag on `mesh` to be defined *somewhere* in `geom`,
            even tags you don't intend to curve (e.g. a symmetry plane): if
            `mesh` has a tag entirely absent from `geom`, this panics,
            uncatchable from Python. That's separate from which tags
            actually get curved: that part is decided by name-matching
            (`tag_names`/`tag_name_map` above), and a `geom` patch whose
            name matches no mesh tag is simply never used as a projection
            target, which is not an issue.
    """
    if not HAVE_FERREUS_RBF:
        raise ImportError(
            "p1_to_p2 requires the optional 'ferreus_rbf' dependency: pip install pytucanos[curving]"
        )

    qm_straight = mesh.to_quadratic()
    calibrate_mid_edge_columns(
        qm_straight
    )  # fail fast on an unexpected to_quadratic() convention

    if isinstance(geom, str):
        geom_bdy, geom_tag_names = load_cgns(geom)
    else:
        geom_bdy, geom_tag_names = geom
    # A 2nd-order geometry file (BAR_3 here) loads as a QuadraticBoundaryMesh2d
    # and projects onto the exact curved geometry instead of its linear faceting.
    if isinstance(geom_bdy, QuadraticBoundaryMesh2d):
        geometry = QuadraticGeometry2d(geom_bdy)
    else:
        geometry = LinearGeometry2d(geom_bdy)
    projected_all = geometry.project(mesh)

    tag_name_map = tag_name_map or {}
    geom_names = set(geom_tag_names.keys())
    geom_tags = [
        tag
        for tag, name in tag_names.items()
        if tag_name_map.get(name, name) in geom_names
    ]
    if not geom_tags:
        raise RuntimeError(
            f"none of the mesh's boundary tags {tag_names} match a tag name "
            f"in the geometry ({geom_names}); pass tag_name_map to bridge a naming mismatch"
        )

    verts_p1 = mesh.get_verts()
    faces_p1 = mesh.get_faces()
    ftags_p1 = mesh.get_ftags()
    boundary_idx = np.unique(faces_p1[np.isin(ftags_p1, geom_tags)])
    control_pts = verts_p1[boundary_idx]
    control_disp = projected_all[boundary_idx] - control_pts

    fitting_accuracy = fitting_accuracy or FittingAccuracy(
        1e-3, FittingAccuracyType.Relative
    )
    settings = InterpolantSettings(
        RBFKernelType.Cubic, fitting_accuracy=fitting_accuracy
    )
    rbfi = RBFInterpolator(control_pts, control_disp, settings)
    verts_p2 = qm_straight.get_verts()
    new_verts = verts_p2 + rbfi.evaluate(verts_p2)

    qm_deformed = QuadraticMesh2d(
        new_verts,
        qm_straight.get_elems(),
        qm_straight.get_etags(),
        qm_straight.get_faces(),
        qm_straight.get_ftags(),
    )
    distortion = qm_deformed.distortion()
    if not np.all(distortion > 0):
        raise RuntimeError(
            f"P1->P2 re-curving produced {np.sum(distortion <= 0)} folded/invalid "
            f"element(s) (min distortion = {distortion.min():.4g})"
        )
    return qm_deformed
