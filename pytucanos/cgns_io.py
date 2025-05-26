import numpy as np
from .mesh import Mesh33, Mesh32, Mesh22

try:
    import CGNS.MAP as CGM
    import CGNS.PAT.cgnslib as CGL
    import CGNS.PAT.cgnsutils as CGU
    import CGNS.PAT.cgnskeywords as CGK
    import CGNS.VAL.simplecheck as CGV

    HAVE_CGNS = True
except:
    HAVE_CGNS = False


def load_cgns(fname):
    """
    Load a Mesh22 / Mesh32 / Mesh33 from a .cgns file
    """

    if not HAVE_CGNS:
        raise RuntimeError("pycgns not available")
    flags = CGM.S2P_DEFAULT
    tree, _, _ = CGM.load(fname, flags=flags)

    edgs = np.zeros((0, 2), dtype=np.uint32)
    tris = np.zeros((0, 3), dtype=np.uint32)
    tets = np.zeros((0, 4), dtype=np.uint32)
    bdy_ids = np.zeros((0), dtype=np.uint32)

    for base in CGU.hasChildType(tree, CGK.CGNSBase_ts):
        cell_dim, phys_dim = CGU.getValue(base)
        for zone in CGU.hasChildType(base, CGK.Zone_ts):
            cg = CGU.hasChildName(zone, CGK.GridCoordinates_s)
            x = CGU.getValue(CGU.getChildByName(cg, CGK.CoordinateX_s))
            y = CGU.getValue(CGU.getChildByName(cg, CGK.CoordinateY_s))
            n = CGU.hasChildName(cg, CGK.CoordinateZ_s)
            if phys_dim == 3:
                z = CGU.getValue(n)
                coords = np.stack([x, y, z], axis=-1, dtype=np.float64)
            else:
                coords = np.stack([x, y], axis=-1, dtype=np.float64)

            for els in CGU.hasChildType(zone, CGK.Elements_ts):
                etype, _ = CGU.getValue(els)
                erange = CGU.getValue(CGU.getChildByName(els, "ElementRange"))
                econn = CGU.getValue(
                    CGU.getChildByName(els, "ElementConnectivity")
                ).astype(np.uint32)
                ids = np.arange(erange[0] - 1, erange[1], dtype=np.uint32)
                if etype == CGK.TRI_3:
                    tris = np.vstack([tris, econn.reshape((-1, 3)) - 1])
                    if phys_dim == 3:
                        bdy_ids = np.append(bdy_ids, ids)
                elif etype == CGK.TETRA_4:
                    tets = np.vstack([tets, econn.reshape((-1, 4)) - 1])
                elif etype == CGK.BAR_2:
                    edgs = np.vstack([edgs, econn.reshape((-1, 2)) - 1])
                    if phys_dim == 2:
                        bdy_ids = np.append(bdy_ids, ids)
                else:
                    raise NotImplementedError()

            bdy_tags = np.zeros(bdy_ids.size, dtype=np.int16)
            zbc = CGU.getChildByName(zone, "ZoneBC")
            tags = {}
            for i_bc, bc in enumerate(CGU.hasChildType(zbc, CGK.BC_ts)):
                if CGU.getChildByName(bc, "PointList") is not None:
                    ids = (
                        CGU.getValue(CGU.getChildByName(bc, "PointList")).squeeze() - 1
                    )
                else:
                    range = CGU.getValue(CGU.getChildByName(bc, "PointRange")).squeeze()
                    ids = np.arange(range[0] - 1, range[1])
                ids = np.searchsorted(bdy_ids, ids)
                bdy_tags[ids] = i_bc + 1
                tags[bc[0]] = i_bc + 1

            if phys_dim == 3 and tets.size > 0:
                tet_tags = np.ones(tets.shape[0], dtype=np.int16)
                return Mesh33(coords, tets, tet_tags, tris, bdy_tags), tags
            elif phys_dim == 3:
                assert edgs.size == 0
                return (
                    Mesh32(
                        coords,
                        tris,
                        bdy_tags,
                        np.zeros([0, 2], dtype=np.uint32),
                        np.zeros(0, dtype=np.int16),
                    ),
                    tags,
                )
            elif phys_dim == 2 and cell_dim == 2:
                tri_tags = np.ones(tris.shape[0], dtype=np.int16)
                return (
                    Mesh22(
                        coords,
                        tris,
                        tri_tags,
                        edgs,
                        bdy_tags,
                    ),
                    tags,
                )
            else:
                raise NotImplementedError(
                    f"Invalid dimensions: phys_dim = {phys_dim}, cell_dim = {cell_dim}"
                )


def write_cgns(
    mesh,
    fname,
    tags,
    elem_data,
    vert_data,
):
    """
    Write a Mesh22 / Mesh32 / Mesh33 to a .cgns file.
    """

    if not HAVE_CGNS:
        raise RuntimeError("pycgns not available")
    phy_dim = 2 if isinstance(mesh, Mesh22) else 3
    cell_dim = 3 if isinstance(mesh, Mesh33) else 2

    tree = CGL.newCGNSTree()
    base = CGL.newBase(tree, "Base", cell_dim, phy_dim)
    s = np.array([[mesh.n_verts(), mesh.n_elems(), 0]], dtype=np.int32)
    zone = CGL.newZone(base, "Zone", s, CGK.Unstructured_s)

    coords = mesh.get_verts()
    gc = CGL.newGridCoordinates(zone, CGK.GridCoordinates_s)
    CGL.newDataArray(gc, CGK.CoordinateX_s, coords[:, 0].copy())
    CGL.newDataArray(gc, CGK.CoordinateY_s, coords[:, 1].copy())
    if coords.shape[1] == 3:
        CGL.newDataArray(gc, CGK.CoordinateZ_s, coords[:, 2].copy())

    if isinstance(mesh, Mesh33):
        CGL.newElements(
            zone,
            "Elems",
            CGK.TETRA_4,
            np.array([1, mesh.n_elems()]),
            mesh.get_elems().astype(np.int32).ravel() + 1,
        )
        CGL.newElements(
            zone,
            "Faces",
            CGK.TRI_3,
            np.array([mesh.n_elems() + 1, mesh.n_faces()]),
            mesh.get_faces().astype(np.int32).ravel() + 1,
        )
    elif isinstance(mesh, (Mesh32, Mesh22)):
        CGL.newElements(
            zone,
            "Elems",
            CGK.TRI_3,
            np.array([1, mesh.n_elems()]),
            mesh.get_elems().astype(np.int32).ravel() + 1,
        )
        CGL.newElements(
            zone,
            "Faces",
            CGK.BAR_2,
            np.array([mesh.n_elems() + 1, mesh.n_faces()]),
            mesh.get_faces().astype(np.int32).ravel() + 1,
        )
    else:
        raise NotImplementedError(f"not implemented for {type(mesh)}")

    zbc = CGL.newZoneBC(zone)
    ftags = mesh.get_ftags()
    for name, tag in tags.items():
        (ids,) = np.nonzero(ftags == tag)
        CGL.newBC(zbc, name, ids + 1, pttype=CGK.PointList_s)

    if vert_data is not None:
        fs = CGL.newFlowSolution(zone, "FlowSolution_Vertex", CGK.Vertex_s)
        for name, arr in vert_data.items():
            CGL.newDataArray(fs, name, arr)

    if elem_data is not None:
        fs = CGL.newFlowSolution(zone, "FlowSolution_Elem", CGK.CellCenter_s)
        for name, arr in elem_data.items():
            CGL.newDataArray(fs, name, arr)

    _, msgs = CGV.compliant(tree)
    is_ok = True
    for pth, msg in msgs:
        if pth != "/CGNSLibraryVersion":
            is_ok = False
        else:
            print(pth, msg)

    if is_ok:
        CGM.save(fname, tree)
