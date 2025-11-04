import numpy as np
from . import Mesh2d, BoundaryMesh2d, Mesh3d, BoundaryMesh3d
import logging

try:
    import CGNS.MAP as CGM
    import CGNS.PAT.cgnslib as CGL
    import CGNS.PAT.cgnsutils as CGU
    import CGNS.PAT.cgnskeywords as CGK
    import CGNS.VAL.simplecheck as CGV

    HAVE_CGNS = True
except ImportError:
    HAVE_CGNS = False


def get_empty_mesh(phys_dim, cell_dim):
    if phys_dim == 2 and cell_dim == 1:
        return BoundaryMesh2d.empty()
    elif phys_dim == 2 and cell_dim == 2:
        return Mesh2d.empty()
    elif phys_dim == 3 and cell_dim == 2:
        return BoundaryMesh3d.empty()
    elif phys_dim == 3 and cell_dim == 3:
        return Mesh3d.empty()
    else:
        raise NotImplementedError(f"Unknown dimensions {phys_dim} / {cell_dim}")


def cgns_elem_name(etype):
    for name, i in CGK.ElementType.items():
        if etype == i:
            return name


def load_cgns(fname, cls=None, xy=True):
    """
    Load a tmesh mesh from a .cgns file
    Bases / zones are merged together, with a single element tag
    """

    if not HAVE_CGNS:
        raise RuntimeError("pycgns not available")
    flags = CGM.S2P_DEFAULT
    logging.info(f"Reading {fname}")
    tree, _, _ = CGM.load(fname, flags=flags)

    res = None

    names = {}
    next_tag = 1
    tags_to_be_removed = []

    for base in CGU.hasChildType(tree, CGK.CGNSBase_ts):
        if cls is not None:
            tmp = cls.empty()
            cell_dim = tmp.get_elems().shape[1] - 1
            phys_dim = tmp.get_verts().shape[1]
        else:
            cell_dim, phys_dim = CGU.getValue(base)
        for zone in CGU.hasChildType(base, CGK.Zone_ts):
            logging.info(f"Reading {base[0]}/{zone[0]}")
            cg = CGU.hasChildName(zone, CGK.GridCoordinates_s)
            x = CGU.getValue(CGU.getChildByName(cg, CGK.CoordinateX_s))
            y = CGU.getValue(CGU.getChildByName(cg, CGK.CoordinateY_s))
            n = CGU.hasChildName(cg, CGK.CoordinateZ_s)
            if phys_dim == 3:
                z = CGU.getValue(n)
                coords = np.stack([x, y, z], axis=-1, dtype=np.float64)
            else:
                if xy:
                    coords = np.stack([x, y], axis=-1, dtype=np.float64)
                else:
                    z = CGU.getValue(n)
                    coords = np.stack([x, z], axis=-1, dtype=np.float64)

            if cls is None:
                msh = get_empty_mesh(phys_dim, cell_dim)
            else:
                msh = cls.empty()

            logging.debug(f"Read {coords.shape[0]} vertices")
            msh.add_verts(coords)

            zbc = CGU.getChildByName(zone, "ZoneBC")
            bcs = []
            for bc in CGU.hasChildType(zbc, CGK.BC_ts):
                if CGU.getChildByName(bc, "PointList") is not None:
                    ids = (
                        CGU.getValue(CGU.getChildByName(bc, "PointList")).squeeze() - 1
                    )
                else:
                    range = CGU.getValue(CGU.getChildByName(bc, "PointRange")).squeeze()
                    ids = np.arange(range[0] - 1, range[1])
                if bc[0] in names:
                    tag = names[bc[0]]
                else:
                    tag = next_tag
                    next_tag += 1
                    names[bc[0]] = tag

                bcs.append((bc[0], tag, ids))

                logging.debug(f"Read BC {bc[0]}: {ids.size} faces, tag = {tag}")

            for els in CGU.hasChildType(zone, CGK.Elements_ts):
                etype, _ = CGU.getValue(els)
                if etype not in [
                    CGK.BAR_2,
                    CGK.TRI_3,
                    CGK.QUAD_4,
                    CGK.TETRA_4,
                    CGK.PYRA_5,
                    CGK.PENTA_6,
                    CGK.HEXA_8,
                ]:
                    raise NotImplementedError(
                        f"Element type {cgns_elem_name(etype)} not implemented"
                    )
                erange = CGU.getValue(CGU.getChildByName(els, "ElementRange"))
                ids = np.arange(erange[0] - 1, erange[1], dtype=np.uint32)
                econn = CGU.getValue(CGU.getChildByName(els, "ElementConnectivity"))
                logging.info(f"Read {ids.size} {cgns_elem_name(etype)}")
                econn = econn.astype(np.uint64).reshape((ids.size, -1)) - 1
                tags = np.zeros(ids.size, dtype=np.int16)
                for name, tag, bdy_ids in bcs:
                    tmp = np.searchsorted(bdy_ids, ids)
                    tmp = np.minimum(tmp, bdy_ids.size - 1)
                    tags[ids == bdy_ids[tmp]] = tag
                if (tags == 0).all():
                    tags[:] = 1
                else:
                    assert tags.min() > 0
                logging.debug(f"tags = {np.unique(tags)}")

                if cell_dim == 3:
                    if etype == CGK.TRI_3:
                        msh.add_faces(econn, tags)
                    elif etype == CGK.QUAD_4:
                        msh.add_quadrangles(econn, tags)
                    elif etype == CGK.TETRA_4:
                        msh.add_elems(econn, tags)
                    elif etype == CGK.PYRA_5:
                        msh.add_pyramids(econn, tags)
                    elif etype == CGK.PENTA_6:
                        msh.add_prisms(econn, tags)
                    elif etype == CGK.HEXA_8:
                        msh.add_hexahedra(econn, tags)
                    else:
                        raise NotImplementedError(
                            f"Unknown element type {cgns_elem_name(etype)} / {etype}"
                        )
                if cell_dim == 2:
                    if etype == CGK.BAR_2:
                        msh.add_faces(econn, tags)
                    elif etype == CGK.TRI_3:
                        msh.add_elems(econn, tags)
                    elif etype == CGK.QUAD_4:
                        msh.add_quadrangles(econn, tags)
                if cell_dim == 1:
                    if etype == CGK.BAR_2:
                        msh.add_elems(econn, tags)
                else:
                    raise NotImplementedError()

            bdy, ifc = msh.fix()
            if cell_dim == 3:
                assert len(ifc) == 0

            for _, i in bdy.items():
                logging.debug(f"Tagging untagged faces with {i}")
                tags_to_be_removed.append(i)
            for (t0, t1), i in ifc.items():
                logging.info(f"Tagging faces between {t0} and {t1} with {i}")
            #     tags_to_be_removed.append(i)

            if res is None:
                res = msh
            else:
                n = res.n_verts() + msh.n_verts()
                res.add(msh, tol=1e-6)
                logging.info(f"Merge zone: {n - res.n_verts()} vertices merged")

    # Remove internal faces
    faces = res.get_faces()
    ftags = res.get_ftags()
    flg = np.ones(ftags.size, dtype=bool)
    for t in tags_to_be_removed:
        flg[ftags == t] = False
    faces = np.ascontiguousarray(faces[flg, :])
    ftags = np.ascontiguousarray(ftags[flg])
    res.clear_faces()
    res.add_faces(faces, np.abs(ftags))

    bdy, ifc = res.fix()
    assert len(bdy) == 0
    assert len(ifc) == 0

    return res, names


def write_cgns(
    mesh,
    fname,
    tags,
    elem_data=None,
    vert_data=None,
):
    """
    Write a mesh to a .cgns file.
    """

    if not HAVE_CGNS:
        raise RuntimeError("pycgns not available")

    if isinstance(mesh, (Mesh2d, Mesh3d)):
        coords = mesh.get_verts()
        elems = mesh.get_elems()

        phy_dim = coords.shape[1]
        cell_dim = elems.shape[1] - 1

        tree = CGL.newCGNSTree()
        base = CGL.newBase(tree, "Base", cell_dim, phy_dim)
        s = np.array([[mesh.n_verts(), mesh.n_elems(), 0]], dtype=np.int32)
        zone = CGL.newZone(base, "Zone", s, CGK.Unstructured_s)

        gc = CGL.newGridCoordinates(zone, CGK.GridCoordinates_s)
        CGL.newDataArray(gc, CGK.CoordinateX_s, coords[:, 0].copy())
        CGL.newDataArray(gc, CGK.CoordinateY_s, coords[:, 1].copy())
        if coords.shape[1] == 3:
            CGL.newDataArray(gc, CGK.CoordinateZ_s, coords[:, 2].copy())

        if cell_dim == 3:
            CGL.newElements(
                zone,
                "Elems",
                CGK.TETRA_4,
                np.array([1, mesh.n_elems()]),
                elems.astype(np.int32).ravel() + 1,
            )
            CGL.newElements(
                zone,
                "Faces",
                CGK.TRI_3,
                np.array([mesh.n_elems() + 1, mesh.n_faces()]),
                mesh.get_faces().astype(np.int32).ravel() + 1,
            )
        elif cell_dim == 2:
            CGL.newElements(
                zone,
                "Elems",
                CGK.TRI_3,
                np.array([1, mesh.n_elems()]),
                elems.astype(np.int32).ravel() + 1,
            )
            CGL.newElements(
                zone,
                "Faces",
                CGK.BAR_2,
                np.array([mesh.n_elems() + 1, mesh.n_faces()]),
                mesh.get_faces().astype(np.int32).ravel() + 1,
            )
        elif cell_dim == 1:
            CGL.newElements(
                zone,
                "Elems",
                CGK.BAR_2,
                np.array([1, mesh.n_elems()]),
                elems.astype(np.int32).ravel() + 1,
            )
        else:
            raise NotImplementedError(f"not implemented for {type(mesh)}")

        zbc = CGL.newZoneBC(zone)
        ftags = mesh.get_ftags()
        for name, tag in tags.items():
            (ids,) = np.nonzero(ftags == tag)
            # CGL.newBC(zbc, name, ids + 1, pttype=CGK.PointList_s)
            CGL.newBC(zbc, name, ids + 1, pttype=CGK.PointRange_s)

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
