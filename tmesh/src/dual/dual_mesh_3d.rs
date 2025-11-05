//! Computation of the dual for `Mesh<3, 4, 3>`
use super::{DualCellCenter, DualMesh, DualType, PolyMesh, PolyMeshType, circumcenter_bcoords};
use crate::{
    Tag, Vert3d,
    mesh::{
        Edge, GEdge, GSimplex, GTetrahedron, GTriangle, Idx, Mesh, Simplex, Tetrahedron, Triangle,
        sort_elem_min_ids,
    },
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::{FxBuildHasher, FxHashMap};

/// Dual of a Tetrahedron mesh in 3d
pub struct DualMesh3d<T: Idx> {
    verts: Vec<Vert3d>,
    faces: Vec<Triangle<T>>,
    ftags: Vec<Tag>,
    elem_to_face_ptr: Vec<T>,
    elem_to_face: Vec<(T, bool)>,
    etags: Vec<Tag>,
    edges: Vec<Edge<T>>,
    edge_normals: Vec<Vert3d>,
    bdy_faces: Vec<(T, Tag, Vert3d)>,
}

impl<T: Idx> DualMesh3d<T> {
    fn get_tet_center(v: &GTetrahedron<3>, t: DualType) -> DualCellCenter<T, 3, Tetrahedron<T>> {
        match t {
            DualType::Median => DualCellCenter::Vertex(v.center()),
            DualType::Barth | DualType::ThresholdBarth(_) => {
                let f = match t {
                    DualType::Barth => 0.0,
                    DualType::ThresholdBarth(l) => l,
                    DualType::Median => unreachable!(),
                };
                let f = f.max(1e-6);
                let bcoords = circumcenter_bcoords(v);
                if bcoords.iter().all(|&x| x > f) {
                    DualCellCenter::Vertex(v.vert(&bcoords))
                } else if bcoords[0] <= f {
                    DualCellCenter::Face(Triangle::<T>::from_iter(
                        [1, 2, 3].into_iter().map(|x| x.try_into().unwrap()),
                    ))
                } else if bcoords[1] <= f {
                    DualCellCenter::Face(Triangle::<T>::from_iter(
                        [2, 0, 3].into_iter().map(|x| x.try_into().unwrap()),
                    ))
                } else if bcoords[2] <= f {
                    DualCellCenter::Face(Triangle::<T>::from_iter(
                        [0, 1, 3].into_iter().map(|x| x.try_into().unwrap()),
                    ))
                } else {
                    DualCellCenter::Face(Triangle::<T>::from_iter(
                        [0, 2, 1].into_iter().map(|x| x.try_into().unwrap()),
                    ))
                }
            }
        }
    }
    fn get_tri_center(v: &GTriangle<3>, t: DualType) -> DualCellCenter<T, 3, Triangle<T>> {
        match t {
            DualType::Median => DualCellCenter::Vertex(v.center()),
            DualType::Barth | DualType::ThresholdBarth(_) => {
                let f = match t {
                    DualType::Barth => 0.0,
                    DualType::ThresholdBarth(l) => l,
                    DualType::Median => unreachable!(),
                };
                let bcoords = circumcenter_bcoords(v);
                if bcoords.iter().all(|&x| x >= f) {
                    DualCellCenter::Vertex(v.vert(&bcoords))
                } else if bcoords[0] < f {
                    DualCellCenter::Face(Edge::<T>::from_iter(
                        [1, 2].into_iter().map(|x| x.try_into().unwrap()),
                    ))
                } else if bcoords[1] < f {
                    DualCellCenter::Face(Edge::<T>::from_iter(
                        [2, 0].into_iter().map(|x| x.try_into().unwrap()),
                    ))
                } else {
                    DualCellCenter::Face(Edge::<T>::from_iter(
                        [1, 0].into_iter().map(|x| x.try_into().unwrap()),
                    ))
                }
            }
        }
    }
}

impl<T: Idx> PolyMesh<T, 3> for DualMesh3d<T> {
    fn poly_type(&self) -> PolyMeshType {
        PolyMeshType::Polyhedra
    }

    fn n_verts(&self) -> T {
        self.verts.len().try_into().unwrap()
    }

    fn vert(&self, i: T) -> Vert3d {
        self.verts[i.try_into().unwrap()]
    }

    fn n_elems(&self) -> T {
        (self.elem_to_face_ptr.len() - 1).try_into().unwrap()
    }

    fn elem(&self, i: T) -> &[(T, bool)] {
        let start = self.elem_to_face_ptr[i.try_into().unwrap()]
            .try_into()
            .unwrap();
        let end = self.elem_to_face_ptr[i.try_into().unwrap() + 1]
            .try_into()
            .unwrap();
        &self.elem_to_face[start..end]
    }

    fn etag(&self, i: T) -> Tag {
        self.etags[i.try_into().unwrap()]
    }

    fn n_faces(&self) -> T {
        self.faces.len().try_into().unwrap()
    }

    fn face(&self, i: T) -> &[T] {
        self.faces[i.try_into().unwrap()].as_ref()
    }

    fn ftag(&self, i: T) -> Tag {
        self.ftags[i.try_into().unwrap()]
    }
}

impl<T: Idx> DualMesh<T, 3, Tetrahedron<T>> for DualMesh3d<T> {
    #[allow(clippy::too_many_lines)]
    fn new<M: Mesh<T, 3, Tetrahedron<T>>>(msh: &M, t: DualType) -> Self {
        // edges
        let all_edges = msh.edges();
        let n_edges: T = all_edges.len().try_into().unwrap();

        // faces
        let all_faces = msh.all_faces();
        let n_faces = all_faces.len().try_into().unwrap();

        let n_elems = msh.n_elems();

        // vertices: boundary
        let mut bdy_verts: FxHashMap<T, T> = msh
            .faces()
            .flatten()
            .map(|i| (i, 0.try_into().unwrap()))
            .collect();
        bdy_verts
            .iter_mut()
            .enumerate()
            .for_each(|(i, (_, i_new))| *i_new = i.try_into().unwrap());
        let n_bdy_verts = bdy_verts.len().try_into().unwrap();

        let n: T = n_bdy_verts + n_edges + n_faces + n_elems;
        let mut verts = Vec::with_capacity(n.try_into().unwrap());
        for (&i_old, i_new) in &mut bdy_verts {
            *i_new = verts.len().try_into().unwrap();
            verts.push(msh.vert(i_old));
        }
        let vert_ids_bdy = |i: T| *bdy_verts.get(&i).unwrap();

        // vertices: edge centers
        verts.resize(verts.len() + n_edges.try_into().unwrap(), Vert3d::zeros());
        let vert_idx_edge = |i: T| i + n_bdy_verts;

        for (&edge, &i_edge) in &all_edges {
            let ge = GEdge::from([msh.vert(edge[0]), msh.vert(edge[1])]);
            verts[vert_idx_edge(i_edge).try_into().unwrap()] = ge.center();
        }

        // vertices: triangle centers
        let mut vert_idx_face = vec![T::MAX; n_faces.try_into().unwrap()];
        for (f, &[i_face, _, _]) in &all_faces {
            let gf = msh.gface(f);
            let center = Self::get_tri_center(&gf, t);
            match center {
                DualCellCenter::Vertex(center) => {
                    vert_idx_face[i_face.try_into().unwrap()] = verts.len().try_into().unwrap();
                    verts.push(center);
                }
                DualCellCenter::Face(e) => {
                    let edge =
                        Edge::from([f[e[0].try_into().unwrap()], f[e[1].try_into().unwrap()]])
                            .sorted();
                    let i_edge = *all_edges.get(&edge).unwrap();
                    vert_idx_face[i_face.try_into().unwrap()] = vert_idx_edge(i_edge);
                }
            }
        }

        // vertices: tet centers
        let mut vert_idx_elem = vec![T::MAX; n_elems.try_into().unwrap()];
        for (i_elem, e) in msh.elems().enumerate() {
            let ge = msh.gelem(&e);
            let center = Self::get_tet_center(&ge, t);
            match center {
                DualCellCenter::Vertex(center) => {
                    vert_idx_elem[i_elem] = verts.len().try_into().unwrap();
                    verts.push(center);
                }
                DualCellCenter::Face(f) => {
                    let face = Triangle::from([
                        e[f[0].try_into().unwrap()],
                        e[f[1].try_into().unwrap()],
                        e[f[2].try_into().unwrap()],
                    ])
                    .sorted();
                    let i_face = all_faces.get(&face).unwrap()[0];
                    vert_idx_elem[i_elem] = vert_idx_face[i_face.try_into().unwrap()];
                }
            }
        }

        // faces and polyhedra

        let n_poly_faces =
            12 * msh.n_elems().try_into().unwrap() + 6 * msh.n_faces().try_into().unwrap();
        // for Barth cells we may build the same face from different edge / face / element
        // combinations, so faces are stored sorted in a hashmap to detect duplicates
        let mut tmp_faces = FxHashMap::with_capacity_and_hasher(n_poly_faces, FxBuildHasher);

        let mut poly_to_face_ptr = vec![0; msh.n_verts().try_into().unwrap() + 1];
        // internal faces
        for e in msh.elems() {
            for edg in e.edges() {
                poly_to_face_ptr[edg[0].try_into().unwrap() + 1] += 2;
                poly_to_face_ptr[edg[1].try_into().unwrap() + 1] += 2;
            }
        }

        // boundary faces
        for f in msh.faces() {
            for edg in f.edges() {
                poly_to_face_ptr[edg[0].try_into().unwrap() + 1] += 1;
                poly_to_face_ptr[edg[1].try_into().unwrap() + 1] += 1;
            }
        }

        for i in 0..msh.n_verts().try_into().unwrap() {
            poly_to_face_ptr[i + 1] += poly_to_face_ptr[i];
        }

        let mut poly_to_face = vec![(T::MAX, true); poly_to_face_ptr[poly_to_face_ptr.len() - 1]];
        let mut edge_normals = vec![Vert3d::zeros(); n_edges.try_into().unwrap()];

        let mut n_empty_faces = 0;
        // build internal faces
        for (i_elem, e) in msh.elems().enumerate() {
            for f in e.faces() {
                let tmp = f.sorted();
                let i_face = all_faces.get(&tmp).unwrap()[0];
                for edg in f.edges() {
                    let (i_edge, sgn) = if edg[0] < edg[1] {
                        (*all_edges.get(&edg).unwrap(), 1.0)
                    } else {
                        let tmp = Edge::from([edg[1], edg[0]]);
                        (*all_edges.get(&tmp).unwrap(), -1.0)
                    };
                    let face = Triangle::from([
                        vert_idx_edge(i_edge),
                        vert_idx_elem[i_elem],
                        vert_idx_face[i_face.try_into().unwrap()],
                    ]);

                    let skip = face[0] == face[1] || face[0] == face[2] || face[1] == face[2];
                    if skip {
                        n_empty_faces += 1;
                    } else {
                        let gf = GTriangle::from([
                            verts[face[0].try_into().unwrap()],
                            verts[face[1].try_into().unwrap()],
                            verts[face[2].try_into().unwrap()],
                        ]);
                        edge_normals[i_edge.try_into().unwrap()] += sgn * gf.normal();

                        let sorted_face = face.sorted();
                        let is_sorted = face.is_same(&sorted_face);
                        let i_face = if let Some((i_face, _)) = tmp_faces.get(&sorted_face) {
                            *i_face
                        } else {
                            let i_face = tmp_faces.len();
                            tmp_faces.insert(sorted_face, (i_face, 0));
                            i_face
                        };
                        let mut ok = false;
                        let slice = &mut poly_to_face[poly_to_face_ptr[edg[0].try_into().unwrap()]
                            ..poly_to_face_ptr[edg[0].try_into().unwrap() + 1]];
                        let n = slice
                            .iter_mut()
                            .filter(|(i, _)| *i == i_face.try_into().unwrap())
                            .map(|(i, _)| *i = T::MAX)
                            .count();
                        if n == 0 {
                            for j in slice {
                                if j.0 == T::MAX {
                                    *j = (i_face.try_into().unwrap(), is_sorted);
                                    ok = true;
                                    break;
                                }
                            }
                            assert!(ok);
                        } else {
                            assert_eq!(n, 1);
                        }

                        let mut ok = false;
                        let slice = &mut poly_to_face[poly_to_face_ptr[edg[1].try_into().unwrap()]
                            ..poly_to_face_ptr[edg[1].try_into().unwrap() + 1]];
                        let n = slice
                            .iter_mut()
                            .filter(|(i, _)| *i == i_face.try_into().unwrap())
                            .map(|(i, _)| *i = T::MAX)
                            .count();
                        if n == 0 {
                            for j in slice {
                                if j.0 == T::MAX {
                                    *j = (i_face.try_into().unwrap(), !is_sorted);
                                    ok = true;
                                    break;
                                }
                            }
                            assert!(ok);
                        } else {
                            assert_eq!(n, 1);
                        }
                    }
                }
            }
        }
        // build boundary faces
        let mut bdy_faces = Vec::with_capacity(msh.n_faces().try_into().unwrap() * 6);

        for (f, tag) in msh.faces().zip(msh.ftags()) {
            let tmp = f.sorted();
            let i_face = all_faces.get(&tmp).unwrap()[0];
            for edg in f.edges() {
                let tmp = edg.sorted();
                let i_edge = *all_edges.get(&tmp).unwrap();

                let face = Triangle::from([
                    vert_ids_bdy(edg[0]),
                    vert_idx_edge(i_edge),
                    vert_idx_face[i_face.try_into().unwrap()],
                ]);
                let skip = face[0] == face[1] || face[0] == face[2] || face[1] == face[2];
                if skip {
                    n_empty_faces += 1;
                } else {
                    let gf = GTriangle::from([
                        verts[face[0].try_into().unwrap()],
                        verts[face[1].try_into().unwrap()],
                        verts[face[2].try_into().unwrap()],
                    ]);
                    bdy_faces.push((edg[0], tag, gf.normal()));

                    let sorted_face = face.sorted();
                    let is_sorted = face.is_same(&sorted_face);
                    let i_face = if let Some((i_face, _)) = tmp_faces.get(&sorted_face) {
                        *i_face
                    } else {
                        let i_face = tmp_faces.len();
                        tmp_faces.insert(sorted_face, (i_face, tag));
                        i_face
                    };

                    let mut ok = false;
                    let slice = &mut poly_to_face[poly_to_face_ptr[edg[0].try_into().unwrap()]
                        ..poly_to_face_ptr[edg[0].try_into().unwrap() + 1]];
                    let n = slice
                        .iter_mut()
                        .filter(|(i, _)| *i == i_face.try_into().unwrap())
                        .count();
                    assert_eq!(n, 0);
                    for j in slice {
                        if j.0 == T::MAX {
                            *j = (i_face.try_into().unwrap(), is_sorted);
                            ok = true;
                            break;
                        }
                    }
                    assert!(ok);
                }

                let face = Triangle::from([
                    vert_ids_bdy(edg[1]),
                    vert_idx_face[i_face.try_into().unwrap()],
                    vert_idx_edge(i_edge),
                ]);
                let skip = face[0] == face[1] || face[0] == face[2] || face[1] == face[2];
                if skip {
                    n_empty_faces += 1;
                } else {
                    let gf = GTriangle::from([
                        verts[face[0].try_into().unwrap()],
                        verts[face[1].try_into().unwrap()],
                        verts[face[2].try_into().unwrap()],
                    ]);
                    bdy_faces.push((edg[0], tag, gf.normal()));

                    let sorted_face = face.sorted();
                    let is_sorted = face.is_same(&sorted_face);
                    let i_face = if let Some((i_face, _)) = tmp_faces.get(&sorted_face) {
                        *i_face
                    } else {
                        let i_face = tmp_faces.len();
                        tmp_faces.insert(sorted_face, (i_face, tag));
                        i_face
                    };

                    let mut ok = false;
                    let slice = &mut poly_to_face[poly_to_face_ptr[edg[1].try_into().unwrap()]
                        ..poly_to_face_ptr[edg[1].try_into().unwrap() + 1]];
                    let n = slice
                        .iter_mut()
                        .filter(|(i, _)| *i == i_face.try_into().unwrap())
                        .count();
                    assert_eq!(n, 0);
                    for j in slice {
                        if j.0 == T::MAX {
                            *j = (i_face.try_into().unwrap(), is_sorted);
                            ok = true;
                            break;
                        }
                    }
                    assert!(ok);
                }
            }
        }
        assert!(tmp_faces.len() <= n_poly_faces - n_empty_faces);
        if matches!(t, DualType::Median) {
            assert_eq!(tmp_faces.len(), n_poly_faces - n_empty_faces);
        }
        let n = tmp_faces.len();
        let mut new_face_idx = vec![0.try_into().unwrap(); n];
        poly_to_face
            .iter()
            .filter(|&i| i.0 != T::MAX)
            .for_each(|&i| new_face_idx[i.0.try_into().unwrap()] += 1.try_into().unwrap());
        let mut count = 0.try_into().unwrap();
        for i in &mut new_face_idx {
            if *i == 0.try_into().unwrap() {
                *i = T::MAX;
            } else {
                assert!(*i <= 2.try_into().unwrap());
                *i = count;
                count += 1.try_into().unwrap();
            }
        }
        if matches!(t, DualType::Median) {
            assert_eq!(count, n.try_into().unwrap());
        }

        let mut faces = vec![Triangle::default(); count.try_into().unwrap()];
        let mut ftags = vec![0; count.try_into().unwrap()];
        for (face, (i_old, tag)) in tmp_faces {
            let i = new_face_idx[i_old];
            if i != T::MAX {
                faces[i.try_into().unwrap()] = face;
                ftags[i.try_into().unwrap()] = tag;
            }
        }

        // remove unused
        let n = poly_to_face.iter().filter(|&i| i.0 != T::MAX).count();

        let mut new_poly_to_face_ptr = Vec::with_capacity(poly_to_face_ptr.len());
        new_poly_to_face_ptr.push(0.try_into().unwrap());
        let mut new_poly_to_face = Vec::with_capacity(n);
        for i_elem in 0..msh.n_verts().try_into().unwrap() {
            for v in poly_to_face
                .iter()
                .take(poly_to_face_ptr[i_elem + 1])
                .skip(poly_to_face_ptr[i_elem])
            {
                if v.0 != T::MAX {
                    new_poly_to_face.push((new_face_idx[v.0.try_into().unwrap()], v.1));
                }
            }
            new_poly_to_face_ptr.push(new_poly_to_face.len().try_into().unwrap());
        }

        assert!(!new_poly_to_face.iter().any(|&i| i.0 == T::MAX));

        let mut edges = vec![Edge::<T>::default(); n_edges.try_into().unwrap()];
        for (&edg, &i_edg) in &all_edges {
            edges[i_edg.try_into().unwrap()] = edg;
        }

        let ids = sort_elem_min_ids(edges.iter().copied());
        let edges = ids
            .iter()
            .filter(|&&i| edge_normals[i.try_into().unwrap()].norm() > 1e-12)
            .map(|&i| edges[i.try_into().unwrap()])
            .collect::<Vec<_>>();
        let edge_normals = ids
            .iter()
            .filter(|&&i| edge_normals[i.try_into().unwrap()].norm() > 1e-12)
            .map(|&i| edge_normals[i.try_into().unwrap()])
            .collect::<Vec<_>>();

        Self {
            verts,
            faces,
            ftags,
            elem_to_face_ptr: new_poly_to_face_ptr,
            elem_to_face: new_poly_to_face,
            etags: vec![1; msh.n_verts().try_into().unwrap()],
            edges,
            edge_normals,
            bdy_faces,
        }
    }

    fn n_edges(&self) -> T {
        self.edges.len().try_into().unwrap()
    }

    fn edge(&self, i: T) -> Edge<T> {
        self.edges[i.try_into().unwrap()]
    }

    fn edge_normal(&self, i: T) -> Vert3d {
        self.edge_normals[i.try_into().unwrap()]
    }

    fn n_boundary_faces(&self) -> T {
        self.bdy_faces.len().try_into().unwrap()
    }
    fn par_boundary_faces(&self) -> impl IndexedParallelIterator<Item = (T, Tag, Vert3d)> + '_ {
        self.bdy_faces.par_iter().copied()
    }
    fn boundary_faces(&self) -> impl ExactSizeIterator<Item = (T, Tag, Vert3d)> + '_ {
        self.bdy_faces.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dual::{DualMesh, DualMesh3d, DualType, PolyMesh},
        mesh::{Mesh, Mesh3d, box_mesh},
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_dual_mesh_3d_simple_median() {
        let msh = box_mesh::<_, Mesh3d>(1.0, 2, 2.0, 2, 1.0, 2);
        let dual = DualMesh3d::new(&msh, DualType::Median);
        dual.check().unwrap();

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);

        dual.write_vtk("median3d.vtu").unwrap();

        // let (bdy, _): (BoundaryMesh3d, _) = dual.boundary();
        // bdy.write_vtk("median3d_bdy.vtu").unwrap();

        // let poly = SimplePolyMesh::<3>::simplify(&dual, true);
        // poly.write_vtk("median3d_simplified.vtu").unwrap();
    }

    #[test]
    fn test_dual_mesh_3d_simple_barth() {
        let msh = box_mesh::<_, Mesh3d>(1.0, 2, 2.0, 2, 1.0, 2);
        msh.write_vtk("mesh3d.vtu").unwrap();

        let dual = DualMesh3d::new(&msh, DualType::Barth);
        dual.check().unwrap();

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);

        dual.write_vtk("barth3d.vtu").unwrap();

        // let (bdy, _): (BoundaryMesh3d, _) = dual.boundary();
        // bdy.write_vtk("barth3d_bdy.vtu").unwrap();

        // let poly = SimplePolyMesh::<3>::simplify(&dual, true);
        // poly.write_vtk("barth3d_simplified.vtu").unwrap();
    }
}
