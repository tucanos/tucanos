//! Computation of the dual for `Mesh<3, 4, 3>`
use crate::{
    dual_mesh::{circumcenter_bcoords, DualCellCenter, DualMesh, DualType},
    mesh::{cell_center, cell_vertex, sort_elem_min_ids, Mesh},
    poly_mesh::{PolyMesh, PolyMeshType},
    simplices::Simplex,
    Edge, Tag, Tetrahedron, Triangle, Vert3d,
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::{FxBuildHasher, FxHashMap};

/// Dual of a Tetrahedron mesh in 3d
pub struct DualMesh3d {
    verts: Vec<Vert3d>,
    faces: Vec<Triangle>,
    ftags: Vec<Tag>,
    elem_to_face_ptr: Vec<usize>,
    elem_to_face: Vec<(usize, bool)>,
    etags: Vec<Tag>,
    edges: Vec<Edge>,
    edge_normals: Vec<Vert3d>,
    bdy_faces: Vec<(usize, Tag, Vert3d)>,
}

impl DualMesh3d {
    fn get_tet_center(v: [&Vert3d; 4], t: DualType) -> DualCellCenter<3, 3> {
        match t {
            DualType::Median => DualCellCenter::Vertex(cell_center(v)),
            DualType::Barth | DualType::ThresholdBarth(_) => {
                let f = match t {
                    DualType::Barth => 0.0,
                    DualType::ThresholdBarth(l) => l,
                    _ => unreachable!(),
                };
                let f = f.max(1e-6);
                let bcoords = circumcenter_bcoords(v);
                if bcoords.iter().all(|&x| x > f) {
                    DualCellCenter::Vertex(cell_vertex(v, bcoords))
                } else if bcoords[0] <= f {
                    DualCellCenter::Face([1, 2, 3])
                } else if bcoords[1] <= f {
                    DualCellCenter::Face([2, 0, 3])
                } else if bcoords[2] <= f {
                    DualCellCenter::Face([0, 1, 3])
                } else {
                    DualCellCenter::Face([0, 2, 1])
                }
            }
        }
    }
    fn get_tri_center(v: [&Vert3d; 3], t: DualType) -> DualCellCenter<3, 2> {
        match t {
            DualType::Median => DualCellCenter::Vertex(cell_center(v)),
            DualType::Barth | DualType::ThresholdBarth(_) => {
                let f = match t {
                    DualType::Barth => 0.0,
                    DualType::ThresholdBarth(l) => l,
                    _ => unreachable!(),
                };
                let bcoords = circumcenter_bcoords(v);
                if bcoords.iter().all(|&x| x >= f) {
                    DualCellCenter::Vertex(cell_vertex(v, bcoords))
                } else if bcoords[0] < f {
                    DualCellCenter::Face([1, 2])
                } else if bcoords[1] < f {
                    DualCellCenter::Face([2, 0])
                } else {
                    DualCellCenter::Face([1, 0])
                }
            }
        }
    }
}

impl PolyMesh<3> for DualMesh3d {
    fn poly_type(&self) -> PolyMeshType {
        PolyMeshType::Polyhedra
    }

    fn n_verts(&self) -> usize {
        self.verts.len()
    }

    fn vert(&self, i: usize) -> &Vert3d {
        &self.verts[i]
    }

    fn n_elems(&self) -> usize {
        self.elem_to_face_ptr.len() - 1
    }

    fn elem(&self, i: usize) -> &[(usize, bool)] {
        let start = self.elem_to_face_ptr[i];
        let end = self.elem_to_face_ptr[i + 1];
        &self.elem_to_face[start..end]
    }

    fn etag(&self, i: usize) -> Tag {
        self.etags[i]
    }

    fn n_faces(&self) -> usize {
        self.faces.len()
    }

    fn face(&self, i: usize) -> &[usize] {
        &self.faces[i]
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
    }
}

impl DualMesh<3, 4, 3> for DualMesh3d {
    fn new<M: Mesh<3, 4, 3>>(msh: &M, t: DualType) -> Self {
        // edges
        let all_edges = msh.compute_edges();
        let n_edges = all_edges.len();

        // faces
        let all_faces = msh.compute_faces();
        let n_faces = all_faces.len();

        let n_elems = msh.n_elems();

        // vertices: boundary
        let mut bdy_verts: FxHashMap<usize, usize> =
            msh.faces().flatten().map(|&i| (i, 0)).collect();
        bdy_verts
            .iter_mut()
            .enumerate()
            .for_each(|(i, (_, i_new))| *i_new = i);
        let n_bdy_verts = bdy_verts.len();

        let mut verts = Vec::with_capacity(n_bdy_verts + n_edges + n_faces + n_elems);
        for (&i_old, i_new) in bdy_verts.iter_mut() {
            *i_new = verts.len();
            verts.push(*msh.vert(i_old));
        }
        let vert_ids_bdy = |i: usize| *bdy_verts.get(&i).unwrap();

        // vertices: edge centers
        verts.resize(verts.len() + n_edges, Vert3d::zeros());
        let vert_idx_edge = |i: usize| i + n_bdy_verts;

        for (&edge, &i_edge) in &all_edges {
            verts[vert_idx_edge(i_edge)] = cell_center([msh.vert(edge[0]), msh.vert(edge[1])]);
        }

        // vertices: triangle centers
        let mut vert_idx_face = vec![usize::MAX; n_faces];
        for (f, &[i_face, _, _]) in &all_faces {
            let gf = msh.gface(f);
            let center = Self::get_tri_center(gf, t);
            match center {
                DualCellCenter::Vertex(center) => {
                    vert_idx_face[i_face] = verts.len();
                    verts.push(center);
                }
                DualCellCenter::Face([i0, i1]) => {
                    let edge = [f[i0], f[i1]].sorted();
                    let i_edge = *all_edges.get(&edge).unwrap();
                    vert_idx_face[i_face] = vert_idx_edge(i_edge);
                }
            }
        }

        // vertices: tet centers
        let mut vert_idx_elem = vec![usize::MAX; n_elems];
        for (i_elem, e) in msh.elems().enumerate() {
            let ge = msh.gelem(e);
            let center = Self::get_tet_center(ge, t);
            match center {
                DualCellCenter::Vertex(center) => {
                    vert_idx_elem[i_elem] = verts.len();
                    verts.push(center);
                }
                DualCellCenter::Face([i0, i1, i2]) => {
                    let face = [e[i0], e[i1], e[i2]].sorted();
                    let i_face = all_faces.get(&face).unwrap()[0];
                    vert_idx_elem[i_elem] = vert_idx_face[i_face];
                }
            }
        }

        // faces and polyhedra
        let elem_to_edges = Tetrahedron::edges();
        let face_to_edges = Triangle::edges();
        let elem_to_faces = M::elem_to_faces();

        let n_poly_faces = 12 * msh.n_elems() + 6 * msh.n_faces();
        // for Barth cells we may build the same face from different edge / face / element
        // combinations, so faces are stored sorted in a hashmap to detect duplicates
        let mut tmp_faces = FxHashMap::with_capacity_and_hasher(n_poly_faces, FxBuildHasher);

        let mut poly_to_face_ptr = vec![0; msh.n_verts() + 1];
        // internal faces
        for e in msh.elems() {
            for edg in &elem_to_edges {
                poly_to_face_ptr[e[edg[0]] + 1] += 2;
                poly_to_face_ptr[e[edg[1]] + 1] += 2;
            }
        }

        // boundary faces
        for f in msh.faces() {
            for edg in &face_to_edges {
                poly_to_face_ptr[f[edg[0]] + 1] += 1;
                poly_to_face_ptr[f[edg[1]] + 1] += 1;
            }
        }

        for i in 0..msh.n_verts() {
            poly_to_face_ptr[i + 1] += poly_to_face_ptr[i];
        }

        let mut poly_to_face =
            vec![(usize::MAX, true); poly_to_face_ptr[poly_to_face_ptr.len() - 1]];
        let mut edge_normals = vec![Vert3d::zeros(); n_edges];

        let mut n_empty_faces = 0;
        // build internal faces
        for (i_elem, e) in msh.elems().enumerate() {
            for f in &elem_to_faces {
                let f = [e[f[0]], e[f[1]], e[f[2]]];
                let tmp = f.sorted();
                let i_face = all_faces.get(&tmp).unwrap()[0];
                for edg in &face_to_edges {
                    let edg = [f[edg[0]], f[edg[1]]];
                    let (i_edge, sgn) = if edg[0] < edg[1] {
                        (*all_edges.get(&edg).unwrap(), 1.0)
                    } else {
                        let tmp = [edg[1], edg[0]];
                        (*all_edges.get(&tmp).unwrap(), -1.0)
                    };
                    let face = [
                        vert_idx_edge(i_edge),
                        vert_idx_elem[i_elem],
                        vert_idx_face[i_face],
                    ];

                    let skip = face[0] == face[1] || face[0] == face[2] || face[1] == face[2];
                    if !skip {
                        let gf = [&verts[face[0]], &verts[face[1]], &verts[face[2]]];
                        edge_normals[i_edge] += sgn * Triangle::normal(gf);

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
                        let slice = &mut poly_to_face
                            [poly_to_face_ptr[edg[0]]..poly_to_face_ptr[edg[0] + 1]];
                        for j in slice {
                            if j.0 == i_face {
                                // face is present twice --> remove it
                                *j = (usize::MAX, true);
                                ok = true;
                                break;
                            } else if j.0 == usize::MAX {
                                *j = (i_face, is_sorted);
                                ok = true;
                                break;
                            }
                        }
                        assert!(ok);

                        let mut ok = false;
                        let slice = &mut poly_to_face
                            [poly_to_face_ptr[edg[1]]..poly_to_face_ptr[edg[1] + 1]];
                        for j in slice {
                            if j.0 == i_face {
                                // face is present twice --> remove it
                                *j = (usize::MAX, true);
                                ok = true;
                                break;
                            } else if j.0 == usize::MAX {
                                *j = (i_face, !is_sorted);
                                ok = true;
                                break;
                            }
                        }
                        assert!(ok);
                    } else {
                        n_empty_faces += 1;
                    }
                }
            }
        }
        // build boundary faces
        let mut bdy_faces = Vec::with_capacity(msh.n_faces() * 6);

        for (f, tag) in msh.faces().zip(msh.ftags()) {
            let tmp = f.sorted();
            let i_face = all_faces.get(&tmp).unwrap()[0];
            for edg in &face_to_edges {
                let edg = [f[edg[0]], f[edg[1]]];
                let tmp = edg.sorted();
                let i_edge = *all_edges.get(&tmp).unwrap();

                let face = [
                    vert_ids_bdy(edg[0]),
                    vert_idx_edge(i_edge),
                    vert_idx_face[i_face],
                ];
                let skip = face[0] == face[1] || face[0] == face[2] || face[1] == face[2];
                if !skip {
                    let gf = [&verts[face[0]], &verts[face[1]], &verts[face[2]]];
                    bdy_faces.push((edg[0], tag, Triangle::normal(gf)));

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
                    let slice =
                        &mut poly_to_face[poly_to_face_ptr[edg[0]]..poly_to_face_ptr[edg[0] + 1]];
                    for j in slice {
                        if j.0 == usize::MAX {
                            *j = (i_face, is_sorted);
                            ok = true;
                            break;
                        }
                    }
                    assert!(ok);
                } else {
                    n_empty_faces += 1;
                }

                let face = [
                    vert_ids_bdy(edg[1]),
                    vert_idx_face[i_face],
                    vert_idx_edge(i_edge),
                ];
                let skip = face[0] == face[1] || face[0] == face[2] || face[1] == face[2];
                if !skip {
                    let gf = [&verts[face[0]], &verts[face[1]], &verts[face[2]]];
                    bdy_faces.push((edg[0], tag, Triangle::normal(gf)));

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
                    let slice =
                        &mut poly_to_face[poly_to_face_ptr[edg[1]]..poly_to_face_ptr[edg[1] + 1]];
                    for j in slice {
                        if j.0 == usize::MAX {
                            *j = (i_face, is_sorted);
                            ok = true;
                            break;
                        }
                    }
                    assert!(ok);
                } else {
                    n_empty_faces += 1;
                }
            }
        }
        assert!(tmp_faces.len() <= n_poly_faces - n_empty_faces);
        if matches!(t, DualType::Median) {
            assert_eq!(tmp_faces.len(), n_poly_faces - n_empty_faces);
        }
        let n = tmp_faces.len();
        let mut new_face_idx = vec![0; n];
        poly_to_face
            .iter()
            .filter(|&i| i.0 != usize::MAX)
            .for_each(|&i| new_face_idx[i.0] += 1);
        let mut count = 0;
        new_face_idx.iter_mut().for_each(|i| {
            if *i != 0 {
                assert!(*i <= 2);
                *i = count;
                count += 1;
            } else {
                *i = usize::MAX;
            }
        });
        if matches!(t, DualType::Median) {
            assert_eq!(count, n);
        }

        let mut faces = vec![[0; 3]; count];
        let mut ftags = vec![0; count];
        for (face, (i_old, tag)) in tmp_faces {
            let i = new_face_idx[i_old];
            if i != usize::MAX {
                faces[i] = face;
                ftags[i] = tag;
            }
        }

        // remove unused
        let n = poly_to_face.iter().filter(|&i| i.0 != usize::MAX).count();

        let mut new_poly_to_face_ptr = Vec::with_capacity(poly_to_face_ptr.len());
        new_poly_to_face_ptr.push(0);
        let mut new_poly_to_face = Vec::with_capacity(n);
        for i_elem in 0..msh.n_verts() {
            for v in poly_to_face
                .iter()
                .take(poly_to_face_ptr[i_elem + 1])
                .skip(poly_to_face_ptr[i_elem])
            {
                if v.0 != usize::MAX {
                    new_poly_to_face.push((new_face_idx[v.0], v.1));
                }
            }
            new_poly_to_face_ptr.push(new_poly_to_face.len());
        }

        assert!(!new_poly_to_face.iter().any(|&i| i.0 == usize::MAX));

        let mut edges = vec![[0; 2]; n_edges];
        for (&edg, &i_edg) in all_edges.iter() {
            edges[i_edg] = edg;
        }

        let ids = sort_elem_min_ids(edges.iter().copied());
        let edges = ids
            .iter()
            .filter(|&&i| edge_normals[i].norm() > 1e-12)
            .map(|&i| edges[i])
            .collect::<Vec<_>>();
        let edge_normals = ids
            .iter()
            .filter(|&&i| edge_normals[i].norm() > 1e-12)
            .map(|&i| edge_normals[i])
            .collect::<Vec<_>>();
        Self {
            verts,
            faces,
            ftags,
            elem_to_face_ptr: new_poly_to_face_ptr,
            elem_to_face: new_poly_to_face,
            etags: vec![1; msh.n_verts()],
            edges,
            edge_normals,
            bdy_faces,
        }
    }

    fn n_edges(&self) -> usize {
        self.edges.len()
    }

    fn edge(&self, i: usize) -> Edge {
        self.edges[i]
    }

    fn edge_normal(&self, i: usize) -> Vert3d {
        self.edge_normals[i]
    }

    fn n_boundary_faces(&self) -> usize {
        self.bdy_faces.len()
    }
    fn par_boundary_faces(&self) -> impl IndexedParallelIterator<Item = (usize, Tag, Vert3d)> + '_ {
        self.bdy_faces.par_iter().copied()
    }
    fn boundary_faces(&self) -> impl ExactSizeIterator<Item = (usize, Tag, Vert3d)> + '_ {
        self.bdy_faces.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::{DualMesh, DualMesh3d};
    use crate::{
        dual_mesh::DualType,
        mesh::Mesh,
        mesh_3d::{box_mesh, Mesh3d},
        poly_mesh::PolyMesh,
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_dual_mesh_3d_simple_median() {
        let msh = box_mesh::<Mesh3d>(1.0, 2, 2.0, 2, 1.0, 2);
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
        let msh = box_mesh::<Mesh3d>(1.0, 2, 2.0, 2, 1.0, 2);
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
