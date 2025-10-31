//! Computation of the dual for `Mesh<2, 3, 2>`
use super::{DualCellCenter, DualMesh, DualType, PolyMesh, PolyMeshType, circumcenter_bcoords};
use crate::{
    Tag, Vert2d,
    mesh::{Edge, GEdge, GSimplex, GTriangle, Mesh, Simplex, Triangle, sort_elem_min_ids},
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;

/// Dual of a Triangle mesh in 2d
pub struct DualMesh2d {
    verts: Vec<Vert2d>,
    faces: Vec<Edge>,
    ftags: Vec<Tag>,
    elem_to_face_ptr: Vec<usize>,
    elem_to_face: Vec<(usize, bool)>,
    etags: Vec<Tag>,
    edges: Vec<Edge>,
    edge_normals: Vec<Vert2d>,
    bdy_faces: Vec<(usize, Tag, Vert2d)>,
}

impl DualMesh2d {
    fn get_tri_center(v: &GTriangle<2>, t: DualType) -> DualCellCenter<2, Triangle> {
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
                    DualCellCenter::Face([1, 2].into())
                } else if bcoords[1] <= f {
                    DualCellCenter::Face([2, 0].into())
                } else {
                    DualCellCenter::Face([1, 0].into())
                }
            }
        }
    }
}

impl PolyMesh<2> for DualMesh2d {
    fn poly_type(&self) -> PolyMeshType {
        PolyMeshType::Polygons
    }

    fn n_verts(&self) -> usize {
        self.verts.len()
    }

    fn vert(&self, i: usize) -> Vert2d {
        self.verts[i]
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
        self.faces[i].as_slice()
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
    }
}

impl DualMesh<2, Triangle> for DualMesh2d {
    #[allow(clippy::too_many_lines)]
    fn new<M: Mesh<2, Triangle>>(msh: &M, t: DualType) -> Self {
        // edges
        let all_edges = msh.edges();
        let n_edges = all_edges.len();

        let n_elems = msh.n_elems();

        // vertices: boundary
        let mut bdy_verts: FxHashMap<usize, usize> =
            msh.faces().flatten().map(|i| (i, 0)).collect();
        let n_bdy_verts = bdy_verts.len();

        let mut verts = Vec::with_capacity(n_bdy_verts + n_edges + n_elems);
        for (&i_old, i_new) in &mut bdy_verts {
            *i_new = verts.len();
            verts.push(msh.vert(i_old));
        }
        let vert_ids_bdy = |i: usize| *bdy_verts.get(&i).unwrap();

        // vertices: edge centers
        verts.resize(verts.len() + n_edges, Vert2d::zeros());
        let vert_idx_edge = |i: usize| i + n_bdy_verts;
        for (&edge, &i_edge) in &all_edges {
            let ge = GEdge::from([msh.vert(edge[0]), msh.vert(edge[1])]);
            verts[vert_idx_edge(i_edge)] = ge.center();
        }

        // vertices: triangle centers
        let mut vert_idx_elem = vec![usize::MAX; n_elems];
        for (i_elem, e) in msh.elems().enumerate() {
            let ge = msh.gelem(&e);
            let center = Self::get_tri_center(&ge, t);
            match center {
                DualCellCenter::Vertex(center) => {
                    vert_idx_elem[i_elem] = verts.len();
                    verts.push(center);
                }
                DualCellCenter::Face(f) => {
                    let edge = Edge::from([e[f[0]], e[f[1]]]).sorted();
                    let i_edge = *all_edges.get(&edge).unwrap();
                    vert_idx_elem[i_elem] = vert_idx_edge(i_edge);
                }
            }
        }

        // faces and elements
        let n_poly_faces = 3 * msh.n_elems() + 2 * msh.n_faces();
        let mut faces = Vec::with_capacity(n_poly_faces);
        let mut ftags = Vec::with_capacity(n_poly_faces);

        let mut poly_to_face_ptr = vec![0; msh.n_verts() + 1];

        // internal faces
        for e in msh.elems() {
            for edg in e.edges() {
                poly_to_face_ptr[edg[0] + 1] += 1;
                poly_to_face_ptr[edg[1] + 1] += 1;
            }
        }

        // boundary faces
        for f in msh.faces() {
            for v in f {
                poly_to_face_ptr[v + 1] += 1;
            }
        }

        for i in 0..msh.n_verts() {
            poly_to_face_ptr[i + 1] += poly_to_face_ptr[i];
        }

        let mut poly_to_face = vec![(usize::MAX, true); poly_to_face_ptr[msh.n_verts()]];
        let mut edge_normals = vec![Vert2d::zeros(); n_edges];

        let mut n_empty_faces = 0;
        // build internal faces
        for (i_elem, e) in msh.elems().enumerate() {
            for edg in e.edges() {
                let (i_edge, sgn) = if edg[0] < edg[1] {
                    (*all_edges.get(&edg).unwrap(), 1.0)
                } else {
                    let tmp = Edge::from([edg[1], edg[0]]);
                    (*all_edges.get(&tmp).unwrap(), -1.0)
                };
                let face = Edge::from([vert_idx_edge(i_edge), vert_idx_elem[i_elem]]);
                if face[0] == face[1] {
                    n_empty_faces += 1;
                } else {
                    let gf = GEdge::from([verts[face[0]], verts[face[1]]]);
                    edge_normals[i_edge] += sgn * gf.normal();

                    let i_new_face = faces.len();
                    faces.push(face);
                    ftags.push(0);

                    let mut ok = false;
                    let slice =
                        &mut poly_to_face[poly_to_face_ptr[edg[0]]..poly_to_face_ptr[edg[0] + 1]];
                    for j in slice {
                        if j.0 == usize::MAX {
                            *j = (i_new_face, true);
                            ok = true;
                            break;
                        }
                    }
                    assert!(ok);

                    let mut ok = false;
                    let slice =
                        &mut poly_to_face[poly_to_face_ptr[edg[1]]..poly_to_face_ptr[edg[1] + 1]];
                    for j in slice {
                        if j.0 == usize::MAX {
                            *j = (i_new_face, false);
                            ok = true;
                            break;
                        }
                    }
                    assert!(ok);
                }
            }
        }

        // build boundary faces
        let mut bdy_faces = Vec::with_capacity(msh.n_faces() * 3);

        for (f, tag) in msh.faces().zip(msh.ftags()) {
            let tmp = f.sorted();
            let i_edge = *all_edges.get(&tmp).unwrap();

            let face = Edge::from([vert_ids_bdy(f[0]), vert_idx_edge(i_edge)]);
            if face[0] == face[1] {
                n_empty_faces += 1;
            } else {
                let gf = GEdge::from([verts[face[0]], verts[face[1]]]);
                bdy_faces.push((f[0], tag, gf.normal()));

                let i_new_face = faces.len();
                faces.push(face);
                ftags.push(tag);

                let mut ok = false;
                let slice = &mut poly_to_face[poly_to_face_ptr[f[0]]..poly_to_face_ptr[f[0] + 1]];
                for j in slice {
                    if j.0 == usize::MAX {
                        *j = (i_new_face, true);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);

                let face = Edge::from([vert_idx_edge(i_edge), vert_ids_bdy(f[1])]);
                let gf = GEdge::from([verts[face[0]], verts[face[1]]]);
                bdy_faces.push((f[0], tag, gf.normal()));

                let i_new_face = faces.len();
                faces.push(face);
                ftags.push(tag);

                let mut ok = false;
                let slice = &mut poly_to_face[poly_to_face_ptr[f[1]]..poly_to_face_ptr[f[1] + 1]];
                for j in slice {
                    if j.0 == usize::MAX {
                        *j = (i_new_face, true);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
        }

        assert_eq!(faces.len(), n_poly_faces - n_empty_faces);
        assert_eq!(ftags.len(), n_poly_faces - n_empty_faces);

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
                    new_poly_to_face.push(*v);
                }
            }
            new_poly_to_face_ptr.push(new_poly_to_face.len());
        }

        assert!(!new_poly_to_face.iter().any(|&i| i.0 == usize::MAX));

        let mut edges = vec![Edge::default(); n_edges];
        for (&edg, &i_edg) in &all_edges {
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

    fn edge_normal(&self, i: usize) -> Vert2d {
        self.edge_normals[i]
    }

    fn n_boundary_faces(&self) -> usize {
        self.bdy_faces.len()
    }
    fn par_boundary_faces(&self) -> impl IndexedParallelIterator<Item = (usize, Tag, Vert2d)> + '_ {
        self.bdy_faces.par_iter().copied()
    }
    fn boundary_faces(&self) -> impl ExactSizeIterator<Item = (usize, Tag, Vert2d)> + '_ {
        self.bdy_faces.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        Vert2d,
        dual::{DualMesh, DualMesh2d, DualType, PolyMesh},
        mesh::{Edge, GSimplex, Mesh, Mesh2d, rectangle_mesh},
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_dual_mesh_2d_simple() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 2, 1.0, 2);
        let dual = DualMesh2d::new(&msh, DualType::Median);
        dual.check().unwrap();

        assert_eq!(dual.n_verts(), 11);
        assert_eq!(dual.n_elems(), 4);
        assert_eq!(dual.n_faces(), 14);
        assert_eq!(dual.n_edges(), 5);

        assert!((dual.par_vols().sum::<f64>() - 1.0) < 1e-10);

        let n_empty_faces = dual
            .par_gfaces()
            .filter(|gf| gf.normal().norm() < 1e-12)
            .count();
        assert_eq!(n_empty_faces, 0);
    }

    #[test]
    fn test_dual_mesh_2d_simple_median() {
        let msh = rectangle_mesh::<Mesh2d>(2.0, 3, 1.0, 2);
        let dual = DualMesh2d::new(&msh, DualType::Median);
        dual.check().unwrap();

        assert_eq!(dual.n_verts(), 6 + 9 + 4);
        assert_eq!(dual.n_elems(), 6);
        assert_eq!(dual.n_faces(), 3 * 4 + 2 * 6);

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);

        let n_empty_faces = dual
            .par_gfaces()
            .filter(|gf| gf.normal().norm() < 1e-10)
            .count();
        assert_eq!(n_empty_faces, 0);

        let mut res = HashMap::new();
        res.insert(Edge::from([0, 1]), Vert2d::new(1. / 3., -1. / 6.));
        res.insert(Edge::from([0, 4]), Vert2d::new(1. / 3., 1. / 3.));
        res.insert(Edge::from([0, 3]), Vert2d::new(-1. / 6., 1. / 3.));
        res.insert(Edge::from([3, 4]), Vert2d::new(1. / 3., -1. / 6.));
        res.insert(Edge::from([1, 4]), Vert2d::new(-1. / 3., 2. / 3.));
        res.insert(Edge::from([1, 2]), Vert2d::new(1. / 3., -1. / 6.));
        res.insert(Edge::from([1, 5]), Vert2d::new(1. / 3., 1. / 3.));
        res.insert(Edge::from([4, 5]), Vert2d::new(1. / 3., -1. / 6.));
        res.insert(Edge::from([2, 5]), Vert2d::new(-1. / 6., 1. / 3.));

        dual.par_edges_and_normals().for_each(|(e, n)| {
            let n_res = *res.get(&e).unwrap();
            assert!((n - n_res).norm() < 1e-10);
        });
    }

    #[test]
    fn test_dual_mesh_2d_simple_barth() {
        let msh = rectangle_mesh::<Mesh2d>(2.0, 3, 1.0, 2);
        let dual = DualMesh2d::new(&msh, DualType::Barth);
        dual.check().unwrap();

        let n_empty_faces = dual
            .par_gfaces()
            .filter(|gf| gf.normal().norm() < 1e-10)
            .count();
        assert_eq!(n_empty_faces, 0);

        let n_faces_removed = 4;
        assert_eq!(dual.n_verts(), 6 + 9 + 4 - n_faces_removed);
        assert_eq!(dual.n_elems(), 6);
        assert_eq!(dual.n_faces(), 3 * 4 + 2 * 6 - n_faces_removed);

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);

        let mut res = HashMap::new();
        res.insert(Edge::from([0, 1]), Vert2d::new(0.5, 0.0));
        res.insert(Edge::from([0, 4]), Vert2d::new(0.0, 0.0));
        res.insert(Edge::from([0, 3]), Vert2d::new(0.0, 0.5));
        res.insert(Edge::from([3, 4]), Vert2d::new(0.5, 0.0));
        res.insert(Edge::from([1, 4]), Vert2d::new(0.0, 1.0));
        res.insert(Edge::from([1, 2]), Vert2d::new(0.5, 0.0));
        res.insert(Edge::from([1, 5]), Vert2d::new(0.0, 0.0));
        res.insert(Edge::from([4, 5]), Vert2d::new(0.5, 0.0));
        res.insert(Edge::from([2, 5]), Vert2d::new(0.0, 0.5));

        dual.par_edges_and_normals().for_each(|(e, n)| {
            let n_res = *res.get(&e).unwrap();
            assert!((n - n_res).norm() < 1e-10);
        });
    }

    #[test]
    fn test_dual_mesh_2d_simple_barth_2() {
        let msh = rectangle_mesh::<Mesh2d>(2.0, 30, 1.0, 20).random_shuffle();
        let dual = DualMesh2d::new(&msh, DualType::Barth);
        dual.check().unwrap();

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);
    }
}
