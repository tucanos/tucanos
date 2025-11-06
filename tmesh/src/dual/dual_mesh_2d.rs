//! Computation of the dual for `Mesh<2, 3, 2>`
use super::{DualCellCenter, DualMesh, DualType, PolyMesh, PolyMeshType, circumcenter_bcoords};
use crate::{
    Tag, Vert2d,
    mesh::{Edge, GEdge, GSimplex, GTriangle, Idx, Mesh, Simplex, Triangle, sort_elem_min_ids},
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;

/// Dual of a Triangle mesh in 2d
pub struct DualMesh2d<T: Idx> {
    verts: Vec<Vert2d>,
    faces: Vec<Edge<T>>,
    ftags: Vec<Tag>,
    elem_to_face_ptr: Vec<T>,
    elem_to_face: Vec<(T, bool)>,
    etags: Vec<Tag>,
    edges: Vec<Edge<T>>,
    edge_normals: Vec<Vert2d>,
    bdy_faces: Vec<(T, Tag, Vert2d)>,
}

impl<T: Idx> DualMesh2d<T> {
    fn get_tri_center(v: &GTriangle<2>, t: DualType) -> DualCellCenter<T, 2, Triangle<T>> {
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
                    DualCellCenter::Face(Edge::<T>::from_iter([1, 2]))
                } else if bcoords[1] <= f {
                    DualCellCenter::Face(Edge::<T>::from_iter([2, 0]))
                } else {
                    DualCellCenter::Face(Edge::<T>::from_iter([1, 0]))
                }
            }
        }
    }
}

impl<T: Idx> PolyMesh<T, 2> for DualMesh2d<T> {
    fn poly_type(&self) -> PolyMeshType {
        PolyMeshType::Polygons
    }

    fn n_verts(&self) -> T {
        self.verts.len().try_into().unwrap()
    }

    fn vert(&self, i: T) -> Vert2d {
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

impl<T: Idx> DualMesh<T, 2, Triangle<T>> for DualMesh2d<T> {
    #[allow(clippy::too_many_lines)]
    fn new<M: Mesh<T, 2, Triangle<T>>>(msh: &M, t: DualType) -> Self {
        // edges
        let all_edges = msh.edges();
        let n_edges = all_edges.len();

        let n_elems = msh.n_elems().try_into().unwrap();

        // vertices: boundary
        let mut bdy_verts: FxHashMap<T, T> = msh.faces().flatten().map(|i| (i, T::ZERO)).collect();
        let n_bdy_verts = bdy_verts.len();

        let n = n_bdy_verts + n_edges + n_elems;
        let mut verts = Vec::with_capacity(n);
        for (&i_old, i_new) in &mut bdy_verts {
            *i_new = verts.len().try_into().unwrap();
            verts.push(msh.vert(i_old));
        }
        let vert_ids_bdy = |i: T| *bdy_verts.get(&i).unwrap();

        // vertices: edge centers
        verts.resize(verts.len() + n_edges, Vert2d::zeros());
        let vert_idx_edge = |i: T| i + n_bdy_verts.try_into().unwrap();
        for (&edge, &i_edge) in &all_edges {
            let ge = GEdge::from([msh.vert(edge[0]), msh.vert(edge[1])]);
            verts[vert_idx_edge(i_edge).try_into().unwrap()] = ge.center();
        }

        // vertices: triangle centers
        let mut vert_idx_elem = vec![T::MAX; n_elems];
        for (i_elem, e) in msh.elems().enumerate() {
            let ge = msh.gelem(&e);
            let center = Self::get_tri_center(&ge, t);
            match center {
                DualCellCenter::Vertex(center) => {
                    vert_idx_elem[i_elem] = verts.len().try_into().unwrap();
                    verts.push(center);
                }
                DualCellCenter::Face(f) => {
                    let edge =
                        Edge::<T>::from([e[f[0].try_into().unwrap()], e[f[1].try_into().unwrap()]])
                            .sorted();
                    let i_edge = *all_edges.get(&edge).unwrap();
                    vert_idx_elem[i_elem] = vert_idx_edge(i_edge);
                }
            }
        }

        // faces and elements
        let n_poly_faces =
            3 * msh.n_elems().try_into().unwrap() + 2 * msh.n_faces().try_into().unwrap();
        let mut faces = Vec::with_capacity(n_poly_faces);
        let mut ftags = Vec::with_capacity(n_poly_faces);

        let n_verts = msh.n_verts().try_into().unwrap();
        let mut poly_to_face_ptr = vec![0; n_verts + 1];

        // internal faces
        for e in msh.elems() {
            for edg in e.edges() {
                poly_to_face_ptr[edg[0].try_into().unwrap() + 1] += 1;
                poly_to_face_ptr[edg[1].try_into().unwrap() + 1] += 1;
            }
        }

        // boundary faces
        for f in msh.faces() {
            for v in f {
                poly_to_face_ptr[v.try_into().unwrap() + 1] += 1;
            }
        }

        for i in 0..msh.n_verts().try_into().unwrap() {
            poly_to_face_ptr[i + 1] += poly_to_face_ptr[i];
        }

        let mut poly_to_face = vec![(T::MAX, true); poly_to_face_ptr[n_verts]];
        let mut edge_normals = vec![Vert2d::zeros(); n_edges];

        let mut n_empty_faces = 0;
        // build internal faces
        for (i_elem, e) in msh.elems().enumerate() {
            for edg in e.edges() {
                let (i_edge, sgn) = if edg[0] < edg[1] {
                    (*all_edges.get(&edg).unwrap(), 1.0)
                } else {
                    let tmp = Edge::<T>::from([edg[1], edg[0]]);
                    (*all_edges.get(&tmp).unwrap(), -1.0)
                };
                let face = Edge::<T>::from([vert_idx_edge(i_edge), vert_idx_elem[i_elem]]);
                if face[0] == face[1] {
                    n_empty_faces += 1;
                } else {
                    let gf = GEdge::from([
                        verts[face[0].try_into().unwrap()],
                        verts[face[1].try_into().unwrap()],
                    ]);
                    edge_normals[i_edge.try_into().unwrap()] += sgn * gf.normal();

                    let i_new_face = faces.len().try_into().unwrap();
                    faces.push(face);
                    ftags.push(0);

                    let mut ok = false;
                    let slice = &mut poly_to_face[poly_to_face_ptr[edg[0].try_into().unwrap()]
                        ..poly_to_face_ptr[edg[0].try_into().unwrap() + 1]];
                    for j in slice {
                        if j.0 == T::MAX {
                            *j = (i_new_face, true);
                            ok = true;
                            break;
                        }
                    }
                    assert!(ok);

                    let mut ok = false;
                    let slice = &mut poly_to_face[poly_to_face_ptr[edg[1].try_into().unwrap()]
                        ..poly_to_face_ptr[edg[1].try_into().unwrap() + 1]];
                    for j in slice {
                        if j.0 == T::MAX {
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
        let mut bdy_faces = Vec::with_capacity(msh.n_faces().try_into().unwrap() * 3);

        for (f, tag) in msh.faces().zip(msh.ftags()) {
            let tmp = f.sorted();
            let i_edge = *all_edges.get(&tmp).unwrap();

            let face = Edge::<T>::from([vert_ids_bdy(f[0]), vert_idx_edge(i_edge)]);
            if face[0] == face[1] {
                n_empty_faces += 1;
            } else {
                let gf = GEdge::from([
                    verts[face[0].try_into().unwrap()],
                    verts[face[1].try_into().unwrap()],
                ]);
                bdy_faces.push((f[0], tag, gf.normal()));

                let i_new_face = faces.len().try_into().unwrap();
                faces.push(face);
                ftags.push(tag);

                let mut ok = false;
                let slice = &mut poly_to_face[poly_to_face_ptr[f[0].try_into().unwrap()]
                    ..poly_to_face_ptr[f[0].try_into().unwrap() + 1]];
                for j in slice {
                    if j.0 == T::MAX {
                        *j = (i_new_face, true);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);

                let face = Edge::<T>::from([vert_idx_edge(i_edge), vert_ids_bdy(f[1])]);
                let gf = GEdge::from([
                    verts[face[0].try_into().unwrap()],
                    verts[face[1].try_into().unwrap()],
                ]);
                bdy_faces.push((f[0], tag, gf.normal()));

                let i_new_face = faces.len().try_into().unwrap();
                faces.push(face);
                ftags.push(tag);

                let mut ok = false;
                let slice = &mut poly_to_face[poly_to_face_ptr[f[1].try_into().unwrap()]
                    ..poly_to_face_ptr[f[1].try_into().unwrap() + 1]];
                for j in slice {
                    if j.0 == T::MAX {
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
        let n = poly_to_face.iter().filter(|&i| i.0 != T::MAX).count();

        let mut new_poly_to_face_ptr = Vec::with_capacity(poly_to_face_ptr.len());
        new_poly_to_face_ptr.push(T::ZERO);
        let mut new_poly_to_face = Vec::with_capacity(n);
        for i_elem in 0..msh.n_verts().try_into().unwrap() {
            for v in poly_to_face
                .iter()
                .take(poly_to_face_ptr[i_elem + 1])
                .skip(poly_to_face_ptr[i_elem])
            {
                if v.0 != T::MAX {
                    new_poly_to_face.push(*v);
                }
            }
            new_poly_to_face_ptr.push(new_poly_to_face.len().try_into().unwrap());
        }

        assert!(!new_poly_to_face.iter().any(|&i| i.0 == T::MAX));

        let mut edges = vec![Edge::<T>::default(); n_edges];
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

    fn edge_normal(&self, i: T) -> Vert2d {
        self.edge_normals[i.try_into().unwrap()]
    }

    fn n_boundary_faces(&self) -> T {
        self.bdy_faces.len().try_into().unwrap()
    }
    fn par_boundary_faces(&self) -> impl IndexedParallelIterator<Item = (T, Tag, Vert2d)> + '_ {
        self.bdy_faces.par_iter().copied()
    }
    fn boundary_faces(&self) -> impl ExactSizeIterator<Item = (T, Tag, Vert2d)> + '_ {
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
        let msh = rectangle_mesh::<_, Mesh2d>(1.0, 2, 1.0, 2);
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
        let msh = rectangle_mesh::<_, Mesh2d>(2.0, 3, 1.0, 2);
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
        let msh = rectangle_mesh::<_, Mesh2d>(2.0, 3, 1.0, 2);
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
        let msh = rectangle_mesh::<_, Mesh2d>(2.0, 30, 1.0, 20).random_shuffle();
        let dual = DualMesh2d::new(&msh, DualType::Barth);
        dual.check().unwrap();

        assert!((dual.par_vols().sum::<f64>() - 2.0) < 1e-10);
    }
}
