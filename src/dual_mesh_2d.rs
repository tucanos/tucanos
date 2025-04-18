use crate::dual_mesh::circumcenter_bcoords;
use crate::mesh::cell_vertex;
use crate::poly_mesh::{PolyMesh, PolyMeshType};
use crate::{
    dual_mesh::{DualMesh, DualType},
    mesh::{cell_center, sort_elem_min_ids, Mesh},
    simplices::Simplex,
    Edge, Tag, Triangle, Vert2d,
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;

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
    fn get_tri_center(v: [&Vert2d; 3], t: DualType) -> Vert2d {
        match t {
            DualType::Median => cell_center(v),
            DualType::Barth | DualType::ThresholdBarth(_) => {
                let f = match t {
                    DualType::Barth => 0.0,
                    DualType::ThresholdBarth(l) => l,
                    _ => unreachable!(),
                };
                let bcoords = circumcenter_bcoords(v);
                if bcoords.iter().all(|&x| x >= f) {
                    cell_vertex(v, bcoords)
                } else {
                    if bcoords[0] < f {
                        cell_center([v[1], v[2]])
                    } else if bcoords[1] < f {
                        cell_center([v[2], v[0]])
                    } else {
                        cell_center([v[1], v[0]])
                    }
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

    fn vert(&self, i: usize) -> &Vert2d {
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

impl DualMesh<2, 3, 2> for DualMesh2d {
    fn new<M: Mesh<2, 3, 2>>(msh: &M, t: DualType) -> Self {
        // boundary vertices
        let mut bdy_verts: FxHashMap<usize, usize> =
            msh.seq_faces().flatten().map(|&i| (i, 0)).collect();
        bdy_verts
            .iter_mut()
            .enumerate()
            .for_each(|(i, (_, i_new))| *i_new = i);
        let n_bdy_verts = bdy_verts.len();

        // edges
        let all_edges = msh.compute_edges();
        let n_edges = all_edges.len();

        let n_elems = msh.n_elems();

        let vert_ids_bdy = |i: usize| *bdy_verts.get(&i).unwrap();
        let vert_idx_edge = |i: usize| i + n_bdy_verts;
        let vert_idx_elem = |i: usize| i + n_bdy_verts + n_edges;

        // vertices
        let mut verts = vec![Vert2d::zeros(); n_bdy_verts + n_edges + n_elems];
        for (&i_old, &i_new) in bdy_verts.iter() {
            verts[i_new] = *msh.vert(i_old);
        }

        for (&edge, &i_edge) in all_edges.iter() {
            verts[vert_idx_edge(i_edge)] = cell_center([msh.vert(edge[0]), msh.vert(edge[1])]);
        }

        for (i_elem, ge) in msh.seq_gelems().enumerate() {
            verts[vert_idx_elem(i_elem)] = Self::get_tri_center(ge, t);
        }

        // faces and polyhedra
        let elem_to_edges = Triangle::edges();

        let n_poly_faces = 3 * msh.n_elems() + 2 * msh.n_faces();
        let mut faces = Vec::with_capacity(n_poly_faces);
        let mut ftags = Vec::with_capacity(n_poly_faces);

        let mut poly_to_face_ptr = vec![0; msh.n_verts() + 1];

        // internal faces
        for e in msh.seq_elems() {
            for edg in &elem_to_edges {
                poly_to_face_ptr[e[edg[0]] + 1] += 1;
                poly_to_face_ptr[e[edg[1]] + 1] += 1;
            }
        }

        // boundary faces
        for f in msh.seq_faces() {
            for v in f {
                poly_to_face_ptr[v + 1] += 1;
            }
        }

        for i in 0..msh.n_verts() {
            poly_to_face_ptr[i + 1] += poly_to_face_ptr[i];
        }

        let mut poly_to_face = vec![(usize::MAX, true); poly_to_face_ptr[msh.n_verts()]];
        let mut edge_normals = vec![Vert2d::zeros(); n_edges];

        // build internal faces
        for (i_elem, e) in msh.seq_elems().enumerate() {
            for edg in &elem_to_edges {
                let edg = [e[edg[0]], e[edg[1]]];
                let (i_edge, sgn) = if edg[0] < edg[1] {
                    (*all_edges.get(&edg).unwrap(), 1.0)
                } else {
                    let tmp = [edg[1], edg[0]];
                    (*all_edges.get(&tmp).unwrap(), -1.0)
                };
                let face = [vert_idx_edge(i_edge), vert_idx_elem(i_elem)];
                let gf = [&verts[face[0]], &verts[face[1]]];
                edge_normals[i_edge] += sgn * Edge::normal(gf);

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

        // build boundary faces
        let mut bdy_faces = Vec::with_capacity(msh.n_faces() * 3);

        for (f, tag) in msh.seq_faces().zip(msh.seq_ftags()) {
            let mut tmp = *f;
            tmp.sort();
            let i_edge = *all_edges.get(&tmp).unwrap();

            let face = [vert_ids_bdy(f[0]), vert_idx_edge(i_edge)];
            let gf = [&verts[face[0]], &verts[face[1]]];
            bdy_faces.push((f[0], tag, Edge::normal(gf)));

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

            let face = [vert_idx_edge(i_edge), vert_ids_bdy(f[1])];
            let gf = [&verts[face[0]], &verts[face[1]]];
            bdy_faces.push((f[0], tag, Edge::normal(gf)));

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

        assert_eq!(faces.len(), n_poly_faces);
        assert_eq!(ftags.len(), n_poly_faces);

        assert!(!poly_to_face.iter().any(|&i| i.0 == usize::MAX));

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
            elem_to_face_ptr: poly_to_face_ptr,
            elem_to_face: poly_to_face,
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
    fn boundary_faces(&self) -> impl IndexedParallelIterator<Item = (usize, Tag, Vert2d)> + '_ {
        self.bdy_faces.par_iter().copied()
    }
    fn seq_boundary_faces(&self) -> impl ExactSizeIterator<Item = (usize, Tag, Vert2d)> + '_ {
        self.bdy_faces.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        dual_mesh::{DualMesh, DualType},
        dual_mesh_2d::DualMesh2d,
        mesh::Mesh,
        mesh_2d::{rectangle_mesh, Mesh2d},
        poly_mesh::PolyMesh,
        simplices::Simplex,
        Edge, Vert2d,
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

        assert!((dual.vols().sum::<f64>() - 1.0) < 1e-10);

        let n_empty_faces = dual
            .gfaces()
            .filter(|&gf| Edge::normal(gf).norm() < 1e-12)
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

        assert!((dual.vols().sum::<f64>() - 2.0) < 1e-10);

        let n_empty_faces = dual
            .gfaces()
            .filter(|&gf| Edge::normal(gf).norm() < 1e-10)
            .count();
        assert_eq!(n_empty_faces, 0);

        let mut res = HashMap::new();
        res.insert([0, 1], Vert2d::new(1. / 3., -1. / 6.));
        res.insert([0, 4], Vert2d::new(1. / 3., 1. / 3.));
        res.insert([0, 3], Vert2d::new(-1. / 6., 1. / 3.));
        res.insert([3, 4], Vert2d::new(1. / 3., -1. / 6.));
        res.insert([1, 4], Vert2d::new(-1. / 3., 2. / 3.));
        res.insert([1, 2], Vert2d::new(1. / 3., -1. / 6.));
        res.insert([1, 5], Vert2d::new(1. / 3., 1. / 3.));
        res.insert([4, 5], Vert2d::new(1. / 3., -1. / 6.));
        res.insert([2, 5], Vert2d::new(-1. / 6., 1. / 3.));

        dual.edges_and_normals().for_each(|(e, n)| {
            let n_res = *res.get(&e).unwrap();
            assert!((n - n_res).norm() < 1e-10);
        });
    }

    #[test]
    fn test_dual_mesh_2d_simple_barth() {
        let msh = rectangle_mesh::<Mesh2d>(2.0, 3, 1.0, 2);
        let dual = DualMesh2d::new(&msh, DualType::Barth);
        dual.check().unwrap();

        assert_eq!(dual.n_verts(), 6 + 9 + 4);
        assert_eq!(dual.n_elems(), 6);
        assert_eq!(dual.n_faces(), 3 * 4 + 2 * 6);

        assert!((dual.vols().sum::<f64>() - 2.0) < 1e-10);

        let n_empty_faces = dual
            .gfaces()
            .filter(|&gf| Edge::normal(gf).norm() < 1e-10)
            .count();
        assert_eq!(n_empty_faces, 4);

        let mut res = HashMap::new();
        res.insert([0, 1], Vert2d::new(0.5, 0.0));
        res.insert([0, 4], Vert2d::new(0.0, 0.0));
        res.insert([0, 3], Vert2d::new(0.0, 0.5));
        res.insert([3, 4], Vert2d::new(0.5, 0.0));
        res.insert([1, 4], Vert2d::new(0.0, 1.0));
        res.insert([1, 2], Vert2d::new(0.5, 0.0));
        res.insert([1, 5], Vert2d::new(0.0, 0.0));
        res.insert([4, 5], Vert2d::new(0.5, 0.0));
        res.insert([2, 5], Vert2d::new(0.0, 0.5));

        dual.edges_and_normals().for_each(|(e, n)| {
            let n_res = *res.get(&e).unwrap();
            assert!((n - n_res).norm() < 1e-10);
        });
    }

    #[test]
    fn test_dual_mesh_2d_simple_barth_2() {
        let msh = rectangle_mesh::<Mesh2d>(2.0, 30, 1.0, 20).random_shuffle();
        let dual = DualMesh2d::new(&msh, DualType::Barth);
        dual.check().unwrap();

        assert!((dual.vols().sum::<f64>() - 2.0) < 1e-10);
    }
}
