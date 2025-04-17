use crate::{
    dual_mesh::{DualMesh, DualType},
    mesh::{cell_center, sort_elem_min_ids, Mesh},
    poly_mesh::{PolyMesh, PolyMeshType},
    simplices::Simplex,
    Edge, Tag, Tetrahedron, Triangle, Vert3d,
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;

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
    fn get_center(v: [&Vert3d; 4], t: DualType) -> Vert3d {
        match t {
            DualType::Median => cell_center(v),
            DualType::Barth => {
                unimplemented!()
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

        // faces
        let all_faces = msh.compute_faces();
        let n_faces = all_faces.len();

        let n_elems = msh.n_elems();

        let vert_ids_bdy = |i: usize| *bdy_verts.get(&i).unwrap();
        let vert_idx_edge = |i: usize| i + n_bdy_verts;
        let vert_idx_face = |i: usize| i + n_bdy_verts + n_edges;
        let vert_idx_elem = |i: usize| i + n_bdy_verts + n_edges + n_faces;

        // vertices
        let mut verts = vec![Vert3d::zeros(); n_bdy_verts + n_edges + n_faces + n_elems];
        for (&i_old, &i_new) in bdy_verts.iter() {
            verts[i_new] = *msh.vert(i_old);
        }

        for (&edge, &i_edge) in all_edges.iter() {
            verts[vert_idx_edge(i_edge)] = cell_center([msh.vert(edge[0]), msh.vert(edge[1])]);
        }

        for (&face, &[i_face, _, _]) in all_faces.iter() {
            verts[vert_idx_face(i_face)] =
                cell_center([msh.vert(face[0]), msh.vert(face[1]), msh.vert(face[2])]);
        }

        for (i_elem, ge) in msh.seq_gelems().enumerate() {
            verts[vert_idx_elem(i_elem)] = Self::get_center(ge, t);
        }

        // faces and polyhedra
        let elem_to_edges = Tetrahedron::edges();
        let face_to_edges = Triangle::edges();
        let elem_to_faces = M::elem_to_faces();

        let n_poly_faces = 12 * msh.n_elems() + 6 * msh.n_faces();
        let mut faces = Vec::with_capacity(n_poly_faces);
        let mut ftags = Vec::with_capacity(n_poly_faces);

        let mut poly_to_face_ptr = vec![0; msh.n_verts() + 1];
        // internal faces
        for e in msh.seq_elems() {
            for edg in &elem_to_edges {
                poly_to_face_ptr[e[edg[0]] + 1] += 2;
                poly_to_face_ptr[e[edg[1]] + 1] += 2;
            }
        }

        // boundary faces
        for f in msh.seq_faces() {
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

        // build internal faces
        for (i_elem, e) in msh.seq_elems().enumerate() {
            for f in &elem_to_faces {
                let f = [e[f[0]], e[f[1]], e[f[2]]];
                let mut tmp = f;
                tmp.sort();
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
                        vert_idx_elem(i_elem),
                        vert_idx_face(i_face),
                    ];
                    let gf = [&verts[face[0]], &verts[face[1]], &verts[face[2]]];
                    edge_normals[i_edge] += sgn * Triangle::normal(gf);

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
        let mut bdy_faces = Vec::with_capacity(msh.n_faces() * 6);

        for (f, tag) in msh.seq_faces().zip(msh.seq_ftags()) {
            let mut tmp = *f;
            tmp.sort();
            let i_face = all_faces.get(&tmp).unwrap()[0];
            for edg in &face_to_edges {
                let edg = [f[edg[0]], f[edg[1]]];
                let mut tmp = edg;
                tmp.sort();
                let i_edge = all_edges.get(&tmp).unwrap();

                let face = [
                    vert_ids_bdy(edg[0]),
                    vert_idx_edge(*i_edge),
                    vert_idx_face(i_face),
                ];
                let gf = [&verts[face[0]], &verts[face[1]], &verts[face[2]]];
                bdy_faces.push((edg[0], tag, Triangle::normal(gf)));

                let i_new_face = faces.len();
                faces.push(face);
                ftags.push(tag);

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

                let face = [
                    vert_ids_bdy(edg[1]),
                    vert_idx_face(i_face),
                    vert_idx_edge(*i_edge),
                ];
                let gf = [&verts[face[0]], &verts[face[1]], &verts[face[2]]];
                bdy_faces.push((edg[0], tag, Triangle::normal(gf)));

                let i_new_face = faces.len();
                faces.push(face);
                ftags.push(tag);

                let mut ok = false;
                let slice =
                    &mut poly_to_face[poly_to_face_ptr[edg[1]]..poly_to_face_ptr[edg[1] + 1]];
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

    fn edge_normal(&self, i: usize) -> Vert3d {
        self.edge_normals[i]
    }

    fn n_boundary_faces(&self) -> usize {
        self.bdy_faces.len()
    }
    fn boundary_faces(&self) -> impl IndexedParallelIterator<Item = (usize, Tag, Vert3d)> + '_ {
        self.bdy_faces.par_iter().copied()
    }
    fn seq_boundary_faces(&self) -> impl ExactSizeIterator<Item = (usize, Tag, Vert3d)> + '_ {
        self.bdy_faces.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::{DualMesh, DualMesh3d};
    use crate::{
        boundary_mesh_3d::BoundaryMesh3d,
        dual_mesh::DualType,
        mesh::Mesh,
        mesh_3d::{box_mesh, Mesh3d},
        poly_mesh::{PolyMesh, SimplePolyMesh},
    };
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_dual_mesh_3d_simple_median() {
        let msh = box_mesh::<Mesh3d>(1.0, 10, 2.0, 15, 1.0, 20);
        let dual = DualMesh3d::new(&msh, DualType::Median);
        dual.check().unwrap();

        assert!((dual.vols().sum::<f64>() - 2.0) < 1e-10);

        dual.write_vtk("dual3d.vtu").unwrap();

        let (bdy, _): (BoundaryMesh3d, _) = dual.boundary();
        bdy.write_vtk("dual3d_bdy.vtu").unwrap();

        let poly = SimplePolyMesh::<3>::from(&dual, true);
        poly.write_vtk("dual3d_simplified.vtu").unwrap();
    }
}
