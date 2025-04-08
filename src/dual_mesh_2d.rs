use crate::dual_mesh::circumcenter_bcoords;
use crate::mesh::cell_vertex;
use crate::{
    dual_mesh::{DualMesh, DualType},
    mesh::{cell_center, sort_elem_min_ids, Mesh},
    mesh_2d::Mesh2d,
    Edge, Result, Tag, Vert2d,
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashMap;
use vtkio::{
    model::{
        Attributes, ByteOrder, CellType, Cells, DataSet, UnstructuredGridPiece, Version,
        VertexNumbers,
    },
    IOBuffer, Vtk,
};

pub struct DualMesh2d {
    verts: Vec<Vert2d>,
    faces: Vec<Edge>,
    ftags: Vec<Tag>,
    elem_to_face_ptr: Vec<usize>,
    elem_to_face: Vec<(usize, bool)>,
    edges: Vec<Edge>,
    edge_normals: Vec<Vert2d>,
    bdy_faces: Vec<(usize, Tag, Vert2d)>,
}

impl DualMesh2d {
    fn get_center(v: [&Vert2d; 3], t: DualType) -> Vert2d {
        match t {
            DualType::Median => cell_center(v),
            DualType::Barth => {
                let bcoords = circumcenter_bcoords(v);
                if bcoords.iter().all(|&x| x >= 0.0) {
                    cell_vertex(v, &bcoords)
                } else {
                    let l0 = (v[2] - v[1]).norm();
                    let l1 = (v[2] - v[0]).norm();
                    let l2 = (v[0] - v[1]).norm();
                    if l0 > l1 && l0 > l2 {
                        cell_center([v[1], v[2]])
                    } else if l1 > l0 && l1 > l2 {
                        cell_center([v[2], v[0]])
                    } else {
                        cell_center([v[1], v[0]])
                    }
                }
            }
        }
    }

    pub fn write_vtk(&self, file_name: &str) -> Result<()> {
        let mut coords = Vec::with_capacity(3 * self.n_verts());
        self.verts.iter().for_each(|&p| {
            coords.push(p[0]);
            coords.push(p[1]);
            coords.push(0.0);
        });

        let cell_type = CellType::Polygon;
        let mut connectivity = Vec::with_capacity(self.elem_to_face_ptr.len());
        let mut offsets = Vec::with_capacity(self.n_elems());
        let mut offset = 0;

        for i_elem in 0..self.n_elems() {
            let elem = self.elem(i_elem);
            let mut mask = vec![true; elem.len()];
            let f = self.face(elem[0].0);
            offset += elem.len() as u64;
            offsets.push(offset);
            let first = f[0] as u64;
            connectivity.push(f[0] as u64);
            connectivity.push(f[1] as u64);
            mask[0] = false;
            while mask.iter().any(|&x| x) {
                let last = *connectivity.last().unwrap();
                let (i, next) = elem
                    .iter()
                    .zip(mask.iter())
                    .enumerate()
                    .map(|(i, ((i_face, _), &m))| {
                        let f = self.face(*i_face);
                        if !m {
                            None
                        } else if f[0] as u64 == last {
                            Some((i, f[1]))
                        } else if f[1] as u64 == last {
                            Some((i, f[0]))
                        } else {
                            None
                        }
                    })
                    .find(|x| x.is_some())
                    .unwrap()
                    .unwrap();
                mask[i] = false;
                if next as u64 != first {
                    connectivity.push(next as u64);
                }
            }
        }

        let vtk = Vtk {
            version: Version { major: 1, minor: 0 },
            title: String::new(),
            byte_order: ByteOrder::LittleEndian,
            file_path: None,
            data: DataSet::inline(UnstructuredGridPiece {
                points: IOBuffer::F64(coords),
                cells: Cells {
                    cell_verts: VertexNumbers::XML {
                        connectivity,
                        offsets,
                    },
                    types: vec![cell_type; self.n_elems()],
                },
                data: Attributes {
                    point: Vec::new(),
                    cell: Vec::new(),
                },
            }),
        };

        vtk.export(file_name)?;

        Ok(())
    }
}

impl DualMesh<2, 3, 2> for DualMesh2d {
    fn new<M: Mesh<2, 3, 2>>(msh: &M, t: DualType) -> Self {
        let all_edges = msh.compute_edges();

        let n_edges = all_edges.len();
        let n_elems = msh.n_elems();

        let mut edges = vec![[0; 2]; n_edges];
        for (&edg, &i_edg) in all_edges.iter() {
            edges[i_edg] = edg;
        }

        let mut bdy_verts: FxHashMap<usize, usize> =
            msh.seq_faces().flatten().map(|&i| (i, 0)).collect();
        bdy_verts
            .iter_mut()
            .enumerate()
            .for_each(|(i, (_, i_new))| *i_new = i);
        let n_bdy_verts = bdy_verts.len();

        let vert_idx_edge = |i: usize| i + n_bdy_verts;
        let vert_idx_elem = |i: usize| i + n_bdy_verts + n_edges;

        let mut verts = vec![Vert2d::zeros(); n_bdy_verts + n_edges + n_elems];
        for (&i_old, &i_new) in bdy_verts.iter() {
            verts[i_new] = *msh.vert(i_old);
        }

        for (i_edge, edge) in edges.iter().enumerate() {
            verts[vert_idx_edge(i_edge)] = cell_center([msh.vert(edge[0]), msh.vert(edge[1])]);
        }

        for (i_elem, ge) in msh.seq_gelems().enumerate() {
            verts[vert_idx_elem(i_elem)] = Self::get_center(ge, t);
        }

        let mut elem_to_face_ptr = vec![0; msh.n_verts() + 1];
        for f in msh.seq_faces() {
            for v in f {
                elem_to_face_ptr[v + 1] += 1;
            }
        }

        let elem_to_edges = M::elem_to_edges();
        for e in msh.seq_elems() {
            for edg in &elem_to_edges {
                elem_to_face_ptr[e[edg[0]] + 1] += 1;
                elem_to_face_ptr[e[edg[1]] + 1] += 1;
            }
        }

        for i in 0..msh.n_verts() {
            elem_to_face_ptr[i + 1] += elem_to_face_ptr[i];
        }

        let mut elem_to_face = vec![(usize::MAX, true); elem_to_face_ptr[msh.n_verts()]];

        let mut edge_normals = vec![Vert2d::zeros(); n_edges];

        let mut faces = vec![[usize::MAX; 2]; 3 * msh.n_elems() + 2 * msh.n_faces()];
        let mut ftags = vec![0; 3 * msh.n_elems() + 2 * msh.n_faces()];

        let mut bdy_faces = Vec::with_capacity(msh.n_faces() * 3);

        let mut i_face = 0;

        for (f, tag) in msh.seq_faces().zip(msh.seq_ftags()) {
            let v0 = f[0];
            let v1 = f[1];

            let mut n = M::normal(msh.gface(f));
            n.iter_mut().for_each(|x| *x *= 0.5);
            bdy_faces.push((v0, tag, n));
            bdy_faces.push((v1, tag, n));

            let mut tmp = *f;
            tmp.sort();
            let i_edge = *all_edges.get(&tmp).unwrap();

            faces[i_face] = [*bdy_verts.get(&v0).unwrap(), vert_idx_edge(i_edge)];
            ftags[i_face] = tag;
            let mut ok = false;
            let slice = &mut elem_to_face[elem_to_face_ptr[v0]..elem_to_face_ptr[v0 + 1]];
            for j in slice {
                if j.0 == usize::MAX {
                    *j = (i_face, true);
                    ok = true;
                    break;
                }
            }
            assert!(ok);
            i_face += 1;

            faces[i_face] = [vert_idx_edge(i_edge), *bdy_verts.get(&v1).unwrap()];
            ftags[i_face] = tag;
            let mut ok = false;
            let slice = &mut elem_to_face[elem_to_face_ptr[v1]..elem_to_face_ptr[v1 + 1]];
            for j in slice {
                if j.0 == usize::MAX {
                    *j = (i_face, true);
                    ok = true;
                    break;
                }
            }
            assert!(ok);

            i_face += 1;
        }

        for (i_elem, e) in msh.seq_elems().enumerate() {
            for edge in &elem_to_edges {
                let mut edge = [e[edge[0]], e[edge[1]]];
                let v0 = edge[0];
                let v1 = edge[1];

                edge.sort();
                let &i_edge = all_edges.get(&edge).unwrap();
                let face = [vert_idx_edge(i_edge), vert_idx_elem(i_elem)];
                faces[i_face] = face;
                let mut ok = false;
                let slice = &mut elem_to_face[elem_to_face_ptr[v0]..elem_to_face_ptr[v0 + 1]];
                for j in slice {
                    if j.0 == usize::MAX {
                        *j = (i_face, true);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);

                let mut ok = false;
                let slice = &mut elem_to_face[elem_to_face_ptr[v1]..elem_to_face_ptr[v1 + 1]];
                for j in slice {
                    if j.0 == usize::MAX {
                        *j = (i_face, false);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);

                let gf = [&verts[face[0]], &verts[face[1]]];
                let n = Self::normal(gf);
                if v0 < v1 {
                    edge_normals[i_edge]
                        .iter_mut()
                        .zip(n.iter())
                        .for_each(|(x, y)| *x += y);
                } else {
                    edge_normals[i_edge]
                        .iter_mut()
                        .zip(n.iter())
                        .for_each(|(x, y)| *x -= y);
                }

                i_face += 1;
            }
        }

        assert!(!elem_to_face.iter().any(|&i| i.0 == usize::MAX));

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
            elem_to_face_ptr,
            elem_to_face,
            edges,
            edge_normals,
            bdy_faces,
        }
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

    fn n_faces(&self) -> usize {
        self.faces.len()
    }

    fn face(&self, i: usize) -> Edge {
        self.faces[i]
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
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

    fn normal(v: [&Vert2d; 2]) -> Vert2d {
        Mesh2d::normal(v)
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

    use super::{DualMesh, DualMesh2d};
    use crate::dual_mesh::circumcenter_bcoords;
    use crate::dual_mesh::DualType;
    use crate::mesh_2d::rectangle_mesh;
    use crate::mesh_2d::Mesh2d;
    use crate::Vert2d;
    use rayon::iter::ParallelIterator;

    #[test]
    fn test_dual_mesh_2d_simple() {
        let msh = rectangle_mesh::<Mesh2d>(1.0, 2, 1.0, 2);
        let dual = DualMesh2d::new(&msh, DualType::Median);
        assert!(dual.is_ok());

        assert_eq!(dual.n_verts(), 11);
        assert_eq!(dual.n_elems(), 4);
        assert_eq!(dual.n_faces(), 14);
        assert_eq!(dual.n_edges(), 5);

        assert!((dual.vols().sum::<f64>() - 1.0) < 1e-10);

        let n_empty_faces = dual
            .gfaces()
            .filter(|&gf| DualMesh2d::normal(gf).norm() < 1e-12)
            .count();
        assert_eq!(n_empty_faces, 0);
    }

    #[test]
    fn test_circumcenter() {
        let p0 = Vert2d::new(0.0, 0.0);
        let p1 = Vert2d::new(4.0, 0.0);
        let bcoords = circumcenter_bcoords([&p0, &p1]);
        assert!((bcoords[0] - 0.5).abs() < 1e-10);
        assert!((bcoords[1] - 0.5).abs() < 1e-10);

        let p2 = Vert2d::new(0.0, 4.0);
        let bcoords = circumcenter_bcoords([&p0, &p1, &p2]);
        assert!((bcoords[0] - 0.0).abs() < 1e-10);
        assert!((bcoords[1] - 0.5).abs() < 1e-10);
        assert!((bcoords[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_dual_mesh_2d_simple_median() {
        let msh = rectangle_mesh::<Mesh2d>(2.0, 3, 1.0, 2);
        let dual = DualMesh2d::new(&msh, DualType::Median);
        assert!(dual.is_ok());

        assert_eq!(dual.n_verts(), 6 + 9 + 4);
        assert_eq!(dual.n_elems(), 6);
        assert_eq!(dual.n_faces(), 3 * 4 + 2 * 6);

        assert!((dual.vols().sum::<f64>() - 2.0) < 1e-10);

        let n_empty_faces = dual
            .gfaces()
            .filter(|&gf| DualMesh2d::normal(gf).norm() < 1e-10)
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
        assert!(dual.is_ok());

        assert_eq!(dual.n_verts(), 6 + 9 + 4);
        assert_eq!(dual.n_elems(), 6);
        assert_eq!(dual.n_faces(), 3 * 4 + 2 * 6);

        assert!((dual.vols().sum::<f64>() - 2.0) < 1e-10);

        let n_empty_faces = dual
            .gfaces()
            .filter(|&gf| DualMesh2d::normal(gf).norm() < 1e-10)
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
}
