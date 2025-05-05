use crate::{
    mesh::Mesh,
    simplices::Simplex,
    vtu_output::{Encoding, VTUFile},
    Cell, Error, Face, Result, Tag, Vertex,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug, Clone, Copy)]
pub enum PolyMeshType {
    Polylines,
    Polygons,
    Polyhedra,
}

pub trait PolyMesh<const D: usize>: Sync + Sized {
    fn poly_type(&self) -> PolyMeshType;
    fn n_verts(&self) -> usize;
    fn vert(&self, i: usize) -> &Vertex<D>;
    fn verts(&self) -> impl IndexedParallelIterator<Item = &Vertex<D>> + Clone + '_ {
        (0..self.n_verts()).into_par_iter().map(|i| self.vert(i))
    }
    fn seq_verts(&self) -> impl ExactSizeIterator<Item = &Vertex<D>> + Clone + '_ {
        (0..self.n_verts()).map(|i| self.vert(i))
    }
    fn n_elems(&self) -> usize;
    fn elem(&self, i: usize) -> &[(usize, bool)];
    fn elems(&self) -> impl IndexedParallelIterator<Item = &[(usize, bool)]> + Clone + '_ {
        (0..self.n_elems()).into_par_iter().map(|i| self.elem(i))
    }
    fn seq_elems(&self) -> impl ExactSizeIterator<Item = &[(usize, bool)]> + Clone + '_ {
        (0..self.n_elems()).map(|i| self.elem(i))
    }
    fn etag(&self, i: usize) -> Tag;
    fn etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        (0..self.n_elems()).into_par_iter().map(|i| self.etag(i))
    }
    fn seq_etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        (0..self.n_elems()).map(|i| self.etag(i))
    }
    fn n_faces(&self) -> usize;
    fn face(&self, i: usize) -> &[usize];
    fn faces(&self) -> impl IndexedParallelIterator<Item = &[usize]> + Clone + '_ {
        (0..self.n_faces()).into_par_iter().map(|i| self.face(i))
    }
    fn seq_faces(&self) -> impl ExactSizeIterator<Item = &[usize]> + Clone + '_ {
        (0..self.n_faces()).map(|i| self.face(i))
    }
    fn ftag(&self, i: usize) -> Tag;
    fn ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        (0..self.n_faces()).into_par_iter().map(|i| self.ftag(i))
    }
    fn seq_ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        (0..self.n_faces()).map(|i| self.ftag(i))
    }
    fn write_vtk(&self, file_name: &str) -> Result<()> {
        let vtu = VTUFile::from_poly_mesh(self, Encoding::Ascii);

        vtu.export(file_name)?;

        Ok(())
    }
}

pub struct SimplePolyMesh<const D: usize> {
    poly_type: PolyMeshType,
    verts: Vec<Vertex<D>>,
    face_to_node_ptr: Vec<usize>,
    face_to_node: Vec<usize>,
    ftags: Vec<Tag>,
    elem_to_face_ptr: Vec<usize>,
    elem_to_face: Vec<(usize, bool)>,
    etags: Vec<Tag>,
}

impl<const D: usize> SimplePolyMesh<D> {
    pub fn simplify<T: PolyMesh<D>>(mesh: &T, simplify: bool) -> Self {
        let poly_type = mesh.poly_type();

        // simplify faces
        let mut face2elems = vec![[usize::MAX, usize::MAX]; mesh.n_faces()];
        for (i_elem, elem) in mesh.seq_elems().enumerate() {
            for &(i_face, orient) in elem {
                if orient {
                    assert_eq!(face2elems[i_face][0], usize::MAX);
                    face2elems[i_face][0] = i_elem;
                } else {
                    assert_eq!(face2elems[i_face][1], usize::MAX);
                    face2elems[i_face][1] = i_elem;
                }
            }
        }

        let mut face_groups =
            FxHashMap::<[usize; 2], Vec<(usize, bool)>>::with_hasher(Default::default());
        for (i_face, elems) in face2elems.iter().enumerate() {
            if let Some(x) = face_groups.get_mut(elems) {
                x.push((i_face, true));
            } else if let Some(x) = face_groups.get_mut(&[elems[1], elems[0]]) {
                x.push((i_face, false));
            } else {
                face_groups.insert(*elems, vec![(i_face, true)]);
            }
        }

        let mut new_faces = Vec::new();
        let mut new_faces_ptr = Vec::new();
        new_faces_ptr.push(0);
        let mut new_faces_elems = Vec::new();
        let mut new_ftags = Vec::new();
        for (elems, faces) in face_groups {
            assert_ne!(elems[0], usize::MAX);
            if elems[1] != usize::MAX && simplify {
                // copy faces to reorient if needed
                let mut tmp_ptr = Vec::new();
                tmp_ptr.push(0);
                let mut tmp = Vec::new();
                for &(i_face, orient) in &faces {
                    if orient {
                        tmp.extend_from_slice(mesh.face(i_face));
                    } else {
                        let mut f = mesh.face(i_face).to_vec();
                        f.reverse();
                        tmp.extend_from_slice(&f);
                    }
                    tmp_ptr.push(tmp.len());
                }
                let n_faces = faces.len();
                let faces = (0..n_faces)
                    .map(|i| {
                        let start = tmp_ptr[i];
                        let end = tmp_ptr[i + 1];
                        &tmp[start..end]
                    })
                    .collect::<Vec<_>>();
                match poly_type {
                    PolyMeshType::Polylines => todo!(),
                    PolyMeshType::Polygons => {
                        let polygons = merge_polylines(&faces);
                        assert_eq!(polygons.len(), 1);

                        new_faces.extend_from_slice(&polygons[0]);
                    }
                    PolyMeshType::Polyhedra => {
                        let polylines = merge_polygons(&faces);
                        assert_eq!(polylines.len(), 1);

                        new_faces.extend_from_slice(&polylines[0]);
                    }
                }

                new_faces_ptr.push(new_faces.len());
                new_faces_elems.push(elems);
                new_ftags.push(0);
            } else {
                for (i_face, _) in faces {
                    new_faces.extend_from_slice(mesh.face(i_face));
                    new_faces_ptr.push(new_faces.len());
                    new_faces_elems.push(elems);
                    new_ftags.push(mesh.ftag(i_face));
                }
            }
        }

        let mut new_elems_ptr = vec![0; mesh.n_elems() + 1];
        for &[i0, i1] in &new_faces_elems {
            new_elems_ptr[i0 + 1] += 1;
            if i1 != usize::MAX {
                new_elems_ptr[i1 + 1] += 1;
            }
        }
        for i in 0..mesh.n_elems() {
            new_elems_ptr[i + 1] += new_elems_ptr[i];
        }
        let mut new_elems = vec![(usize::MAX, false); *new_elems_ptr.last().unwrap()];
        for (i_face, &[i0, i1]) in new_faces_elems.iter().enumerate() {
            let mut ok = false;
            for v in new_elems
                .iter_mut()
                .take(new_elems_ptr[i0 + 1])
                .skip(new_elems_ptr[i0])
            {
                if v.0 == usize::MAX {
                    *v = (i_face, true);
                    ok = true;
                    break;
                }
            }
            assert!(ok);
            if i1 != usize::MAX {
                let mut ok = false;
                for v in new_elems
                    .iter_mut()
                    .take(new_elems_ptr[i1 + 1])
                    .skip(new_elems_ptr[i1])
                {
                    if v.0 == usize::MAX {
                        *v = (i_face, false);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
        }

        let mut res = Self {
            poly_type: mesh.poly_type(),
            verts: mesh.verts().cloned().collect(),
            face_to_node_ptr: new_faces_ptr,
            face_to_node: new_faces,
            ftags: new_ftags,
            elem_to_face_ptr: new_elems_ptr,
            elem_to_face: new_elems,
            etags: mesh.etags().collect(),
        };

        // unused vertices
        let mut new_ids = vec![usize::MAX; mesh.n_verts()];
        let mut new_verts = Vec::new();
        let mut next = 0;
        for face in res.seq_faces() {
            for &i in face {
                if new_ids[i] == usize::MAX {
                    new_ids[i] = next;
                    new_verts.push(*mesh.vert(i));
                    next += 1;
                }
            }
        }
        res.face_to_node.iter_mut().for_each(|i| *i = new_ids[*i]);
        res.verts = new_verts;

        res
    }

    pub fn from_mesh<const C: usize, const F: usize, M: Mesh<D, C, F>>(mesh: &M) -> Self
    where
        Cell<C>: Simplex<C>,
        Face<F>: Simplex<F>,
    {
        let poly_type = match C {
            4 => PolyMeshType::Polyhedra,
            3 => PolyMeshType::Polygons,
            2 => PolyMeshType::Polylines,
            _ => unimplemented!(),
        };
        let all_faces = mesh.compute_faces();
        let tagged_faces = mesh
            .faces()
            .zip(mesh.ftags())
            .map(|(f, t)| {
                let mut f = *f;
                f.sort();
                (f, t)
            })
            .collect::<FxHashMap<_, _>>();

        let mut elem_to_face_ptr = vec![0; mesh.n_elems() + 1];
        let mut elem_to_face = vec![(usize::MAX, true); mesh.n_elems() * C];
        let mut face_to_node_ptr = vec![0; all_faces.len() + 1];
        let mut face_to_node = vec![0; all_faces.len() * F];
        let mut ftags = vec![0; all_faces.len()];

        for i_elem in 0..mesh.n_elems() {
            elem_to_face_ptr[i_elem + 1] = C * (i_elem + 1);
        }
        for (f, &[i_face, i0, i1]) in &all_faces {
            face_to_node_ptr[i_face + 1] = F * (i_face + 1);
            for k in 0..F {
                face_to_node[F * i_face + k] = f[k];
            }
            if i0 != usize::MAX {
                let mut ok = false;
                for v in elem_to_face
                    .iter_mut()
                    .take(elem_to_face_ptr[i0 + 1])
                    .skip(elem_to_face_ptr[i0])
                {
                    if v.0 == usize::MAX {
                        *v = (i_face, true);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
            if i1 != usize::MAX {
                let mut ok = false;
                for v in elem_to_face
                    .iter_mut()
                    .take(elem_to_face_ptr[i1 + 1])
                    .skip(elem_to_face_ptr[i1])
                {
                    if v.0 == usize::MAX {
                        *v = (i_face, false);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
            if i0 == usize::MAX && i1 == usize::MAX {
                let mut f = *f;
                f.sort();
                ftags[i_face] = *tagged_faces.get(&f).unwrap();
            } else {
                ftags[i_face] = 0;
            }
        }

        Self {
            poly_type,
            verts: mesh.verts().cloned().collect(),
            face_to_node_ptr,
            face_to_node,
            ftags,
            elem_to_face_ptr,
            elem_to_face,
            etags: mesh.etags().collect(),
        }
    }
}

impl<const D: usize> PolyMesh<D> for SimplePolyMesh<D> {
    fn poly_type(&self) -> PolyMeshType {
        self.poly_type
    }

    fn n_verts(&self) -> usize {
        self.verts.len()
    }

    fn vert(&self, i: usize) -> &Vertex<D> {
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
        self.face_to_node_ptr.len() - 1
    }

    fn face(&self, i: usize) -> &[usize] {
        let start = self.face_to_node_ptr[i];
        let end = self.face_to_node_ptr[i + 1];
        &self.face_to_node[start..end]
    }

    fn ftag(&self, i: usize) -> Tag {
        self.ftags[i]
    }
}

pub(crate) fn merge_polylines(polylines: &[&[usize]]) -> Vec<Vec<usize>> {
    let mut res = Vec::new();
    let (polyline, next) = try_merge_polylines(polylines);
    res.push(polyline);
    if !next.is_empty() {
        res.append(&mut merge_polylines(&next));
    }
    res
}

fn try_merge_polylines<'a>(polylines: &[&'a [usize]]) -> (Vec<usize>, Vec<&'a [usize]>) {
    let mut mask = vec![true; polylines.len()];
    let mut connectivity = Vec::with_capacity(polylines.len());

    // add the first edge
    connectivity.extend(polylines[0].iter().cloned());
    mask[0] = false;

    loop {
        let mut added_one = false;
        // add at the end?
        let last = *connectivity.last().unwrap();
        for (edg, m) in polylines.iter().zip(mask.iter_mut()) {
            assert!(edg.len() > 1);
            if *m && edg[0] == last {
                *m = false;
                connectivity.extend(edg.iter().cloned().skip(1));
                added_one = true;
                break;
            }
        }
        // add at the start?
        let first = *connectivity.first().unwrap();
        for (&edg, m) in polylines.iter().zip(mask.iter_mut()) {
            assert!(edg.len() > 1);
            if *m && edg[edg.len() - 1] == first {
                *m = false;
                let mut tmp = edg.to_vec();
                tmp.extend(connectivity.iter().cloned().skip(1));
                connectivity = tmp;
                added_one = true;
                break;
            }
        }
        if !added_one {
            break;
        }
    }

    let unmerged = polylines
        .iter()
        .zip(mask.iter())
        .filter(|&(_, m)| *m)
        .map(|(&e, _)| e)
        .collect::<Vec<_>>();
    (connectivity, unmerged)
}

fn polygon_edges(face: &[usize]) -> impl ExactSizeIterator<Item = [usize; 2]> + '_ {
    let n_edgs = face.len();
    (0..n_edgs).map(move |i_edg| {
        if i_edg == n_edgs - 1 {
            [face[n_edgs - 1], face[0]]
        } else {
            [face[i_edg], face[i_edg + 1]]
        }
    })
}

pub(crate) fn merge_polygons(polylines: &[&[usize]]) -> Vec<Vec<usize>> {
    let mut res = Vec::new();
    let (polyline, next) = try_merge_polygons(polylines);
    res.push(polyline);
    if !next.is_empty() {
        res.append(&mut merge_polygons(&next));
    }
    res
}

fn try_merge_two_polygons(p0: &[usize], p1: &[usize]) -> Result<Vec<usize>> {
    for (i0, e0) in polygon_edges(p0).enumerate() {
        for (i1, e1) in polygon_edges(p1).enumerate() {
            if e0[0] == e1[1] && e0[1] == e1[0] {
                let mut p0 = p0.to_vec();
                let mut p1 = p1.to_vec();
                p0.rotate_left(i0 + 1);
                p1.rotate_left(i1 + 1);
                loop {
                    if p0.len() < 3 || p1.len() < 3 {
                        break;
                    }
                    let n0 = p0.len();
                    let n1 = p1.len();
                    if p0[n0 - 2] == p1[1] {
                        p0.pop().unwrap();
                        p1.remove(0);
                    } else if p0[1] == p1[n1 - 2] {
                        p0.remove(0);
                        p1.pop().unwrap();
                    } else {
                        break;
                    }
                }
                let (mut polyline, tmp) = try_merge_polylines(&[&p0, &p1]);
                assert!(tmp.is_empty());
                let n = polyline.len();
                assert_eq!(polyline[0], polyline[n - 1]);
                polyline.pop().unwrap();
                return Ok(polyline);
            }
        }
    }
    Err(Error::from("unable to merge faces"))
}

fn try_merge_polygons<'a>(polygons: &[&'a [usize]]) -> (Vec<usize>, Vec<&'a [usize]>) {
    let mut mask = vec![true; polygons.len()];
    let mut connectivity = Vec::with_capacity(polygons.len());

    // add the first polygon
    connectivity.extend(polygons[0].iter().cloned());
    mask[0] = false;

    loop {
        let mut added_one = false;
        for (face, m) in polygons.iter().zip(mask.iter_mut()) {
            if !*m {
                continue;
            }
            if let Ok(res) = try_merge_two_polygons(&connectivity, face) {
                // check that no edge apears more than once
                let tmp = polygon_edges(&res)
                    .map(|e| if e[0] < e[1] { e } else { [e[1], e[0]] })
                    .collect::<FxHashSet<_>>();
                if tmp.len() == res.len() {
                    connectivity = res;
                    added_one = true;
                    *m = false;
                    break;
                }
            }
        }
        if !added_one {
            break;
        }
    }
    let unmerged = polygons
        .iter()
        .zip(mask.iter())
        .filter(|&(_, m)| *m)
        .map(|(&e, _)| e)
        .collect::<Vec<_>>();
    (connectivity, unmerged)
}

#[cfg(test)]
mod tests {
    use crate::{
        mesh_3d::{box_mesh, Mesh3d},
        poly_mesh::merge_polygons,
    };

    use super::{merge_polylines, PolyMesh, SimplePolyMesh};

    #[test]
    fn test_merge_polylines() {
        let polyline: [&[usize]; 4] = [&[0, 1], &[3, 4, 5], &[2, 3], &[1, 2]];
        let res = merge_polylines(&polyline);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 1, 2, 3, 4, 5]);

        let polyline: [&[usize]; 4] = [&[3, 4, 5], &[2, 3], &[1, 2], &[0, 1]];
        let res = merge_polylines(&polyline);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 1, 2, 3, 4, 5]);

        let polyline: [&[usize]; 4] = [&[3, 4, 5], &[2, 3], &[1, 2], &[1, 0]];
        let res = merge_polylines(&polyline);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0], [1, 2, 3, 4, 5]);
        assert_eq!(res[1], [1, 0]);

        let polyline: [&[usize]; 4] = [&[3, 4, 5], &[2, 3], &[2, 1], &[0, 1]];
        let res = merge_polylines(&polyline);
        assert_eq!(res.len(), 3);
        assert_eq!(res[0], [2, 3, 4, 5]);
        assert_eq!(res[1], [2, 1]);
        assert_eq!(res[2], [0, 1]);
    }

    #[test]
    fn test_merge_polygons() {
        let polygons: [&[usize]; 5] = [&[0, 1, 2], &[1, 3, 2], &[2, 3, 4], &[5, 2, 4], &[0, 2, 5]];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 1, 3, 4, 5]);

        let polygons: [&[usize]; 5] = [&[0, 1, 2], &[2, 3, 4], &[1, 3, 2], &[5, 2, 4], &[0, 2, 5]];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [0, 1, 3, 4, 5]);

        let polygons: [&[usize]; 5] = [&[0, 1, 2], &[3, 2, 4], &[1, 3, 2], &[5, 2, 4], &[0, 2, 5]];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0], [5, 0, 1, 3, 2, 4]);
        assert_eq!(res[1], [3, 2, 4]);
    }

    #[test]
    fn test_merge_polygons_2() {
        let polygons: [&[usize]; 5] = [
            &[0, 1, 7],
            &[1, 2, 3],
            &[3, 4, 5],
            &[5, 6, 7],
            &[7, 1, 3, 5],
        ];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [7, 0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_merge_polygons_3() {
        let polygons: [&[usize]; 8] = [
            &[0, 3, 1],
            &[3, 4, 1],
            &[1, 4, 2],
            &[4, 5, 2],
            &[0, 9, 6, 3],
            &[3, 6, 7, 4],
            &[7, 8, 5, 4],
            &[9, 10, 8, 7, 6],
        ];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [8, 5, 2, 1, 0, 9, 10]);

        let polygons: [&[usize]; 8] = [
            &[0, 3, 1],
            &[3, 4, 1],
            &[1, 4, 2],
            &[4, 5, 2],
            &[0, 9, 6, 3],
            &[9, 10, 8, 7, 6],
            &[7, 8, 5, 4],
            &[3, 6, 7, 4],
        ];
        let res = merge_polygons(&polygons);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0], [5, 2, 1, 0, 9, 10, 8]);
    }

    #[test]
    fn test_vtk_polyhedra() {
        let msh: Mesh3d = box_mesh(1.0, 2, 2.0, 2, 3.0, 2);
        let poly = SimplePolyMesh::from_mesh(&msh);
        poly.write_vtk("poly3d.vtu").unwrap();
    }
}
