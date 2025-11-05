//! General polyline, polygon and polyhedral meshes
use crate::{
    Error, Result, Tag, Vertex,
    io::{VTUEncoding, VTUFile},
    mesh::{Idx, Mesh, Simplex},
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

/// Polymesh type
#[derive(Debug, Clone, Copy)]
pub enum PolyMeshType {
    /// Polylines
    Polylines,
    /// Polygons
    Polygons,
    /// Polyhedra
    Polyhedra,
}

/// Polylines, polygons or polyhedra meshes in D dimensions
///   - faces are represented by the indices of their vertices (oriented)
///   - elements are indices of their faces and flags indicating if the face is oriented
///     outwards or inwards
pub trait PolyMesh<T: Idx, const D: usize>: Sync + Sized {
    /// Element type
    fn poly_type(&self) -> PolyMeshType;

    /// Number of vertices
    fn n_verts(&self) -> T;

    /// Get the `i`th vertex
    fn vert(&self, i: T) -> Vertex<D>;

    /// Parallel iterator over the mesh vertices
    fn par_verts(&self) -> impl IndexedParallelIterator<Item = Vertex<D>> + Clone + '_ {
        (0..self.n_verts().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.vert(i.try_into().unwrap()))
    }

    /// Sequential iterator over the mesh vertices
    fn verts(&self) -> impl ExactSizeIterator<Item = Vertex<D>> + Clone + '_ {
        (0..self.n_verts().try_into().unwrap()).map(|i| self.vert(i.try_into().unwrap()))
    }

    /// Number of elements
    fn n_elems(&self) -> T;

    /// Get the `i`th face
    fn elem(&self, i: T) -> &[(T, bool)];

    /// Number of vertices in the `i`the element
    fn elem_n_verts(&self, i: T) -> T {
        let mut verts = FxHashSet::with_hasher(FxBuildHasher);
        for &(j, _) in self.elem(i) {
            for k in self.face(j) {
                verts.insert(k);
            }
        }
        verts.len().try_into().unwrap()
    }

    /// Parallel iterator over the mesh elements
    fn par_elems(&self) -> impl IndexedParallelIterator<Item = &[(T, bool)]> + Clone + '_ {
        (0..self.n_elems().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.elem(i.try_into().unwrap()))
    }

    /// Sequantial iterator over the mesh elements
    fn elems(&self) -> impl ExactSizeIterator<Item = &[(T, bool)]> + Clone + '_ {
        (0..self.n_elems().try_into().unwrap()).map(|i| self.elem(i.try_into().unwrap()))
    }

    /// Get the tag of the `i`the element
    fn etag(&self, i: T) -> Tag;

    /// Parallel iterator over the element tags
    fn par_etags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        (0..self.n_elems().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.etag(i.try_into().unwrap()))
    }

    /// Sequential iterator over the element tags
    fn etags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        (0..self.n_elems().try_into().unwrap()).map(|i| self.etag(i.try_into().unwrap()))
    }

    /// Number of faces
    fn n_faces(&self) -> T;

    /// Get the `i`the face
    fn face(&self, i: T) -> &[T];

    /// Parallel iterator over the faces
    fn par_faces(&self) -> impl IndexedParallelIterator<Item = &[T]> + Clone + '_ {
        (0..self.n_faces().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.face(i.try_into().unwrap()))
    }

    /// Sequential iterator over the faces
    fn faces(&self) -> impl ExactSizeIterator<Item = &[T]> + Clone + '_ {
        (0..self.n_faces().try_into().unwrap()).map(|i| self.face(i.try_into().unwrap()))
    }

    /// Get the tag of the `i`th face
    fn ftag(&self, i: T) -> Tag;

    /// Parallel iterator over the face tags
    fn par_ftags(&self) -> impl IndexedParallelIterator<Item = Tag> + Clone + '_ {
        (0..self.n_faces().try_into().unwrap())
            .into_par_iter()
            .map(|i| self.ftag(i.try_into().unwrap()))
    }

    /// Sequential iterator over the face tags
    fn ftags(&self) -> impl ExactSizeIterator<Item = Tag> + Clone + '_ {
        (0..self.n_faces().try_into().unwrap()).map(|i| self.ftag(i.try_into().unwrap()))
    }

    /// Export the mesh to a `.vtu` file
    fn write_vtk(&self, file_name: &str) -> Result<()> {
        let vtu = VTUFile::from_poly_mesh(self, VTUEncoding::Ascii);

        vtu.export(file_name)?;

        Ok(())
    }
}

/// General PolyMesh<D>
pub struct SimplePolyMesh<T: Idx, const D: usize> {
    poly_type: PolyMeshType,
    verts: Vec<Vertex<D>>,
    face_to_node_ptr: Vec<T>,
    face_to_node: Vec<T>,
    ftags: Vec<Tag>,
    elem_to_face_ptr: Vec<T>,
    elem_to_face: Vec<(T, bool)>,
    etags: Vec<Tag>,
}

impl<T: Idx, const D: usize> SimplePolyMesh<T, D> {
    /// Create a new mesh from coordinates, connectivities and tags
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        poly_type: PolyMeshType,
        verts: Vec<Vertex<D>>,
        face_to_node_ptr: Vec<T>,
        face_to_node: Vec<T>,
        ftags: Vec<Tag>,
        elem_to_face_ptr: Vec<T>,
        elem_to_face: Vec<(T, bool)>,
        etags: Vec<Tag>,
    ) -> Self {
        Self {
            poly_type,
            verts,
            face_to_node_ptr,
            face_to_node,
            ftags,
            elem_to_face_ptr,
            elem_to_face,
            etags,
        }
    }

    /// Create an empty mesh with a given element type
    #[must_use]
    pub fn empty(poly_type: PolyMeshType) -> Self {
        Self {
            poly_type,
            verts: Vec::new(),
            face_to_node_ptr: vec![0.try_into().unwrap()],
            face_to_node: Vec::new(),
            ftags: Vec::new(),
            elem_to_face_ptr: vec![0.try_into().unwrap()],
            elem_to_face: Vec::new(),
            etags: Vec::new(),
        }
    }

    /// Insert a vertex and return its index
    pub fn insert_vert(&mut self, v: Vertex<D>) -> T {
        self.verts.push(v);
        (self.verts.len() - 1).try_into().unwrap()
    }

    /// Insert a face and return its index
    pub fn insert_face(&mut self, f: &[T], t: Tag) -> T {
        self.face_to_node.extend(f);
        self.face_to_node_ptr
            .push(self.face_to_node.len().try_into().unwrap());
        self.ftags.push(t);
        (self.ftags.len() - 1).try_into().unwrap()
    }

    /// Insert an element and return its index
    pub fn insert_elem(&mut self, e: &[(T, bool)], t: Tag) -> T {
        self.elem_to_face.extend(e);
        self.elem_to_face_ptr
            .push(self.elem_to_face.len().try_into().unwrap());
        self.etags.push(t);
        (self.etags.len() - 1).try_into().unwrap()
    }

    /// Simplify a `PolyMesh<D>` by merging its faces connecting the same two elements
    #[allow(clippy::too_many_lines)]
    pub fn simplify<P: PolyMesh<T, D>>(mesh: &P, simplify: bool) -> Self {
        let poly_type = mesh.poly_type();

        // simplify faces
        let mut face2elems = vec![[T::MAX, T::MAX]; mesh.n_faces().try_into().unwrap()];
        for (i_elem, elem) in mesh.elems().enumerate() {
            let i_elem = i_elem.try_into().unwrap();
            for &(i_face, orient) in elem {
                let i_face = i_face.try_into().unwrap();
                if orient {
                    assert_eq!(face2elems[i_face][0], T::MAX);
                    face2elems[i_face][0] = i_elem;
                } else {
                    assert_eq!(face2elems[i_face][1], T::MAX);
                    face2elems[i_face][1] = i_elem;
                }
            }
        }

        let mut face_groups = FxHashMap::<[T; 2], Vec<(T, bool)>>::with_hasher(FxBuildHasher);
        for (i_face, elems) in face2elems.iter().enumerate() {
            let i_face = i_face.try_into().unwrap();
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
        new_faces_ptr.push(0.try_into().unwrap());
        let mut new_faces_elems = Vec::new();
        let mut new_ftags = Vec::new();
        for (elems, faces) in face_groups {
            assert_ne!(elems[0], T::MAX);
            if elems[1] != T::MAX && simplify {
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

                new_faces_ptr.push(new_faces.len().try_into().unwrap());
                new_faces_elems.push(elems);
                new_ftags.push(0);
            } else {
                for (i_face, _) in faces {
                    new_faces.extend_from_slice(mesh.face(i_face));
                    new_faces_ptr.push(new_faces.len().try_into().unwrap());
                    new_faces_elems.push(elems);
                    new_ftags.push(mesh.ftag(i_face));
                }
            }
        }

        let mut new_elems_ptr: Vec<T> =
            vec![0.try_into().unwrap(); mesh.n_elems().try_into().unwrap() + 1];
        for &[i0, i1] in &new_faces_elems {
            new_elems_ptr[i0.try_into().unwrap() + 1] += 1.try_into().unwrap();
            if i1 != T::MAX {
                new_elems_ptr[i1.try_into().unwrap() + 1] += 1.try_into().unwrap();
            }
        }
        for i in 0..mesh.n_elems().try_into().unwrap() {
            new_elems_ptr[i + 1] += new_elems_ptr[i];
        }
        let n = *new_elems_ptr.last().unwrap();
        let mut new_elems = vec![(T::MAX, false); n.try_into().unwrap()];
        for (i_face, &[i0, i1]) in new_faces_elems.iter().enumerate() {
            let mut ok = false;
            for v in new_elems
                .iter_mut()
                .take(
                    new_elems_ptr[i0.try_into().unwrap() + 1]
                        .try_into()
                        .unwrap(),
                )
                .skip(new_elems_ptr[i0.try_into().unwrap()].try_into().unwrap())
            {
                if v.0 == T::MAX {
                    *v = (i_face.try_into().unwrap(), true);
                    ok = true;
                    break;
                }
            }
            assert!(ok);
            if i1 != T::MAX {
                let mut ok = false;
                for v in new_elems
                    .iter_mut()
                    .take(
                        new_elems_ptr[i1.try_into().unwrap() + 1]
                            .try_into()
                            .unwrap(),
                    )
                    .skip(new_elems_ptr[i1.try_into().unwrap()].try_into().unwrap())
                {
                    if v.0 == T::MAX {
                        *v = (i_face.try_into().unwrap(), false);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
        }

        let mut res = Self {
            poly_type: mesh.poly_type(),
            verts: mesh.verts().collect(),
            face_to_node_ptr: new_faces_ptr,
            face_to_node: new_faces,
            ftags: new_ftags,
            elem_to_face_ptr: new_elems_ptr,
            elem_to_face: new_elems,
            etags: mesh.etags().collect(),
        };

        // unused vertices
        let mut new_ids = vec![T::MAX; mesh.n_verts().try_into().unwrap()];
        let mut new_verts = Vec::new();
        let mut next = 0.try_into().unwrap();
        for face in res.faces() {
            for &i in face {
                if new_ids[i.try_into().unwrap()] == T::MAX {
                    new_ids[i.try_into().unwrap()] = next;
                    new_verts.push(mesh.vert(i));
                    next += 1.try_into().unwrap();
                }
            }
        }
        res.face_to_node
            .iter_mut()
            .for_each(|i| *i = new_ids[(*i).try_into().unwrap()]);
        res.verts = new_verts;

        res
    }

    /// PolyMesh representation of a `Mesh<D, C, F>`
    pub fn from_mesh<C: Simplex<T>, M: Mesh<T, D, C>>(mesh: &M) -> Self {
        let poly_type = match C::N_VERTS {
            4 => PolyMeshType::Polyhedra,
            3 => PolyMeshType::Polygons,
            2 => PolyMeshType::Polylines,
            _ => unimplemented!(),
        };
        let all_faces = mesh.all_faces();
        let tagged_faces = mesh
            .par_faces()
            .zip(mesh.par_ftags())
            .map(|(f, t)| (f.sorted(), t))
            .collect::<FxHashMap<_, _>>();

        let mut elem_to_face_ptr: Vec<T> =
            vec![0.try_into().unwrap(); mesh.n_elems().try_into().unwrap() + 1];
        let mut elem_to_face =
            vec![(T::MAX, true); mesh.n_elems().try_into().unwrap() * C::N_VERTS];
        let mut face_to_node_ptr: Vec<T> = vec![0.try_into().unwrap(); all_faces.len() + 1];
        let mut face_to_node = vec![0.try_into().unwrap(); all_faces.len() * C::FACE::N_VERTS];
        let mut ftags = vec![0; all_faces.len()];

        for i_elem in 0..mesh.n_elems().try_into().unwrap() {
            elem_to_face_ptr[i_elem + 1] = (C::N_VERTS * (i_elem + 1)).try_into().unwrap();
        }
        for (f, &[i_face, i0, i1]) in &all_faces {
            face_to_node_ptr[i_face.try_into().unwrap() + 1] = (C::FACE::N_VERTS
                * (i_face.try_into().unwrap() + 1))
                .try_into()
                .unwrap();
            for k in 0..C::FACE::N_VERTS {
                face_to_node[C::FACE::N_VERTS * i_face.try_into().unwrap() + k] = f[k];
            }
            if i0 != T::MAX {
                let mut ok = false;
                for v in elem_to_face
                    .iter_mut()
                    .take(
                        elem_to_face_ptr[i0.try_into().unwrap() + 1]
                            .try_into()
                            .unwrap(),
                    )
                    .skip(elem_to_face_ptr[i0.try_into().unwrap()].try_into().unwrap())
                {
                    if v.0 == T::MAX {
                        *v = (i_face, true);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
            if i1 != T::MAX {
                let mut ok = false;
                for v in elem_to_face
                    .iter_mut()
                    .take(
                        elem_to_face_ptr[i1.try_into().unwrap() + 1]
                            .try_into()
                            .unwrap(),
                    )
                    .skip(elem_to_face_ptr[i1.try_into().unwrap()].try_into().unwrap())
                {
                    if v.0 == T::MAX {
                        *v = (i_face, false);
                        ok = true;
                        break;
                    }
                }
                assert!(ok);
            }
            if i0 == T::MAX && i1 == T::MAX {
                let f = f.sorted();
                ftags[i_face.try_into().unwrap()] = *tagged_faces.get(&f).unwrap();
            } else {
                ftags[i_face.try_into().unwrap()] = 0;
            }
        }

        Self {
            poly_type,
            verts: mesh.par_verts().collect(),
            face_to_node_ptr,
            face_to_node,
            ftags,
            elem_to_face_ptr,
            elem_to_face,
            etags: mesh.par_etags().collect(),
        }
    }
}

impl<T: Idx, const D: usize> PolyMesh<T, D> for SimplePolyMesh<T, D> {
    fn poly_type(&self) -> PolyMeshType {
        self.poly_type
    }

    fn n_verts(&self) -> T {
        self.verts.len().try_into().unwrap()
    }

    fn vert(&self, i: T) -> Vertex<D> {
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
        (self.face_to_node_ptr.len() - 1).try_into().unwrap()
    }

    fn face(&self, i: T) -> &[T] {
        let start = self.face_to_node_ptr[i.try_into().unwrap()]
            .try_into()
            .unwrap();
        let end = self.face_to_node_ptr[i.try_into().unwrap() + 1]
            .try_into()
            .unwrap();
        &self.face_to_node[start..end]
    }

    fn ftag(&self, i: T) -> Tag {
        self.ftags[i.try_into().unwrap()]
    }
}

pub fn merge_polylines<T: Idx>(polylines: &[&[T]]) -> Vec<Vec<T>> {
    let mut res = Vec::new();
    let (polyline, next) = try_merge_polylines(polylines);
    res.push(polyline);
    if !next.is_empty() {
        res.append(&mut merge_polylines(&next));
    }
    res
}

fn try_merge_polylines<'a, T: Idx>(polylines: &[&'a [T]]) -> (Vec<T>, Vec<&'a [T]>) {
    let mut mask = vec![true; polylines.len()];
    let mut connectivity = Vec::with_capacity(polylines.len());

    // add the first edge
    connectivity.extend(polylines[0].iter().copied());
    mask[0] = false;

    loop {
        let mut added_one = false;
        // add at the end?
        let last = *connectivity.last().unwrap();
        for (edg, m) in polylines.iter().zip(mask.iter_mut()) {
            assert!(edg.len() > 1);
            if *m && edg[0] == last {
                *m = false;
                connectivity.extend(edg.iter().copied().skip(1));
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
                tmp.extend(connectivity.iter().copied().skip(1));
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

fn polygon_edges<T: Idx>(face: &[T]) -> impl ExactSizeIterator<Item = [T; 2]> + '_ {
    let n_edgs = face.len();
    (0..n_edgs).map(move |i_edg| {
        if i_edg == n_edgs - 1 {
            [face[n_edgs - 1], face[0]]
        } else {
            [face[i_edg], face[i_edg + 1]]
        }
    })
}

pub fn merge_polygons<T: Idx>(polylines: &[&[T]]) -> Vec<Vec<T>> {
    let mut res = Vec::new();
    let (polyline, next) = try_merge_polygons(polylines);
    res.push(polyline);
    if !next.is_empty() {
        res.append(&mut merge_polygons(&next));
    }
    res
}

fn try_merge_two_polygons<T: Idx>(p0: &[T], p1: &[T]) -> Result<Vec<T>> {
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

fn try_merge_polygons<'a, T: Idx>(polygons: &[&'a [T]]) -> (Vec<T>, Vec<&'a [T]>) {
    let mut mask = vec![true; polygons.len()];
    let mut connectivity = Vec::with_capacity(polygons.len());

    // add the first polygon
    connectivity.extend(polygons[0].iter().copied());
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
        dual::merge_polygons,
        mesh::{Mesh3d, box_mesh},
    };

    use super::{PolyMesh, SimplePolyMesh, merge_polylines};

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
