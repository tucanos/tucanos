use crate::{
    mesh::SimplexMesh,
    topo_elems::get_elem,
    topo_elems::{get_face_to_elem, Edge, Elem, Tetrahedron, Triangle, Vertex},
    Dim, Idx, Mesh, Tag, TopoTag,
};
use core::cmp::max;
use log::info;
use rustc_hash::FxHashMap;
use std::collections::HashSet;
use std::fmt;
use std::hash::BuildHasherDefault;

#[derive(Debug, Clone)]
pub struct TopoNode {
    pub tag: TopoTag,
    pub children: HashSet<Tag>,
    pub parents: HashSet<Tag>,
}

#[derive(Debug)]
pub struct Topology {
    dim: Dim,
    entities: Vec<Vec<TopoNode>>,
    parents: FxHashMap<(TopoTag, TopoTag), TopoTag>,
}

impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Topology: dim = {}", self.dim)?;
        for dim in 0..=self.dim {
            writeln!(f, "  dim = {dim}:")?;
            for e in &self.entities[dim as usize] {
                writeln!(f, "    {:?}", *e)?;
            }
        }
        for ((k0, k1), v) in &self.parents {
            writeln!(f, "  {k0:?},{k1:?} -> {v:?}")?;
        }
        Ok(())
    }
}

impl Topology {
    pub fn new(dim: Dim) -> Self {
        Self {
            dim,
            entities: vec![vec![]; (dim + 1) as usize],
            parents: FxHashMap::default(),
        }
    }
    pub fn ntags(&self, dim: Dim) -> usize {
        self.entities[dim as usize].len()
    }
    pub fn tags(&self, dim: Dim) -> Vec<Tag> {
        self.entities[dim as usize]
            .iter()
            .map(|x| x.tag.1)
            .collect()
    }
    pub fn get(&self, tag: TopoTag) -> Option<&TopoNode> {
        assert!(tag.0 >= 0, "Invalid dimension");
        assert!(tag.0 <= self.dim, "Invalid dimension");
        self.entities[tag.0 as usize]
            .iter()
            .find(|&e| e.tag.1 == tag.1)
    }
    fn get_mut(&mut self, tag: TopoTag) -> Option<&mut TopoNode> {
        assert!(tag.0 >= 0, "Invalid dimension");
        assert!(tag.0 <= self.dim + 1, "Invalid dimension");
        self.entities[tag.0 as usize]
            .iter_mut()
            .find(|e| e.tag.1 == tag.1)
    }
    pub fn insert(&mut self, tag: TopoTag, parents: &[Tag]) {
        self.insert_iter(tag, parents.iter().copied());
    }

    pub fn insert_iter<I>(&mut self, tag: TopoTag, parents: I)
    where
        I: Iterator<Item = Tag> + Clone,
    {
        assert!(self.get(tag).is_none(), "Tag {tag:?} already exists");

        self.entities[tag.0 as usize].push(TopoNode {
            tag,
            children: HashSet::new(),
            parents: parents.clone().collect(),
        });

        for ptag in parents {
            let e = self.get_mut((tag.0 + 1, ptag));
            assert!(e.is_some());
            e.unwrap().children.insert(tag.1);
        }
    }

    pub fn get_from_parents(&self, dim: Dim, parents: &[Tag]) -> Option<&TopoNode> {
        self.get_from_parents_iter(dim, parents.iter().copied())
    }

    pub fn get_from_parents_iter<I>(&self, dim: Dim, parents: I) -> Option<&TopoNode>
    where
        I: Iterator<Item = Tag> + Clone,
    {
        let parents_hs: HashSet<Tag> = parents.collect();
        self.entities[dim as usize]
            .iter()
            .find(|&e| e.parents == parents_hs)
    }

    fn add_children(&self, e: &TopoNode, res: &mut HashSet<TopoTag>) {
        res.insert(e.tag);
        for c in &e.children {
            self.add_children(self.get((e.tag.0 - 1, *c)).unwrap(), res);
        }
    }
    fn children(&self, e: &TopoNode) -> HashSet<TopoTag> {
        let mut res = HashSet::new();
        self.add_children(e, &mut res);
        res
    }

    pub fn compute_parents(&mut self) {
        for dim in (0..=self.dim).rev() {
            for e in &self.entities[dim as usize] {
                let children = self.children(e);
                for tag0 in children.iter().copied() {
                    for tag1 in children.iter().copied() {
                        self.parents.insert((tag0, tag1), e.tag);
                    }
                }
            }
        }
    }

    pub fn parent(&self, topo0: TopoTag, topo1: TopoTag) -> Option<TopoTag> {
        self.parents.get(&(topo0, topo1)).copied()
    }

    pub fn elem_tag<I: Iterator<Item = TopoTag>>(&self, mut tags: I) -> Option<TopoTag> {
        let first = tags.next()?;
        let second = tags.next();
        if second.is_none() {
            return Some(first);
        }
        let mut tag = self.parents.get(&(first, second.unwrap()))?;
        for other in tags {
            tag = self.parents.get(&(*tag, other))?;
        }
        Some(*tag)
    }

    fn get_faces_and_tags<E: Elem>(
        elems: &[Idx],
        etags: &[Tag],
        faces: &[Idx],
        ftags: &[Tag],
        topo: &mut Topology,
    ) -> (Vec<Idx>, Vec<Tag>) {
        // Input faces
        let n_faces = faces.len() as Idx / E::Face::N_VERTS;
        let mut tagged_faces = FxHashMap::with_hasher(BuildHasherDefault::default());
        let mut next_tag: Tag = 1;
        for (i_face, ftag) in ftags.iter().enumerate() {
            let mut face = get_elem::<E::Face>(faces, i_face as Idx);
            face.sort();
            tagged_faces.insert(face, *ftag);
            next_tag = max(next_tag, *ftag + 1);
        }

        let face2elem = get_face_to_elem::<E>(elems);
        let mut next_elems = Vec::new();
        next_elems.reserve((E::Face::N_VERTS * n_faces) as usize);
        let mut next_tags = Vec::new();
        next_tags.reserve(n_faces as usize);

        for (face, f2e) in face2elem {
            let mut face_tag: Tag = 0;
            let mut add = false;
            if f2e.len() == 1 {
                //boundary face
                let i_elem = f2e[0];
                let elem_tag = etags[i_elem as usize];
                let tagged_face = tagged_faces.get(&face);
                if let Some(tagged_face) = tagged_face {
                    face_tag = *tagged_face;
                    let e = topo.get((E::Face::DIM as Dim, face_tag));
                    if e.is_none() {
                        topo.insert((E::Face::DIM as Dim, face_tag), &[elem_tag]);
                    }
                } else {
                    assert!(faces.is_empty(), "not implemented");
                    let e = topo.get_from_parents(E::Face::DIM as Dim, &[elem_tag]);
                    if let Some(e) = e {
                        face_tag = e.tag.1;
                    } else {
                        face_tag = next_tag;
                        next_tag += 1;
                        topo.insert((E::Face::DIM as Dim, face_tag), &[elem_tag]);
                    }
                }
                add = true;
            } else {
                // internal  or multiple face
                let mut elem_tags: HashSet<Tag> = HashSet::new();
                for i_elem in f2e {
                    elem_tags.insert(etags[i_elem as usize]);
                }
                if elem_tags.len() > 1 {
                    let e =
                        topo.get_from_parents_iter(E::Face::DIM as Dim, elem_tags.iter().copied());
                    if let Some(e) = e {
                        face_tag = e.tag.1;
                    } else {
                        let tagged_face = tagged_faces.get(&face);
                        if let Some(tagged_face) = tagged_face {
                            face_tag = *tagged_face;
                        } else {
                            face_tag = next_tag;
                            next_tag += 1;
                        }
                        topo.insert_iter(
                            (E::Face::DIM as Dim, face_tag),
                            elem_tags.iter().copied(),
                        );
                    }
                    add = true;
                }
            }
            if add {
                for idx in face {
                    next_elems.push(idx);
                }
                next_tags.push(face_tag);
            }
        }
        (next_elems, next_tags)
    }

    fn from_elems_and_faces<E: Elem>(
        elems: &[Idx],
        etags: &[Tag],
        faces: &[Idx],
        ftags: &[Tag],
        topo: &mut Topology,
        ntags: &mut [TopoTag],
    ) -> (Vec<Idx>, Vec<Tag>) {
        for (i_elem, elem_tag) in etags.iter().enumerate() {
            let e = topo.get((E::DIM as Dim, *elem_tag));
            if e.is_none() {
                topo.insert((E::DIM as Dim, *elem_tag), &[]);
            }
            let elem = get_elem::<E>(elems, i_elem as Idx);
            for idx in elem {
                ntags[idx as usize] = topo.get((E::DIM as Dim, *elem_tag)).unwrap().tag;
            }
        }
        if E::DIM > 0 {
            Self::get_faces_and_tags::<E>(elems, etags, faces, ftags, topo)
        } else {
            (vec![], vec![])
        }
    }

    pub fn from_mesh<const D: usize, E: Elem>(
        mesh: &SimplexMesh<D, E>,
    ) -> (Topology, Vec<TopoTag>) {
        info!("Building topology from mesh");

        let mut topo = Topology::new(E::N_VERTS as Dim - 1);
        let n_verts = mesh.n_verts();
        let mut vtags: Vec<TopoTag> = vec![(0, 0); n_verts as usize];

        let mut next_elems;
        let mut next_tags;
        if E::N_VERTS == 4 {
            (next_elems, next_tags) = Self::from_elems_and_faces::<Tetrahedron>(
                &mesh.elems,
                &mesh.etags,
                &mesh.faces,
                &mesh.ftags,
                &mut topo,
                &mut vtags,
            );
            (next_elems, next_tags) = Self::from_elems_and_faces::<Triangle>(
                &next_elems,
                &next_tags,
                &[0 as Idx; 0],
                &[0 as Tag; 0],
                &mut topo,
                &mut vtags,
            );
        } else {
            (next_elems, next_tags) = Self::from_elems_and_faces::<Triangle>(
                &mesh.elems,
                &mesh.etags,
                &mesh.faces,
                &mesh.ftags,
                &mut topo,
                &mut vtags,
            );
        }

        let (next_elems, next_tags) = Self::from_elems_and_faces::<Edge>(
            &next_elems,
            &next_tags,
            &[0 as Idx; 0],
            &[0 as Tag; 0],
            &mut topo,
            &mut vtags,
        );

        Self::from_elems_and_faces::<Vertex>(
            &next_elems,
            &next_tags,
            &[0 as Idx; 0],
            &[0 as Tag; 0],
            &mut topo,
            &mut vtags,
        );

        topo.compute_parents();

        (topo, vtags)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        test_meshes::{test_mesh_2d, test_mesh_2d_nobdy, test_mesh_3d},
        topo_elems::Triangle,
        topology::{Dim, Tag, Topology},
        Idx,
    };

    #[test]
    fn test_insert() {
        let mut t = Topology::new(3);
        t.insert((3, 1), &[]);
        t.insert((2, 1), &[1]);
        t.insert((2, 2), &[1]);
        t.insert((2, 3), &[1]);
        t.insert((2, 4), &[1]);
        t.insert((1, 1), &[1, 2]);
        t.insert((1, 2), &[1, 3]);
        t.insert((1, 3), &[3, 2]);

        let p = vec![1 as Tag, 2 as Tag];
        let c = t.get_from_parents(1, &p);
        assert!(c.is_some());
        assert_eq!(c.unwrap().tag.1, 1);

        let p = vec![1 as Tag, 3 as Tag];
        let c = t.get_from_parents(1, &p);
        assert!(c.is_some());
        assert_eq!(c.unwrap().tag.1, 2);

        let p = vec![2 as Tag, 3 as Tag];
        let c = t.get_from_parents(1, &p);
        assert!(c.is_some());
        assert_eq!(c.unwrap().tag.1, 3);
    }

    #[test]
    fn test_2d() {
        let mut t = Topology::new(3);
        t.insert((2, 1), &[]);
        t.insert((2, 2), &[]);
        t.insert((1, 1), &[1]);
        t.insert((1, 2), &[1]);
        t.insert((1, 3), &[2]);
        t.insert((1, 4), &[2]);
        t.insert((1, 5), &[1, 2]);
        t.insert((0, 1), &[1, 4, 5]);
        t.insert((0, 2), &[1, 2]);
        t.insert((0, 3), &[2, 3, 5]);
        t.insert((0, 4), &[3, 4]);

        t.compute_parents();

        let p = t.parents.get(&((0, 1), (0, 2)));
        assert!(p.is_some());
        assert_eq!(p.unwrap().0, 1);
        assert_eq!(p.unwrap().1, 1);

        let p = t.parents.get(&((0, 1), (1, 2)));
        assert!(p.is_some());
        assert_eq!(p.unwrap().0, 2);
        assert_eq!(p.unwrap().1, 1);

        let p = t.parents.get(&((0, 1), (1, 3)));
        assert!(p.is_some());
        assert_eq!(p.unwrap().0, 2);
        assert_eq!(p.unwrap().1, 2);

        let p = t.parents.get(&((0, 4), (1, 5)));
        assert!(p.is_some());
        assert_eq!(p.unwrap().0, 2);
        assert_eq!(p.unwrap().1, 2);

        let p = t.parents.get(&((0, 2), (1, 3)));
        assert!(p.is_none());

        let mut tags = [(2 as Dim, 1 as Tag)].iter().copied();
        let p = t.elem_tag(&mut tags);
        assert!(p.is_some());
        assert_eq!(p.unwrap().0, 2);
        assert_eq!(p.unwrap().1, 1);

        let mut tags = [(2 as Dim, 1 as Tag), (2 as Dim, 2 as Tag)].iter().copied();
        let p = t.elem_tag(&mut tags);
        assert!(p.is_none());

        let mut tags = [(1 as Dim, 1 as Tag), (1 as Dim, 2 as Tag)].iter().copied();
        let p = t.elem_tag(&mut tags);
        assert!(p.is_some());
        assert_eq!(p.unwrap().0, 2);
        assert_eq!(p.unwrap().1, 1);

        let mut tags = [(1 as Dim, 1 as Tag), (1 as Dim, 5 as Tag)].iter().copied();
        let p = t.elem_tag(&mut tags);
        assert!(p.is_some());
        assert_eq!(p.unwrap().0, 2);
        assert_eq!(p.unwrap().1, 1);

        let mut tags = [(0 as Dim, 1 as Tag), (0 as Dim, 3 as Tag)].iter().copied();
        let p = t.elem_tag(&mut tags);
        assert!(p.is_some());
        assert_eq!(p.unwrap().0, 1);
        assert_eq!(p.unwrap().1, 5);

        let mut tags = [
            (0 as Dim, 1 as Tag),
            (0 as Dim, 2 as Tag),
            (0 as Dim, 3 as Tag),
        ]
        .iter()
        .copied();
        let p = t.elem_tag(&mut tags);
        assert!(p.is_some());
        assert_eq!(p.unwrap().0, 2);
        assert_eq!(p.unwrap().1, 1);
    }

    #[test]
    fn test_get_faces_and_tags_2d_nobdy() {
        let elems: Vec<Idx> = vec![0, 1, 2, 0, 2, 3];
        let etags: Vec<Tag> = vec![1, 2];
        let faces: Vec<Idx> = vec![];
        let ftags: Vec<Tag> = vec![];

        let mut topo = Topology::new(2);
        topo.insert((2, 1), &[]);
        topo.insert((2, 2), &[]);
        let (_elems, _etags) =
            Topology::get_faces_and_tags::<Triangle>(&elems, &etags, &faces, &ftags, &mut topo);

        assert_eq!(topo.entities[0].len(), 0);
        assert_eq!(topo.entities[1].len(), 3);
        assert_eq!(topo.entities[2].len(), 2);
    }

    #[test]
    fn test_get_faces_and_tags_2d() {
        let elems: Vec<Idx> = vec![0, 1, 2, 0, 2, 3];
        let etags: Vec<Tag> = vec![1, 2];
        let faces: Vec<Idx> = vec![0, 1, 1, 2, 2, 3, 3, 0];
        let ftags: Vec<Tag> = vec![1, 2, 3, 4];

        let mut topo = Topology::new(2);
        topo.insert((2, 1), &[]);
        topo.insert((2, 2), &[]);
        let (_elems, _etags) =
            Topology::get_faces_and_tags::<Triangle>(&elems, &etags, &faces, &ftags, &mut topo);

        assert_eq!(topo.entities[0].len(), 0);
        assert_eq!(topo.entities[1].len(), 5);
        assert_eq!(topo.entities[2].len(), 2);
    }

    #[test]
    fn test_from_elems_and_faces_2d_nobdy() {
        let mesh = test_mesh_2d_nobdy();
        let (topo, vtags) = Topology::from_mesh(&mesh);

        assert_eq!(topo.entities[0].len(), 1);
        assert_eq!(topo.entities[1].len(), 3);
        assert_eq!(topo.entities[2].len(), 2);
        assert_eq!(vtags[0].0, 0);
        assert_eq!(vtags[1].0, 1);
        assert_eq!(vtags[2].0, 0);
        assert_eq!(vtags[3].0, 1);
    }

    #[test]
    fn test_from_elems_and_faces_2d() {
        let mesh = test_mesh_2d();

        let (topo, vtags) = Topology::from_mesh(&mesh);

        assert_eq!(topo.entities[0].len(), 4);
        assert_eq!(topo.entities[1].len(), 5);
        assert_eq!(topo.entities[2].len(), 2);
        assert_eq!(vtags[0], topo.get_from_parents(0, &[1, 4, 5]).unwrap().tag);
        assert_eq!(vtags[1], topo.get_from_parents(0, &[1, 2]).unwrap().tag);
        assert_eq!(vtags[2], topo.get_from_parents(0, &[2, 3, 5]).unwrap().tag);
        assert_eq!(vtags[3], topo.get_from_parents(0, &[3, 4]).unwrap().tag);
    }

    #[test]
    #[should_panic]
    fn test_insert_double() {
        let mut t = Topology::new(3);
        t.insert((3, 1), &[]);
        t.insert((3, 1), &[]);
    }

    #[test]
    #[should_panic]
    fn test_insert_invalid_parent() {
        let mut t = Topology::new(3);
        t.insert((2, 1), &[1]);
    }

    #[test]
    fn test_from_elems_and_faces_3d() {
        let mesh = test_mesh_3d();

        let (topo, _vtags) = Topology::from_mesh(&mesh);

        assert_eq!(topo.entities[0].len(), 8);
        assert_eq!(topo.entities[1].len(), 12);
        assert_eq!(topo.entities[2].len(), 6);
        assert_eq!(topo.entities[3].len(), 1);

        // assert_eq!(
        //     vtags[0],
        //     topo.get_from_parents(0, vec![1, 4, 5]).unwrap().tag
        // );
        // assert_eq!(vtags[1], topo.get_from_parents(0, vec![1, 2]).unwrap().tag);
        // assert_eq!(
        //     vtags[2],
        //     topo.get_from_parents(0, vec![2, 3, 5]).unwrap().tag
        // );
        // assert_eq!(vtags[3], topo.get_from_parents(0, vec![3, 4]).unwrap().tag);
    }
}
