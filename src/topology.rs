use crate::{
    mesh::SimplexMesh,
    topo_elems::{get_face_to_elem, Elem},
    Dim, Idx, Tag, TopoTag,
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

#[derive(Debug, Clone)]
pub struct Topology {
    dim: Dim,
    /// List of topology nodes for each dimension (i.e. entities.len() == 4 in dimension 3)
    entities: Vec<Vec<TopoNode>>,
    /// Map a pair of tag to the closest common parent of those tags
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

    /// Return the list containing the tag of `node` and the tags of its descendants
    fn children(&self, node: &TopoNode) -> HashSet<TopoTag> {
        let mut res = HashSet::new();
        self.add_children(node, &mut res);
        res
    }

    fn compute_parents(&mut self) {
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

    /// Compute the tag of an element from the `tags` of its vertices
    pub fn elem_tag<I: Iterator<Item = TopoTag>>(&self, mut tags: I) -> Option<TopoTag> {
        // Look for the closest common parent of all vertices tags
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

    fn get_faces_and_tags<
        E: Elem,
        I1: ExactSizeIterator<Item = E>,
        I2: ExactSizeIterator<Item = Tag>,
        I3: ExactSizeIterator<Item = E::Face>,
        I4: ExactSizeIterator<Item = Tag>,
    >(
        elems: I1,
        etags: I2,
        faces: I3,
        ftags: I4,
        topo: &mut Self,
    ) -> (Vec<E::Face>, Vec<Tag>) {
        // Input faces
        let n_faces = faces.len() as Idx;
        let mut tagged_faces = FxHashMap::with_hasher(BuildHasherDefault::default());
        let mut next_tag: Tag = 1;
        for (mut face, ftag) in faces.zip(ftags) {
            face.sort();
            tagged_faces.insert(face, ftag);
            next_tag = max(next_tag, ftag + 1);
        }

        let face2elem = get_face_to_elem(elems);
        let mut next_elems = Vec::with_capacity(n_faces as usize);
        let mut next_tags = Vec::with_capacity(n_faces as usize);

        let etags: Vec<_> = etags.collect();

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
                    if let Some(e) = e {
                        assert_eq!(
                            e.parents.len(),
                            1,
                            "a boundary face belongs to elements with multiple tags"
                        );
                        let p = e.parents.iter().next().unwrap();
                        assert_eq!(
                            elem_tag, *p,
                            "a boundary face belongs to elements with multiple tags"
                        );
                    } else {
                        topo.insert((E::Face::DIM as Dim, face_tag), &[elem_tag]);
                    }
                } else {
                    assert!(n_faces == 0, "not implemented");
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
                            if elem_tags.iter().any(|&t| t < 0) {
                                face_tag = -face_tag;
                            }
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
                next_elems.push(face);
                next_tags.push(face_tag);
            }
        }
        (next_elems, next_tags)
    }

    fn elem_tags_to_vert_tags<
        E: Elem,
        I1: ExactSizeIterator<Item = E>,
        I2: ExactSizeIterator<Item = Tag>,
    >(
        elems: I1,
        etags: I2,
        topo: &mut Self,
        ntags: &mut [TopoTag],
    ) {
        for (elem, etag) in elems.zip(etags) {
            let e = topo.get((E::DIM as Dim, etag));
            if e.is_none() {
                topo.insert((E::DIM as Dim, etag), &[]);
            }
            for idx in elem {
                ntags[idx as usize] = topo.get((E::DIM as Dim, etag)).unwrap().tag;
            }
        }
    }

    pub fn from_mesh<const D: usize, E: Elem>(mesh: &SimplexMesh<D, E>) -> (Self, Vec<TopoTag>) {
        info!("Building topology from mesh");

        let mut topo = Self::new(E::N_VERTS as Dim - 1);
        let n_verts = mesh.n_verts();
        let mut vtags: Vec<TopoTag> = vec![(0, 0); n_verts as usize];

        Self::elem_tags_to_vert_tags(mesh.elems(), mesh.etags(), &mut topo, &mut vtags);
        if E::N_VERTS == 4 {
            let (next_elems, next_tags) = Self::get_faces_and_tags(
                mesh.elems(),
                mesh.etags(),
                mesh.faces(),
                mesh.ftags(),
                &mut topo,
            );
            Self::elem_tags_to_vert_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                &mut topo,
                &mut vtags,
            );

            let (next_elems, next_tags) = Self::get_faces_and_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                std::iter::empty(),
                std::iter::empty(),
                &mut topo,
            );
            Self::elem_tags_to_vert_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                &mut topo,
                &mut vtags,
            );

            let (next_elems, next_tags) = Self::get_faces_and_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                std::iter::empty(),
                std::iter::empty(),
                &mut topo,
            );
            Self::elem_tags_to_vert_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                &mut topo,
                &mut vtags,
            );

            Self::get_faces_and_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                std::iter::empty(),
                std::iter::empty(),
                &mut topo,
            );
            Self::elem_tags_to_vert_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                &mut topo,
                &mut vtags,
            );
        } else if E::N_VERTS == 3 {
            let (next_elems, next_tags) = Self::get_faces_and_tags(
                mesh.elems(),
                mesh.etags(),
                mesh.faces(),
                mesh.ftags(),
                &mut topo,
            );
            Self::elem_tags_to_vert_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                &mut topo,
                &mut vtags,
            );

            let (next_elems, next_tags) = Self::get_faces_and_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                std::iter::empty(),
                std::iter::empty(),
                &mut topo,
            );
            Self::elem_tags_to_vert_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                &mut topo,
                &mut vtags,
            );

            Self::get_faces_and_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                std::iter::empty(),
                std::iter::empty(),
                &mut topo,
            );
            Self::elem_tags_to_vert_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                &mut topo,
                &mut vtags,
            );
        } else if E::N_VERTS == 2 {
            let (next_elems, next_tags) = Self::get_faces_and_tags(
                mesh.elems(),
                mesh.etags(),
                mesh.faces(),
                mesh.ftags(),
                &mut topo,
            );
            Self::elem_tags_to_vert_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                &mut topo,
                &mut vtags,
            );

            Self::get_faces_and_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                std::iter::empty(),
                std::iter::empty(),
                &mut topo,
            );
            Self::elem_tags_to_vert_tags(
                next_elems.iter().copied(),
                next_tags.iter().copied(),
                &mut topo,
                &mut vtags,
            );
        } else {
            unreachable!();
        }

        topo.compute_parents();

        (topo, vtags)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        test_meshes::{test_mesh_2d, test_mesh_2d_nobdy, test_mesh_3d},
        topo_elems::{Edge, Triangle},
        topology::{Dim, Tag, Topology},
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

        let mut tags = std::iter::once(&(2 as Dim, 1 as Tag)).copied();
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
        let elems = [Triangle::new(0, 1, 2), Triangle::new(0, 2, 3)];
        let etags = [1, 2];
        let faces = Vec::new();
        let ftags = Vec::new();

        let mut topo = Topology::new(2);
        topo.insert((2, 1), &[]);
        topo.insert((2, 2), &[]);
        let (_elems, _etags) = Topology::get_faces_and_tags(
            elems.iter().copied(),
            etags.iter().copied(),
            faces.iter().copied(),
            ftags.iter().copied(),
            &mut topo,
        );

        assert_eq!(topo.entities[0].len(), 0);
        assert_eq!(topo.entities[1].len(), 3);
        assert_eq!(topo.entities[2].len(), 2);
    }

    #[test]
    fn test_get_faces_and_tags_2d() {
        let elems = [Triangle::new(0, 1, 2), Triangle::new(0, 2, 3)];
        let etags = [1, 2];
        let faces = [
            Edge::new(0, 1),
            Edge::new(1, 2),
            Edge::new(2, 3),
            Edge::new(3, 0),
        ];
        let ftags = [1, 2, 3, 4];

        let mut topo = Topology::new(2);
        topo.insert((2, 1), &[]);
        topo.insert((2, 2), &[]);
        let (_elems, _etags) = Topology::get_faces_and_tags(
            elems.iter().copied(),
            etags.iter().copied(),
            faces.iter().copied(),
            ftags.iter().copied(),
            &mut topo,
        );

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

    #[test]
    #[should_panic]
    fn test_invalid_tags() {
        let mut mesh = test_mesh_2d();
        mesh.mut_ftags().for_each(|tag| *tag = 1);
        mesh.compute_topology();
    }

    #[test]
    #[should_panic]
    fn test_invalid_tags_2() {
        let mut mesh = test_mesh_2d();
        mesh.mut_ftags()
            .zip([2, 1, 1, 3])
            .for_each(|(tag, v)| *tag = v);
        mesh.compute_topology();
    }

    #[test]
    fn test_valid_tags() {
        let mut mesh = test_mesh_2d();
        mesh.mut_ftags()
            .zip([1, 1, 2, 2])
            .for_each(|(tag, v)| *tag = v);
        mesh.compute_topology();
    }
}
