use crate::{Dim, Result, Tag, TopoTag};
use core::result;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize, ser::SerializeStruct};
use std::fmt;
use std::{collections::HashSet, fs::File, io::Write};
use tmesh::graph::CSRGraph;
use tmesh::mesh::{Mesh, Simplex, get_face_to_elem};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopoNode {
    pub tag: TopoTag,
    pub children: HashSet<Tag>,
    pub parents: HashSet<Tag>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Topology {
    dim: Dim,
    /// List of topology nodes for each dimension (i.e. entities.len() == 4 in dimension 3)
    entities: Vec<Vec<TopoNode>>,
    /// Map a pair of tag to the closest common parent of those tags
    parents: FxHashMap<(TopoTag, TopoTag), TopoTag>,
}

impl Serialize for Topology {
    fn serialize<S>(&self, serializer: S) -> result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Topology", 2)?;
        state.serialize_field("dim", &self.dim)?;
        state.serialize_field("entities", &self.entities)?;
        state.end()
    }
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
        // for ((k0, k1), v) in &self.parents {
        //     writeln!(f, "  {k0:?},{k1:?} -> {v:?}")?;
        // }
        Ok(())
    }
}

impl Topology {
    #[must_use]
    pub fn new(dim: Dim) -> Self {
        Self {
            dim,
            entities: vec![vec![]; (dim + 1) as usize],
            parents: FxHashMap::default(),
        }
    }

    pub fn from_json(fname: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct MiniTopology {
            pub dim: Dim,
            pub entities: Vec<Vec<TopoNode>>,
        }
        let file = File::open(fname)?;
        let tmp: MiniTopology = serde_json::from_reader(file)?;
        let mut res = Self {
            dim: tmp.dim,
            entities: tmp.entities,
            parents: FxHashMap::default(),
        };
        res.compute_parents();
        Ok(res)
    }

    pub fn to_json(&self, fname: &str) -> Result<()> {
        let mut file = File::create(fname)?;
        writeln!(file, "{}", serde_json::to_string_pretty(self).unwrap())?;
        Ok(())
    }

    #[must_use]
    pub fn ntags(&self, dim: Dim) -> usize {
        self.entities[dim as usize].len()
    }

    #[must_use]
    pub fn tags(&self, dim: Dim) -> Vec<Tag> {
        self.entities[dim as usize]
            .iter()
            .map(|x| x.tag.1)
            .collect()
    }

    #[must_use]
    pub fn first_available_tag(&self, dim: Dim) -> Tag {
        self.entities[dim as usize]
            .iter()
            .map(|x| x.tag.1.abs())
            .max()
            .unwrap_or(0)
            + 1
    }

    #[must_use]
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
        if let Some(node) = self.get(tag) {
            let existing_parents = &node.parents;
            let parents = parents.collect::<Vec<_>>();
            panic!(
                "Tag {tag:?} already exists with parents {existing_parents:?}. parents = {parents:?}"
            );
        }

        self.entities[tag.0 as usize].push(TopoNode {
            tag,
            children: HashSet::new(),
            parents: parents.clone().collect(),
        });

        for ptag in parents {
            let e = self.get_mut((tag.0 + 1, ptag));
            assert!(e.is_some(), "TopoTag {:?} not present", (tag.0 + 1, ptag));
            e.unwrap().children.insert(tag.1);
        }
    }

    #[must_use]
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

    /// Return the list containing the tag of `node` and the tags of its descendants (recursively)
    fn children(&self, node: &TopoNode) -> HashSet<TopoTag> {
        let mut res = HashSet::new();
        self.add_children(node, &mut res);
        res
    }

    /// Find out if a node is a child (recursively)
    fn is_child(&self, parent: TopoTag, tag: TopoTag) -> bool {
        if tag.0 >= parent.0 {
            return false;
        } else if tag.0 == parent.0 - 1 {
            let node = self.get(parent).unwrap();
            return node.children.iter().any(|&t| t == tag.1);
        }
        let node = self.get(parent).unwrap();
        let dim = parent.0 - 1;
        node.children
            .iter()
            .any(|&child| self.is_child((dim, child), tag))
    }

    fn compute_parents(&mut self) {
        self.parents.clear();
        for dim in (0..=self.dim).rev() {
            for e in &self.entities[dim as usize] {
                let e_and_children = self.children(e);
                for tag0 in e_and_children.iter().copied() {
                    for tag1 in e_and_children.iter().copied() {
                        self.parents.insert((tag0, tag1), e.tag);
                    }
                }
            }
        }
    }

    #[must_use]
    pub fn parent(&self, topo0: TopoTag, topo1: TopoTag) -> Option<TopoTag> {
        self.parents.get(&(topo0, topo1)).copied()
    }

    fn update_from_elems<C: Simplex>(
        &mut self,
        elems: impl ExactSizeIterator<Item = C> + Clone,
        etags: impl ExactSizeIterator<Item = Tag> + Clone,
        vtags: &mut [TopoTag],
    ) {
        elems.into_iter().zip(etags).for_each(|(e, t)| {
            e.into_iter().for_each(|i| {
                if vtags[i].1 != 0 && vtags[i].1 != t {
                    vtags[i] = (C::DIM as Dim, 0);
                } else {
                    let tag = (C::DIM as Dim, t);
                    if self.get(tag).is_none() {
                        self.insert(tag, &[]);
                    }
                    vtags[i] = tag;
                }
            });
        });
    }

    fn update_from_faces<C: Simplex>(
        &mut self,
        elems: impl ExactSizeIterator<Item = C> + Clone,
        etags: impl ExactSizeIterator<Item = Tag> + Clone,
        faces: impl ExactSizeIterator<Item = C::FACE> + Clone,
        ftags: impl ExactSizeIterator<Item = Tag> + Clone,
        vtags: &mut [TopoTag],
    ) {
        let face2elem = get_face_to_elem(elems);
        let etags = etags.collect::<Vec<_>>();

        faces.into_iter().zip(ftags).for_each(|(f, t)| {
            let tag = (C::FACE::DIM as Dim, t);
            let parents = face2elem
                .get(&f.sorted())
                .unwrap()
                .iter()
                .map(|&i| etags[i])
                .collect::<Vec<_>>();
            assert!(!parents.is_empty());
            if let Some(node) = self.get(tag) {
                assert_eq!(parents.len(), node.parents.len());
                for x in parents {
                    assert!(node.parents.contains(&x));
                }
            } else {
                self.insert(tag, &parents);
            }
            f.into_iter().for_each(|i| {
                if vtags[i].0 == C::FACE::DIM as Dim && (vtags[i].1 == 0 || vtags[i].1 != t) {
                    vtags[i] = (C::FACE::DIM as Dim, 0);
                } else {
                    vtags[i] = (C::FACE::DIM as Dim, t);
                }
            });
        });
    }

    fn max_tag(&self, dim: Dim) -> Tag {
        self.entities[dim as usize]
            .iter()
            .map(|n| n.tag.1.abs())
            .max()
            .unwrap_or(0)
    }

    fn check_untagged<C: Simplex>(
        &self,
        elems: impl ExactSizeIterator<Item = C> + Clone,
        etags: impl ExactSizeIterator<Item = Tag> + Clone,
    ) -> bool {
        let dim = C::FACE::DIM as Dim;
        let edg2face = get_face_to_elem(elems);
        let etags = etags.collect::<Vec<_>>();

        for e2f in edg2face.values() {
            let ftags = e2f.iter().map(|&i| etags[i]).collect::<FxHashSet<_>>();
            let should_be_tagged = e2f.len() != 2 || ftags.len() != 1;
            if should_be_tagged
                && self
                    .get_from_parents_iter(dim, ftags.iter().copied())
                    .is_none()
            {
                return false;
            }
        }
        true
    }

    fn update_and_get_children<C: Simplex>(
        &mut self,
        faces: impl ExactSizeIterator<Item = C> + Clone,
        ftags: impl ExactSizeIterator<Item = Tag> + Clone,
        vtags: &mut [TopoTag],
    ) -> (Vec<C::FACE>, Vec<Tag>, Tag) {
        let dim = C::FACE::DIM as Dim;
        let edg2face = get_face_to_elem(faces);
        let mut next_tag = self.max_tag(dim);
        let mut edgs = Vec::new();
        let mut etags = Vec::new();
        let ftags = ftags.collect::<Vec<_>>();

        for (e, e2f) in &edg2face {
            let ftags = e2f.iter().map(|&i| ftags[i]).collect::<FxHashSet<_>>();
            let should_be_tagged = e2f.len() != 2 || ftags.len() != 1;
            if should_be_tagged {
                #[allow(clippy::option_if_let_else)]
                let t = if let Some(node) = self.get_from_parents_iter(dim, ftags.iter().copied()) {
                    node.tag.1
                } else {
                    next_tag += 1;
                    let flg = if ftags.iter().copied().any(|t| t < 0) {
                        -1
                    } else {
                        1
                    };
                    self.insert_iter((dim, flg * next_tag), ftags.iter().copied());
                    flg * next_tag
                };
                e.into_iter().for_each(|i| {
                    if vtags[i].0 == dim && (vtags[i].1 == 0 || vtags[i].1 != t) {
                        vtags[i] = (dim, 0);
                    } else {
                        vtags[i] = (dim, t);
                    }
                });
                edgs.push(*e);
                etags.push(t);
            }
        }
        (edgs, etags, next_tag)
    }

    fn check_and_fix<C: Simplex>(
        &mut self,
        elems: impl ExactSizeIterator<Item = C> + Clone,
        etags: impl ExactSizeIterator<Item = Tag> + Clone,
        vtags: &mut [TopoTag],
    ) {
        let vert2elems = CSRGraph::transpose(elems, Some(vtags.len()));
        let etags = etags.collect::<Vec<_>>();
        for (i_vert, vtag) in vtags.iter_mut().enumerate() {
            let ids = vert2elems.row(i_vert);
            let mut tags = ids.iter().map(|&i| etags[i]).collect::<FxHashSet<_>>();
            if vtag.0 > C::DIM as Dim {
                assert!(vtag.1 != 0);
            }
            if tags.len() > 1 {
                if vtag.0 == C::DIM as Dim {
                    if vtag.1 != 0 {
                        tags.insert(vtag.1);
                    }
                    let dim = C::FACE::DIM as Dim;
                    if let Some(node) = self.get_from_parents_iter(dim + 1, tags.iter().copied()) {
                        *vtag = node.tag;
                    } else {
                        let tag = self.max_tag(dim) + 1;
                        let flg = if tags.iter().copied().any(|t| t < 0) {
                            -1
                        } else {
                            1
                        };
                        self.insert_iter((dim, flg * tag), tags.iter().copied());
                        *vtag = (dim, flg * tag);
                    }
                }

                if tags
                    .iter()
                    .copied()
                    .any(|t| !self.is_child((C::DIM as Dim, t), *vtag))
                {
                    // assert_eq!(
                    //     vtag.1,
                    //     0,
                    //     "Cannot fix tag {vtag:?}: parents dim = {}, tags = {tags:?}\n{self}",
                    //     C::DIM
                    // );
                    let dim = C::FACE::DIM as Dim;
                    if let Some(node) = self.get_from_parents_iter(dim + 1, tags.iter().copied()) {
                        *vtag = node.tag;
                    } else {
                        let tag = self.max_tag(dim) + 1;
                        let flg = if tags.iter().copied().any(|t| t < 0) {
                            -1
                        } else {
                            1
                        };
                        self.insert_iter((dim, flg * tag), tags.iter().copied());
                        *vtag = (dim, flg * tag);
                    }
                }

                for &t in &tags {
                    assert!(
                        self.is_child((C::DIM as Dim, t), *vtag),
                        "{vtag:?} is not a children of {:?}\n{self}",
                        (C::DIM as Dim, t)
                    );
                }
            }
        }
    }

    pub fn update_from_elems_and_faces<C: Simplex>(
        &mut self,
        elems: impl ExactSizeIterator<Item = C> + Clone,
        etags: impl ExactSizeIterator<Item = Tag> + Clone,
        faces: impl ExactSizeIterator<Item = C::FACE> + Clone,
        ftags: impl ExactSizeIterator<Item = Tag> + Clone,
    ) -> Vec<TopoTag> {
        assert!(
            C::DIM == 2 || C::DIM == 3,
            "Invalid element dimension {}",
            C::DIM
        );
        assert!(!etags.clone().any(|t| t == 0));
        assert!(!ftags.clone().any(|t| t == 0));

        let n_verts = elems.clone().flatten().max().unwrap() + 1;

        let mut vtags = vec![(C::DIM as Dim, 0); n_verts];

        // Tag vertices based on element tags
        // (will not be valid if the vertex belongs to elements with different tags)
        self.update_from_elems(elems.clone(), etags.clone(), &mut vtags);

        // Tag vertices based on face tags
        // (will not be valid if the vertex belongs to faces with different tags)
        self.update_from_faces(
            elems.clone(),
            etags.clone(),
            faces.clone(),
            ftags.clone(),
            &mut vtags,
        );

        // Check that all faces are tagged
        assert!(
            self.check_untagged(elems.clone(), etags.clone()),
            "All the faces were not properly tagged"
        );

        if C::DIM == 3 {
            let (edgs, edgtags, _next_edg_tag) =
                self.update_and_get_children(faces.clone(), ftags.clone(), &mut vtags);

            let (verts, verttags, _next_edg_tag) = self.update_and_get_children(
                edgs.iter().copied(),
                edgtags.iter().copied(),
                &mut vtags,
            );

            // Check
            self.check_and_fix(elems, etags, &mut vtags);
            self.check_and_fix(faces, ftags, &mut vtags);
            self.check_and_fix(edgs.iter().copied(), edgtags.iter().copied(), &mut vtags);
            self.check_and_fix(verts.iter().copied(), verttags.iter().copied(), &mut vtags);
        } else {
            let (verts, verttags, _next_edg_tag) =
                self.update_and_get_children(faces.clone(), ftags.clone(), &mut vtags);

            // Check
            self.check_and_fix(elems, etags, &mut vtags);
            self.check_and_fix(faces, ftags, &mut vtags);
            self.check_and_fix(verts.iter().copied(), verttags.iter().copied(), &mut vtags);
        }

        self.compute_parents();

        vtags
    }

    pub fn clear<F: FnMut(TopoTag) -> bool>(&mut self, mut filter: F) {
        let mut new_entities = vec![Vec::new(); self.dim as usize + 1];
        for dim in 0..=self.dim {
            for node in &self.entities[dim as usize] {
                if !filter(node.tag) {
                    let dim = node.tag.0;
                    let new_node = TopoNode {
                        tag: node.tag,
                        children: node
                            .children
                            .iter()
                            .copied()
                            .filter(|&n| !filter((dim - 1, n)))
                            .collect(),
                        parents: node
                            .parents
                            .iter()
                            .copied()
                            .filter(|&n| !filter((dim + 1, n)))
                            .collect(),
                    };
                    new_entities[dim as usize].push(new_node);
                }
            }
        }
        self.entities = new_entities;
    }
}

pub struct MeshTopology {
    topo: Topology,
    vtags: Vec<TopoTag>,
}

impl MeshTopology {
    pub fn new<const D: usize, M: Mesh<D>>(msh: &M) -> Self {
        let topo = Topology::new(M::C::DIM as Dim);
        Self::new_from(msh, topo)
    }

    pub fn new_from<const D: usize, M: Mesh<D>>(msh: &M, mut topo: Topology) -> Self {
        let vtags =
            topo.update_from_elems_and_faces(msh.elems(), msh.etags(), msh.faces(), msh.ftags());
        Self { topo, vtags }
    }

    #[must_use]
    pub const fn topo(&self) -> &Topology {
        &self.topo
    }

    #[must_use]
    pub fn vtags(&self) -> &[TopoTag] {
        &self.vtags
    }
}

#[cfg(test)]
mod tests {
    use tmesh::{
        Vert2d, Vert3d,
        mesh::{GenericMesh, Mesh, Mesh3d, Tetrahedron, Triangle, box_mesh},
    };

    use crate::{
        Tag,
        mesh::{
            Topology,
            test_meshes::{square_two_tags, square_two_tags_nobdy},
            topology::MeshTopology,
        },
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
    #[should_panic]
    fn test_invalid_2d_1() {
        let mut mesh = square_two_tags();
        mesh.ftags_mut().for_each(|tag| *tag = 1);
        let _ = MeshTopology::new(&mesh);
    }

    #[test]
    #[should_panic]
    fn test_invalid_2d_2() {
        let mesh = square_two_tags_nobdy();
        let _ = MeshTopology::new(&mesh);
    }

    #[test]
    #[should_panic]
    fn test_invalid_2d_3() {
        let mut mesh = square_two_tags();
        mesh.ftags_mut()
            .zip([2, 1, 1, 3])
            .for_each(|(tag, v)| *tag = v);
        let _ = MeshTopology::new(&mesh);
    }

    #[test]
    #[should_panic]
    fn test_invalid_2d_4() {
        let mut mesh = square_two_tags();
        mesh.ftags_mut()
            .zip([1, 1, 2, 2])
            .for_each(|(tag, v)| *tag = v);
        let _ = MeshTopology::new(&mesh);
    }

    #[test]
    fn test_valid_2d_1() {
        let mut mesh = square_two_tags();
        mesh.ftags_mut()
            .zip([1, 1, 2, 2])
            .for_each(|(tag, v)| *tag = v);
        mesh.fix().unwrap();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();
        assert_eq!(topo.entities[0].len(), 1);
        assert_eq!(topo.entities[1].len(), 3);
        assert_eq!(topo.entities[2].len(), 2);
    }

    #[test]
    fn test_valid_2d_1_split() {
        let mut mesh = square_two_tags();
        mesh.ftags_mut()
            .zip([1, 1, 2, 2])
            .for_each(|(tag, v)| *tag = v);
        mesh.fix().unwrap();

        let mesh = mesh.split().split();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 1);
        assert_eq!(topo.entities[1].len(), 3);
        assert_eq!(topo.entities[2].len(), 2);
    }

    #[test]
    fn test_valid_2d_2() {
        let mut mesh = square_two_tags();
        mesh.fix().unwrap();

        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 4);
        assert_eq!(topo.entities[1].len(), 5);
        assert_eq!(topo.entities[2].len(), 2);
    }

    #[test]
    fn test_valid_2d_2_split() {
        let mut mesh = square_two_tags();
        mesh.fix().unwrap();

        let mesh = mesh.split().split();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 4);
        assert_eq!(topo.entities[1].len(), 5);
        assert_eq!(topo.entities[2].len(), 2);
    }

    #[test]
    fn test_valid_3d() {
        let mesh: Mesh3d = box_mesh(1.0, 2, 1.0, 2, 1.0, 2);
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 8);
        assert_eq!(topo.entities[1].len(), 12);
        assert_eq!(topo.entities[2].len(), 6);
        assert_eq!(topo.entities[3].len(), 1);
    }

    #[test]
    fn test_corner_2d_1() {
        let verts = vec![
            Vert2d::new(0.0, 0.0),
            Vert2d::new(1.0, 0.0),
            Vert2d::new(2.0, 0.0),
            Vert2d::new(0.0, 1.0),
            Vert2d::new(2.0, 1.0),
        ];
        let elems = vec![Triangle::<u32>::new(0, 1, 3), Triangle::new(1, 2, 4)];
        let etags = vec![1, 1];
        let faces = vec![];
        let ftags = vec![];

        let mut mesh = GenericMesh::from_vecs(verts, elems, etags, faces, ftags);
        mesh.fix().unwrap();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 1);
        assert_eq!(topo.entities[1].len(), 1);
        assert_eq!(topo.entities[2].len(), 1);
    }

    #[test]
    fn test_corner_2d_2() {
        let verts = vec![
            Vert2d::new(0.0, 0.0),
            Vert2d::new(1.0, 0.0),
            Vert2d::new(2.0, 0.0),
            Vert2d::new(0.0, 1.0),
            Vert2d::new(2.0, 1.0),
        ];
        let elems = vec![Triangle::<u32>::new(0, 1, 3), Triangle::new(1, 2, 4)];
        let etags = vec![1, 2];
        let faces = vec![];
        let ftags = vec![];

        let mut mesh = GenericMesh::from_vecs(verts, elems, etags, faces, ftags);
        mesh.fix().unwrap();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 1);
        assert_eq!(topo.entities[1].len(), 2);
        assert_eq!(topo.entities[2].len(), 2);
    }

    #[test]
    fn test_corner_2d_2_split() {
        let verts = vec![
            Vert2d::new(0.0, 0.0),
            Vert2d::new(1.0, 0.0),
            Vert2d::new(2.0, 0.0),
            Vert2d::new(0.0, 1.0),
            Vert2d::new(2.0, 1.0),
        ];
        let elems = vec![Triangle::<u32>::new(0, 1, 3), Triangle::new(1, 2, 4)];
        let etags = vec![1, 2];
        let faces = vec![];
        let ftags = vec![];

        let mut mesh = GenericMesh::from_vecs(verts, elems, etags, faces, ftags)
            .split()
            .split();
        mesh.fix().unwrap();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 1);
        assert_eq!(topo.entities[1].len(), 2);
        assert_eq!(topo.entities[2].len(), 2);
    }

    #[test]
    fn test_corner_3d_1() {
        let verts = vec![
            Vert3d::new(0.0, 0.0, 0.0),
            Vert3d::new(1.0, 0.0, 0.0),
            Vert3d::new(0.0, 1.0, 0.0),
            Vert3d::new(0.0, 0.0, 1.0),
            Vert3d::new(1.0, 0.0, 1.0),
            Vert3d::new(0.0, 1.0, 1.0),
            Vert3d::new(0.0, 0.0, 0.5),
        ];
        let elems = vec![
            Tetrahedron::<u32>::new(0, 1, 2, 6),
            Tetrahedron::new(3, 5, 4, 6),
        ];
        let etags = vec![1, 2];
        let faces = vec![];
        let ftags = vec![];

        let mut mesh = GenericMesh::from_vecs(verts, elems, etags, faces, ftags);
        mesh.fix().unwrap();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 0);
        assert_eq!(topo.entities[1].len(), 1);
        assert_eq!(topo.entities[2].len(), 3);
        assert_eq!(topo.entities[3].len(), 2);
    }

    #[test]
    fn test_corner_3d_1_split() {
        let verts = vec![
            Vert3d::new(0.0, 0.0, 0.0),
            Vert3d::new(1.0, 0.0, 0.0),
            Vert3d::new(0.0, 1.0, 0.0),
            Vert3d::new(0.0, 0.0, 1.0),
            Vert3d::new(1.0, 0.0, 1.0),
            Vert3d::new(0.0, 1.0, 1.0),
            Vert3d::new(0.0, 0.0, 0.5),
        ];
        let elems = vec![
            Tetrahedron::<u32>::new(0, 1, 2, 6),
            Tetrahedron::new(3, 5, 4, 6),
        ];
        let etags = vec![1, 2];
        let faces = vec![];
        let ftags = vec![];

        let mut mesh = GenericMesh::from_vecs(verts, elems, etags, faces, ftags);
        mesh.fix().unwrap();

        let mesh = mesh.split().split();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 0);
        assert_eq!(topo.entities[1].len(), 1);
        assert_eq!(topo.entities[2].len(), 3);
        assert_eq!(topo.entities[3].len(), 2);
    }

    #[test]
    fn test_corner_3d_2() {
        let verts = vec![
            Vert3d::new(0.0, 0.0, 0.0),
            Vert3d::new(1.0, 0.0, 0.0),
            Vert3d::new(0.0, 1.0, 0.0),
            Vert3d::new(0.0, 0.0, 1.0),
            Vert3d::new(1.0, 0.0, 1.0),
            Vert3d::new(0.0, 1.0, 1.0),
            Vert3d::new(0.0, 0.0, 0.5),
        ];
        let elems = vec![
            Tetrahedron::<u32>::new(0, 1, 2, 6),
            Tetrahedron::new(3, 5, 4, 6),
        ];
        let etags = vec![1, 2];
        let faces = vec![];
        let ftags = vec![];

        let mut mesh = GenericMesh::from_vecs(verts, elems, etags, faces, ftags);
        mesh.fix().unwrap();
        mesh.etags_mut().for_each(|t| *t = 1);
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 0);
        assert_eq!(topo.entities[1].len(), 1);
        assert_eq!(topo.entities[2].len(), 2);
        assert_eq!(topo.entities[3].len(), 1);
    }

    #[test]
    fn test_corner_3d_2_split() {
        let verts = vec![
            Vert3d::new(0.0, 0.0, 0.0),
            Vert3d::new(1.0, 0.0, 0.0),
            Vert3d::new(0.0, 1.0, 0.0),
            Vert3d::new(0.0, 0.0, 1.0),
            Vert3d::new(1.0, 0.0, 1.0),
            Vert3d::new(0.0, 1.0, 1.0),
            Vert3d::new(0.0, 0.0, 0.5),
        ];
        let elems = vec![
            Tetrahedron::<u32>::new(0, 1, 2, 6),
            Tetrahedron::new(3, 5, 4, 6),
        ];
        let etags = vec![1, 2];
        let faces = vec![];
        let ftags = vec![];

        let mut mesh = GenericMesh::from_vecs(verts, elems, etags, faces, ftags);
        mesh.fix().unwrap();
        mesh.etags_mut().for_each(|t| *t = 1);

        let mesh = mesh.split().split();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 0);
        assert_eq!(topo.entities[1].len(), 1);
        assert_eq!(topo.entities[2].len(), 2);
        assert_eq!(topo.entities[3].len(), 1);
    }
}
