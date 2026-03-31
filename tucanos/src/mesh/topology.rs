/// Topology management for meshes.
///
/// The topology represents the hierarchy of element tags in a mesh.
/// - All mesh elements (dimension d) must have a tag
/// - Elements of dimension d' < d are tagged iif
///     - they belong to only one tagged parent element
///     - they belong to 2 tagged parents with different tags
///     - they belong to 3 or more tagged parents
///
/// This is consistent with the mesh tags for elements and faces, but also introduces edge tags
/// (for tetrahedron meshes) and node tags.
///
/// For meshes containing triangles (as faces or elements) in 3D, surfaces may be in contact (locally)
/// at a vertex without sharing an edge. In order to represent this, we introduct fictitious tags to represent
/// the element / face and edge tags at the vertex.
///
/// Vertices are then tagged with tuples (dim, tag) that represent the dimension of the lowest entity that
/// includes the vertex and the tag of that entity.
use crate::{Dim, Error, Result, Tag, TopoTag};
use core::result;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize, ser::SerializeStruct};
use std::fmt;
use std::{collections::HashSet, fs::File, io::Write};
use tmesh::graph::CSRGraph;
use tmesh::mesh::{Mesh, Simplex, get_face_to_elem};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopoNode {
    pub tag: TopoTag,
    pub children: FxHashSet<Tag>,
    pub parents: FxHashSet<Tag>,
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
    /// Create an empty topology with the given dimension
    #[must_use]
    pub fn new(dim: Dim) -> Self {
        Self {
            dim,
            entities: vec![vec![]; (dim + 1) as usize],
            parents: FxHashMap::default(),
        }
    }

    /// Load a topology from a json file.  
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

    /// Save the topology to a json file.
    pub fn to_json(&self, fname: &str) -> Result<()> {
        let mut file = File::create(fname)?;
        writeln!(file, "{}", serde_json::to_string_pretty(self).unwrap())?;
        Ok(())
    }

    /// Return the number of tags for the given dimension
    #[must_use]
    pub fn ntags(&self, dim: Dim) -> usize {
        self.entities[dim as usize].len()
    }

    /// Return the list of tags for the given dimension
    #[must_use]
    pub fn tags(&self, dim: Dim) -> Vec<Tag> {
        self.entities[dim as usize]
            .iter()
            .map(|x| x.tag.1)
            .collect()
    }

    /// Get a tag that is not used for the given dimension.
    #[must_use]
    pub fn get_available_tag(&self, dim: Dim) -> Tag {
        let mut res = 1;
        while self.entities[dim as usize]
            .iter()
            .any(|x| x.tag.1.abs() == res)
        {
            res += 1;
        }
        res
    }

    /// Return the node corresponding to the given tag, if it exists. Panics if the dimension of the tag is invalid.
    #[must_use]
    pub fn get(&self, tag: TopoTag) -> Option<&TopoNode> {
        assert!(tag.0 >= 0, "Invalid dimension");
        assert!(tag.0 <= self.dim, "Invalid dimension");
        self.entities[tag.0 as usize]
            .iter()
            .find(|&e| e.tag.1 == tag.1)
    }

    /// Return the node corresponding to the given tag, if it exists. Panics if the dimension of the tag is invalid.
    fn get_mut(&mut self, tag: TopoTag) -> Option<&mut TopoNode> {
        assert!(tag.0 >= 0, "Invalid dimension");
        assert!(tag.0 <= self.dim + 1, "Invalid dimension");
        self.entities[tag.0 as usize]
            .iter_mut()
            .find(|e| e.tag.1 == tag.1)
    }

    /// Insert a tag with the given parents. Panics if the tag already exists or if one of the parents does not exist.
    pub fn insert(&mut self, tag: TopoTag, parents: impl IntoIterator<Item = Tag>) {
        assert_ne!(tag.1, 0, "tag cannot be 0");
        if let Some(node) = self.get(tag) {
            let existing_parents = &node.parents;
            let parents = parents.into_iter().collect::<Vec<_>>();
            panic!(
                "Tag {tag:?} already exists with parents {existing_parents:?}. parents = {parents:?}"
            );
        }

        let parents = parents.into_iter().collect::<FxHashSet<_>>();
        for &ptag in &parents {
            let e = self.get_mut((tag.0 + 1, ptag));
            assert!(e.is_some(), "TopoTag {:?} not present", (tag.0 + 1, ptag));
            e.unwrap().children.insert(tag.1);
        }
        self.entities[tag.0 as usize].push(TopoNode {
            tag,
            children: FxHashSet::with_hasher(FxBuildHasher),
            parents,
        });
    }

    /// Return the list of nodes with the given parents. Panics if the dimension of the parents is invalid.
    #[must_use]
    pub fn get_from_parents(&self, dim: Dim, parents: &FxHashSet<Tag>) -> Vec<&TopoNode> {
        assert!(dim >= 0, "Invalid dimension");
        assert!(dim <= self.dim, "Invalid dimension");
        self.entities[dim as usize]
            .iter()
            .filter(|&e| e.parents == *parents)
            .collect()
    }

    pub fn get_from_parents_or_create(
        &mut self,
        dim: Dim,
        parents: impl IntoIterator<Item = Tag> + Clone,
    ) -> Result<Tag> {
        let parents: FxHashSet<_> = parents.into_iter().collect();
        let nodes = self.get_from_parents(dim, &parents);
        if nodes.is_empty() {
            let mut t = self.get_available_tag(dim);
            if parents.iter().any(|&x| x < 0) {
                t = -t;
            }
            self.insert((dim, t), parents);
            Ok(t)
        } else if nodes.len() == 1 {
            Ok(nodes[0].tag.1)
        } else {
            let msg = format!("Multiple nodes with parents {parents:?}: {nodes:?}");
            Err(Error::from(&msg))
        }
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
    #[allow(dead_code)]
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

    /// Compute the parents map from the entities. This should be called after all the entities have been inserted.
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

    /// Return the closest common parent of the two given tags, if it exists.
    #[must_use]
    pub fn parent(&self, topo0: TopoTag, topo1: TopoTag) -> Option<TopoTag> {
        self.parents.get(&(topo0, topo1)).copied()
    }

    /// Remove all the tags for which `filter` returns true, and all their children.
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

    fn update_tags<C: Simplex>(
        &mut self,
        f: C,
        f2tag: Option<&FxHashMap<C, Tag>>,
        elems: &tmesh::mesh::twovec::Vec<usize>,
        etag: impl Fn(usize) -> Tag,
        vtags: &mut [TopoTag],
    ) -> Option<Tag> {
        let dim = C::DIM as Dim;

        let etags = elems.iter().map(|&i| etag(i)).collect::<FxHashSet<_>>();
        if elems.len() == 2 && etags.len() == 1 {
            return None; // should not be tagged
        }

        let t = if let Some(f2tag) = f2tag {
            let &t = f2tag
                .get(&f)
                .unwrap_or_else(|| panic!("Face {f:?} not found in f2tag"));
            if let Some(node) = self.get((dim, t)) {
                // If the tag already exists, check that the parents are correct
                assert_eq!(
                    etags, node.parents,
                    "Invalid parents for face tag {t}: {:?} != {:?}",
                    etags, node.parents
                );
            } else {
                // Insert the tag if it does not exist
                self.insert((dim, t), etags);
            }
            t
        } else {
            self.get_from_parents_or_create(dim, etags).unwrap()
        };

        // Set vertex tags. If a vertex is tagged with multiple tags, set it to (dim, 0)
        for i in f {
            if vtags[i].0 > dim {
                vtags[i] = (dim, t);
            } else if vtags[i].1 != t {
                vtags[i] = (dim, 0);
            }
        }
        Some(t)
    }

    // Check if faces share a vertex but not an edge
    fn single_vertex_contact<C: Simplex>(i: usize, faces: impl Iterator<Item = C>) -> bool {
        assert_eq!(C::DIM, 2);

        // we use a set because of HO meshes
        let mut edgs = FxHashSet::with_hasher(FxBuildHasher);
        let mut local_ids = FxHashMap::with_hasher(FxBuildHasher);
        for f in faces {
            for e in f.edges() {
                if !e.contains(i) {
                    let n = local_ids.len();
                    let i0 = *local_ids.entry(e.get(0)).or_insert(n);
                    let n = local_ids.len();
                    let i1 = *local_ids.entry(e.get(1)).or_insert(n);
                    if i0 < i1 {
                        edgs.insert([i0, i1]);
                    } else {
                        edgs.insert([i1, i0]);
                    }
                }
            }
        }

        let g = CSRGraph::from_edges(edgs.iter().copied(), Some(local_ids.len()));

        g.connected_components().unwrap().into_iter().any(|x| x > 0)
    }

    fn check_and_fix_single_vertex_contact<const D: usize, M: Mesh<D>>(
        &mut self,
        i_verts: &[usize],
        mesh: &M,
        vtags: &mut [TopoTag],
    ) {
        let (v2e, v2f) = match M::C::DIM {
            3 => (
                Some(CSRGraph::transpose(mesh.elems(), Some(mesh.n_verts()))),
                CSRGraph::transpose(mesh.faces(), Some(mesh.n_verts())),
            ),
            2 => (
                None,
                CSRGraph::transpose(mesh.elems(), Some(mesh.n_verts())),
            ),
            _ => panic!("Invalid dimension"),
        };

        for &i in i_verts {
            let tris = v2f.row(i);

            match M::C::DIM {
                3 => {
                    if !Self::single_vertex_contact(i, tris.iter().map(|&j| mesh.face(j))) {
                        continue;
                    }
                }
                2 => {
                    if !Self::single_vertex_contact(i, tris.iter().map(|&j| mesh.elem(j))) {
                        continue;
                    }
                }
                _ => unreachable!(),
            }

            let mut ftags: FxHashSet<_> = match M::C::DIM {
                3 => tris.iter().map(|&i| mesh.ftag(i)).collect(),
                2 => tris.iter().map(|&i| mesh.etag(i)).collect(),
                _ => unreachable!(),
            };

            if let Some(v2e) = &v2e {
                let elems = v2e.row(i);
                let etags = elems
                    .iter()
                    .map(|&i| mesh.etag(i))
                    .collect::<FxHashSet<_>>();

                if etags.len() > 1 {
                    // If multiple element tags, check that the interface tag exists or create it, and
                    // add it to the face tags
                    let etags = etags.into_iter().collect::<Vec<_>>();
                    let n = etags.len();
                    for i in 0..n {
                        for j in i + 1..n {
                            ftags.insert(
                                self.get_from_parents_or_create(2, [etags[i], etags[j]])
                                    .unwrap(),
                            );
                        }
                    }
                }
            }

            let (dim, tag) = vtags[i];
            let t0 = if dim == 2 {
                let t1 = self.get_from_parents_or_create(1, ftags).unwrap();
                self.get_from_parents_or_create(0, [t1]).unwrap()
            } else if dim == 1 {
                let mut missing = Vec::new();
                for t in ftags {
                    if !self.is_child((2, t), (1, tag)) {
                        missing.push(t);
                    }
                }
                let t1 = self.get_from_parents_or_create(1, missing).unwrap();
                self.get_from_parents_or_create(0, [t1, tag]).unwrap()
            } else if dim == 0 {
                let node = self.get((0, tag)).unwrap();
                let mut edge_tags = node.parents.clone();
                let mut missing = Vec::new();
                for t in ftags {
                    if !self.is_child((2, t), (0, tag)) {
                        missing.push(t);
                    }
                }
                let t1 = self.get_from_parents_or_create(1, missing).unwrap();
                edge_tags.insert(t1);
                self.get_from_parents_or_create(0, edge_tags).unwrap()
            } else {
                panic!("Invalid dimension")
            };
            vtags[i] = (0, t0);
        }
    }

    pub fn update_from_mesh<const D: usize, M: Mesh<D>>(&mut self, mesh: &M) -> Vec<TopoTag> {
        let mut vtags = vec![(Dim::MAX, 0); mesh.n_verts()];

        // Store the indices of vertices that belong to multiple face tags
        let mut ids = Vec::new();

        // Add element tags
        for (e, t) in mesh.elems().zip(mesh.etags()) {
            if self.get((M::C::DIM as Dim, t)).is_none() {
                self.insert((M::C::DIM as Dim, t), []);
            }
            // Set vertex tags. If a vertex is tagged with multiple tags, set it to (0, 0)
            for i in e {
                if vtags[i].0 == Dim::MAX {
                    vtags[i] = (M::C::DIM as Dim, t);
                } else if vtags[i].1 != t {
                    vtags[i] = (Dim::MAX, 0);
                }
            }
        }

        if M::C::DIM == 2 {
            ids.extend(
                vtags
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.0 == 2)
                    .map(|(i, _)| i),
            );
        }

        // Add face tags
        let face2elems = get_face_to_elem(mesh.elems());
        let face2tag = mesh
            .faces()
            .zip(mesh.ftags())
            .map(|(f, t)| (f.sorted(), t))
            .collect::<FxHashMap<_, _>>();
        for (&f, elems) in &face2elems {
            self.update_tags(f, Some(&face2tag), elems, |i| mesh.etag(i), &mut vtags);
        }

        if M::C::DIM == 3 {
            ids.extend(
                vtags
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.0 == 2)
                    .map(|(i, _)| i),
            );
        }

        match M::C::DIM {
            3 => {
                // Add edge tags and collect the tagged edges
                let mut edges = Vec::new();
                let mut edgtags = Vec::new();
                let edge2faces = get_face_to_elem(mesh.faces());
                for (&edge, faces) in &edge2faces {
                    if let Some(t) =
                        self.update_tags(edge, None, faces, |i| mesh.ftag(i), &mut vtags)
                    {
                        edges.push(edge);
                        edgtags.push(t);
                    }
                }

                // Add node tags and collect the tagged nodes
                let node2edge = get_face_to_elem(edges);
                for (&node, edges) in &node2edge {
                    self.update_tags(node, None, edges, |i| edgtags[i], &mut vtags);
                }

                self.check_and_fix_single_vertex_contact(&ids, mesh, &mut vtags);
            }
            2 => {
                // Add node tags and collect the tagged nodes
                let node2edge = get_face_to_elem(mesh.faces());
                for (&node, edges) in &node2edge {
                    self.update_tags(node, None, edges, |i| mesh.ftag(i), &mut vtags);
                }
                if D == 3 {
                    self.check_and_fix_single_vertex_contact(&ids, mesh, &mut vtags);
                }
            }
            _ => panic!("Invalid elem dimension: {}", M::C::DIM),
        }

        assert!(
            !vtags.iter().any(|&t| t.1 == 0),
            "Some vertices are not tagged"
        );
        self.compute_parents();
        vtags
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
        let vtags = topo.update_from_mesh(msh);
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
        mesh::{GenericMesh, Mesh, Simplex, Tetrahedron, Triangle},
    };

    use crate::{
        Tag,
        mesh::{
            Topology,
            test_meshes::{test_mesh_2d, test_mesh_2d_nobdy, test_mesh_3d},
            topology::MeshTopology,
        },
    };

    #[test]
    fn test_insert() {
        let mut t = Topology::new(3);
        t.insert((3, 1), []);
        t.insert((2, 1), [1]);
        t.insert((2, 2), [1]);
        t.insert((2, 3), [1]);
        t.insert((2, 4), [1]);
        t.insert((1, 1), [1, 2]);
        t.insert((1, 2), [1, 3]);
        t.insert((1, 3), [3, 2]);

        let p = vec![1 as Tag, 2 as Tag];
        let c = t.get_from_parents(1, &(p.into_iter().collect()));
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].tag.1, 1);

        let p = vec![1 as Tag, 3 as Tag];
        let c = t.get_from_parents(1, &(p.into_iter().collect()));
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].tag.1, 2);

        let p = vec![2 as Tag, 3 as Tag];
        let c = t.get_from_parents(1, &(p.into_iter().collect()));
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].tag.1, 3);
    }

    #[test]
    fn test_2d() {
        let mut t = Topology::new(3);
        t.insert((2, 1), []);
        t.insert((2, 2), []);
        t.insert((1, 1), [1]);
        t.insert((1, 2), [1]);
        t.insert((1, 3), [2]);
        t.insert((1, 4), [2]);
        t.insert((1, 5), [1, 2]);
        t.insert((0, 1), [1, 4, 5]);
        t.insert((0, 2), [1, 2]);
        t.insert((0, 3), [2, 3, 5]);
        t.insert((0, 4), [3, 4]);

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
        t.insert((3, 1), []);
        t.insert((3, 1), []);
    }

    #[test]
    #[should_panic]
    fn test_insert_invalid_parent() {
        let mut t = Topology::new(3);
        t.insert((2, 1), [1]);
    }

    #[test]
    #[should_panic]
    fn test_invalid_2d_1() {
        let mut mesh = test_mesh_2d();
        mesh.ftags_mut().for_each(|tag| *tag = 1);
        let _ = MeshTopology::new(&mesh);
    }

    #[test]
    #[should_panic]
    fn test_invalid_2d_2() {
        let mesh = test_mesh_2d_nobdy();
        let _ = MeshTopology::new(&mesh);
    }

    #[test]
    #[should_panic]
    fn test_invalid_2d_3() {
        let mut mesh = test_mesh_2d();
        mesh.ftags_mut()
            .zip([2, 1, 1, 3])
            .for_each(|(tag, v)| *tag = v);
        let _ = MeshTopology::new(&mesh);
    }

    #[test]
    #[should_panic]
    fn test_invalid_2d_4() {
        let mut mesh = test_mesh_2d();
        mesh.ftags_mut()
            .zip([1, 1, 2, 2])
            .for_each(|(tag, v)| *tag = v);
        let _ = MeshTopology::new(&mesh);
    }

    #[test]
    fn test_valid_2d_1() {
        let mut mesh = test_mesh_2d();
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
        let mut mesh = test_mesh_2d();
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
        let mut mesh = test_mesh_2d();
        mesh.fix().unwrap();

        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 4);
        assert_eq!(topo.entities[1].len(), 5);
        assert_eq!(topo.entities[2].len(), 2);
    }

    #[test]
    fn test_valid_2d_2_split() {
        let mut mesh = test_mesh_2d();
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
        let mesh = test_mesh_3d();
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

        assert_eq!(topo.entities[0].len(), 1);
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

        assert_eq!(topo.entities[0].len(), 1);
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

        assert_eq!(topo.entities[0].len(), 1);
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

        assert_eq!(topo.entities[0].len(), 1);
        assert_eq!(topo.entities[1].len(), 1);
        assert_eq!(topo.entities[2].len(), 2);
        assert_eq!(topo.entities[3].len(), 1);
    }

    #[test]
    fn test_corner_3d_3() {
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
        assert_eq!(topo.entities[3].len(), 1);
    }

    #[test]
    fn test_corner_3d_3_split() {
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
        let etags = vec![1, 1];
        let faces = vec![];
        let ftags = vec![];

        let mut mesh = GenericMesh::from_vecs(verts, elems, etags, faces, ftags);
        mesh.fix().unwrap();

        let mesh = mesh.split().split();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 1);
        assert_eq!(topo.entities[1].len(), 1);
        assert_eq!(topo.entities[2].len(), 1);
        assert_eq!(topo.entities[3].len(), 1);
    }

    #[test]
    fn test_corner_3d_4() {
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
        mesh.ftags_mut().enumerate().for_each(|(i, t)| {
            if *t == 2 {
                *t = 10 + i as Tag;
            }
        });

        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 5);
        assert_eq!(topo.entities[1].len(), 7);
        assert_eq!(topo.entities[2].len(), 6);
        assert_eq!(topo.entities[3].len(), 2);
    }

    #[test]
    fn test_corner_3d_4_split() {
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
        mesh.ftags_mut().enumerate().for_each(|(i, t)| {
            if *t == 2 {
                *t = 10 + i as Tag;
            }
        });

        let mesh = mesh.split().split();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 5);
        assert_eq!(topo.entities[1].len(), 7);
        assert_eq!(topo.entities[2].len(), 6);
        assert_eq!(topo.entities[3].len(), 2);
    }

    #[test]
    fn test_corner_3d_5() {
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
        mesh.ftags_mut().enumerate().for_each(|(i, t)| {
            if *t == 2 {
                *t = 10 + i as Tag;
            }
        });

        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 5);
        assert_eq!(topo.entities[1].len(), 7);
        assert_eq!(topo.entities[2].len(), 5);
        assert_eq!(topo.entities[3].len(), 1);
    }

    #[test]
    fn test_corner_3d_5_split() {
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
        mesh.ftags_mut().enumerate().for_each(|(i, t)| {
            if *t == 2 {
                *t = 10 + i as Tag;
            }
        });

        let mesh = mesh.split().split();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 5);
        assert_eq!(topo.entities[1].len(), 7);
        assert_eq!(topo.entities[2].len(), 5);
        assert_eq!(topo.entities[3].len(), 1);
    }

    #[test]
    fn test_corner_3d_6() {
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
        let etags = vec![1, 1];
        let faces = vec![];
        let ftags = vec![];

        let mut mesh = GenericMesh::from_vecs(verts, elems, etags, faces, ftags);
        mesh.fix().unwrap();
        mesh.etags_mut().for_each(|t| *t = 1);
        let faces = mesh.faces().collect::<Vec<_>>();
        mesh.ftags_mut().zip(faces.iter()).for_each(|(t, &f)| {
            if f.contains(0) && f.contains(6) {
                *t = 1;
            } else if !f.contains(1) {
                *t = 3;
            } else {
                *t = 2;
            }
        });

        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 1);
        assert_eq!(topo.entities[1].len(), 2);
        assert_eq!(topo.entities[2].len(), 3);
        assert_eq!(topo.entities[3].len(), 1);
    }

    #[test]
    fn test_corner_3d_6_split() {
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
        let etags = vec![1, 1];
        let faces = vec![];
        let ftags = vec![];

        let mut mesh = GenericMesh::from_vecs(verts, elems, etags, faces, ftags);
        mesh.fix().unwrap();
        mesh.etags_mut().for_each(|t| *t = 1);
        let faces = mesh.faces().collect::<Vec<_>>();
        mesh.ftags_mut().zip(faces.iter()).for_each(|(t, &f)| {
            if f.contains(0) && f.contains(6) {
                *t = 1;
            } else if !f.contains(1) {
                *t = 3;
            } else {
                *t = 2;
            }
        });

        let mesh = mesh.split().split();
        let mtopo = MeshTopology::new(&mesh);
        let topo = mtopo.topo();

        assert_eq!(topo.entities[0].len(), 1);
        assert_eq!(topo.entities[1].len(), 2);
        assert_eq!(topo.entities[2].len(), 3);
        assert_eq!(topo.entities[3].len(), 1);
    }
}
